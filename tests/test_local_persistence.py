import random
import tempfile

import numpy as np
import pytest

from qdrant_client import QdrantClient
import qdrant_client.http.models as rest
from qdrant_client._pydantic_compat import construct
from tests.fixtures.points import generate_random_sparse_vector_list

default_collection_name = "example"


def ingest_dense_vector_data(
    vector_size: int = 1500,
    path: str | None = None,
    collection_name: str = default_collection_name,
):
    lines = [x for x in range(10)]

    embeddings = np.random.randn(len(lines), vector_size).tolist()
    client = QdrantClient(path=path)

    if client.collection_exists(collection_name):
        client.delete_collection(collection_name)
    client.create_collection(
        collection_name,
        vectors_config=rest.VectorParams(
            size=vector_size,
            distance=rest.Distance.COSINE,
        ),
    )

    client.upsert(
        collection_name=collection_name,
        points=construct(
            rest.Batch,
            ids=random.sample(range(100), len(lines)),
            vectors=embeddings,
        ),
    )
    return client


def ingest_sparse_vector_data(
    vector_count: int = 10,
    max_vector_size: int = 100,
    path: str | None = None,
    collection_name: str = default_collection_name,
    add_dense_to_config: bool = False,
):
    sparse_vectors = generate_random_sparse_vector_list(vector_count, max_vector_size, 0.2)
    client = QdrantClient(path=path)

    if client.collection_exists(collection_name):
        client.delete_collection(collection_name)
    client.create_collection(
        collection_name,
        vectors_config={}
        if not add_dense_to_config
        else rest.VectorParams(size=1500, distance=rest.Distance.COSINE),
        sparse_vectors_config={
            "text": rest.SparseVectorParams(),
        },
    )

    batch = construct(
        rest.Batch,
        ids=random.sample(range(100), vector_count),
        vectors={"text": sparse_vectors},
    )

    client.upsert(
        collection_name=collection_name,
        points=batch,
    )

    return client


def test_prevent_parallel_access():
    with tempfile.TemporaryDirectory() as tmpdir:
        _client = QdrantClient(path=tmpdir)

        with pytest.raises(Exception) as e:
            _client2 = QdrantClient(path=tmpdir)

        assert "already accessed by another instance" in str(e)


def test_local_dense_persistence():
    with tempfile.TemporaryDirectory() as tmpdir:
        client = ingest_dense_vector_data(path=tmpdir)
        assert client.count(default_collection_name).count == 10
        client.close()

        client = ingest_dense_vector_data(path=tmpdir)
        assert client.count(default_collection_name).count == 10
        client.close()

        client = ingest_dense_vector_data(path=tmpdir)
        client.close()

        client = ingest_dense_vector_data(path=tmpdir, collection_name="example_2")
        assert client.count(default_collection_name).count == 10
        assert client.count("example_2").count == 10

        client.close()


@pytest.mark.parametrize("add_dense_to_config", [True, False])
def test_local_sparse_persistence(add_dense_to_config):
    with tempfile.TemporaryDirectory() as tmpdir:
        client = ingest_sparse_vector_data(path=tmpdir, add_dense_to_config=add_dense_to_config)
        assert client.count(default_collection_name).count == 10

        (post_result, _) = client.scroll(
            collection_name=default_collection_name,
            limit=10,
            with_vectors=True,
        )
        client.close()

        client = QdrantClient(path=tmpdir)

        (pre_result, _) = client.scroll(
            collection_name=default_collection_name,
            limit=10,
            with_vectors=True,
        )

        for i in range(len(pre_result)):
            assert pre_result[i].vector["text"] == post_result[i].vector["text"]
            assert len(pre_result[i].vector["text"].indices) > 0
            assert len(pre_result[i].vector["text"].values) > 0
            assert len(pre_result[i].vector["text"].indices) == len(
                pre_result[i].vector["text"].values
            )
        client.close()

        client = ingest_sparse_vector_data(path=tmpdir)
        assert client.count(default_collection_name).count == 10
        client.close()

        client = ingest_sparse_vector_data(path=tmpdir)
        client.close()
        client = ingest_sparse_vector_data(path=tmpdir, collection_name="example_2")
        assert client.count(default_collection_name).count == 10
        assert client.count("example_2").count == 10
        client.close()


def test_update_persistence():
    collection_name = "update_persistence"
    with tempfile.TemporaryDirectory() as tmpdir:
        client = QdrantClient(path=tmpdir)

        if client.collection_exists(collection_name):
            client.delete_collection(collection_name)

        client.create_collection(
            collection_name,
            vectors_config={"dense": rest.VectorParams(size=20, distance=rest.Distance.COSINE)},
            sparse_vectors_config={
                "text": rest.SparseVectorParams(),
            },
            metadata={"important": "meta information"},
        )

        original_collection_info = client.get_collection(collection_name)

        assert original_collection_info.config.params.sparse_vectors["text"].modifier is None
        assert original_collection_info.config.metadata == {"important": "meta information"}

        client.update_collection(
            collection_name,
            sparse_vectors_config={"text": rest.SparseVectorParams(modifier=rest.Modifier.IDF)},
            metadata={"not_important": "missing"},
        )
        updated_collection_info = client.get_collection(collection_name)
        assert (
            updated_collection_info.config.params.sparse_vectors["text"].modifier
            == rest.Modifier.IDF
        )
        assert updated_collection_info.config.metadata == {
            "important": "meta information",
            "not_important": "missing",
        }

        client.close()

        client = QdrantClient(path=tmpdir)
        persisted_collection_info = client.get_collection(collection_name)
        assert (
            persisted_collection_info.config.params.sparse_vectors["text"].modifier
            == rest.Modifier.IDF
        )
        assert persisted_collection_info.config.metadata == {
            "important": "meta information",
            "not_important": "missing",
        }
        client.close()
