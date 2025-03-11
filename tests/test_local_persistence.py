import random
import tempfile
from typing import Optional

import numpy as np
import pytest

import qdrant_client
import qdrant_client.http.models as rest
from qdrant_client._pydantic_compat import construct
from tests.fixtures.points import generate_random_sparse_vector_list

default_collection_name = "example"


def ingest_dense_vector_data(
    vector_size: int = 1500,
    path: Optional[str] = None,
    collection_name: str = default_collection_name,
):
    lines = [x for x in range(10)]

    embeddings = np.random.randn(len(lines), vector_size).tolist()
    client = qdrant_client.QdrantClient(path=path)

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


def ingest_sparse_vector_data(
    vector_count: int = 10,
    max_vector_size: int = 100,
    path: Optional[str] = None,
    collection_name: str = default_collection_name,
    add_dense_to_config: bool = False,
):
    sparse_vectors = generate_random_sparse_vector_list(vector_count, max_vector_size, 0.2)
    client = qdrant_client.QdrantClient(path=path)

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
        _client = qdrant_client.QdrantClient(path=tmpdir)

        with pytest.raises(Exception) as e:
            _client2 = qdrant_client.QdrantClient(path=tmpdir)

        assert "already accessed by another instance" in str(e)


def test_local_dense_persistence():
    with tempfile.TemporaryDirectory() as tmpdir:
        ingest_dense_vector_data(path=tmpdir)
        client = qdrant_client.QdrantClient(path=tmpdir)
        assert client.count(default_collection_name).count == 10
        del client

        ingest_dense_vector_data(path=tmpdir)
        client = qdrant_client.QdrantClient(path=tmpdir)
        assert client.count(default_collection_name).count == 10
        del client

        ingest_dense_vector_data(path=tmpdir)
        ingest_dense_vector_data(path=tmpdir, collection_name="example_2")
        client = qdrant_client.QdrantClient(path=tmpdir)
        assert client.count(default_collection_name).count == 10
        assert client.count("example_2").count == 10


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

        del client

        client = qdrant_client.QdrantClient(path=tmpdir)

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

        del client

        ingest_sparse_vector_data(path=tmpdir)
        client = qdrant_client.QdrantClient(path=tmpdir)
        assert client.count(default_collection_name).count == 10
        del client

        ingest_sparse_vector_data(path=tmpdir)
        ingest_sparse_vector_data(path=tmpdir, collection_name="example_2")
        client = qdrant_client.QdrantClient(path=tmpdir)
        assert client.count(default_collection_name).count == 10
        assert client.count("example_2").count == 10
