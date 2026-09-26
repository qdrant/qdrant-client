import os
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


@pytest.mark.parametrize("operation", ["points", "vectors"])
def test_idf_persistence_after_deletion(operation: str):
    """Reopening a collection must not move the scores a deletion left behind.

    IDF statistics are maintained incrementally while the client is open and rebuilt from the
    stored points when it reopens. A deletion that misses the live statistics therefore scores
    one way before a restart and another way after.
    """
    collection_name = "idf_persistence"
    vector = rest.SparseVector(indices=[0], values=[1.0])

    def scores(client: QdrantClient) -> list[float]:
        collected = []
        for idf in (rest.IdfScope.GLOBAL, rest.IdfCorpusParams(corpus=rest.Filter())):
            points = client.query_points(
                collection_name,
                using="text",
                query=vector,
                search_params=rest.SearchParams(idf=idf),
            ).points
            collected.append([point.score for point in points])
        assert collected[0] == pytest.approx(
            collected[1]
        ), "the global scope disagrees with a corpus of every live point"
        return collected[0]

    with tempfile.TemporaryDirectory() as tmpdir:
        client = QdrantClient(path=tmpdir)
        client.create_collection(
            collection_name,
            vectors_config={},
            sparse_vectors_config={"text": rest.SparseVectorParams(modifier=rest.Modifier.IDF)},
        )
        client.upsert(
            collection_name,
            [rest.PointStruct(id=i, vector={"text": vector}) for i in range(4)],
        )

        if operation == "points":
            client.delete(collection_name, points_selector=[1, 2, 3])
        else:
            client.delete_vectors(collection_name, vectors=["text"], points=[1, 2, 3])

        before_reopen = scores(client)
        assert len(before_reopen) == 1
        client.close()

        client = QdrantClient(path=tmpdir)
        assert scores(client) == pytest.approx(
            before_reopen
        ), "reopening the collection changed the IDF scores"
        # deleting the vectors leaves the points themselves in place
        surviving = [0] if operation == "points" else [0, 1, 2, 3]
        assert [point.id for point in client.scroll(collection_name, limit=10)[0]] == surviving
        client.close()


def test_alias_persistence():
    """Alias changes must survive a restart, and a rejected batch of them must leave no trace.

    Regression: a rejected batch kept the operations before the rejected one in memory, so the
    next save of any kind wrote them to disk.
    """

    def aliases(client: QdrantClient) -> list[tuple[str, str]]:
        return [
            (alias.alias_name, alias.collection_name) for alias in client.get_aliases().aliases
        ]

    def create_alias(collection_name: str, alias_name: str) -> rest.CreateAliasOperation:
        return rest.CreateAliasOperation(
            create_alias=rest.CreateAlias(collection_name=collection_name, alias_name=alias_name)
        )

    missing_rename = rest.RenameAliasOperation(
        rename_alias=rest.RenameAlias(old_alias_name="missing", new_alias_name="other")
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        client = QdrantClient(path=tmpdir)
        client.create_collection("docs_v1", vectors_config={})
        client.create_collection("docs_v2", vectors_config={})
        client.update_collection_aliases([create_alias("docs_v1", "live")])

        # the rename is rejected, so `live` must not switch to docs_v2 either
        with pytest.raises(ValueError):
            client.update_collection_aliases([create_alias("docs_v2", "live"), missing_rename])
        # any successful write saves the aliases along with the collections
        client.create_collection("unrelated", vectors_config={})
        client.close()

        client = QdrantClient(path=tmpdir)
        assert aliases(client) == [("live", "docs_v1")]
        client.update_collection_aliases([create_alias("docs_v2", "live")])
        client.close()

        client = QdrantClient(path=tmpdir)
        assert aliases(client) == [("live", "docs_v2")]
        client.close()


def test_failed_alias_save_keeps_aliases():
    """A batch whose save fails must not change the aliases in memory either."""
    with tempfile.TemporaryDirectory() as tmpdir:
        client = QdrantClient(path=tmpdir)
        client.create_collection("docs", vectors_config={})

        # a directory in place of meta.json makes the next save fail
        meta_path = os.path.join(tmpdir, "meta.json")
        os.remove(meta_path)
        os.mkdir(meta_path)

        with pytest.raises(OSError):
            client.update_collection_aliases(
                [
                    rest.CreateAliasOperation(
                        create_alias=rest.CreateAlias(collection_name="docs", alias_name="live")
                    )
                ]
            )

        assert client.get_aliases().aliases == []
        client.close()
