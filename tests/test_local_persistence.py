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


def ingest_multivector_data(
    vector_size: int = 32,
    tokens: int = 8,
    path: str | None = None,
    collection_name: str = default_collection_name,
):
    client = QdrantClient(path=path)

    if client.collection_exists(collection_name):
        client.delete_collection(collection_name)
    client.create_collection(
        collection_name,
        vectors_config=rest.VectorParams(
            size=vector_size,
            distance=rest.Distance.COSINE,
            multivector_config=rest.MultiVectorConfig(
                comparator=rest.MultiVectorComparator.MAX_SIM
            ),
        ),
    )
    client.upsert(
        collection_name=collection_name,
        points=[
            rest.PointStruct(id=i, vector=np.random.randn(tokens, vector_size).tolist())
            for i in range(10)
        ],
    )
    return client


def test_dense_cosine_persistence_is_stable():
    """Reopening a cosine collection must not change its vectors or its scores.

    Cosine vectors are unit-normalized on upsert, but `load_vectors` handed the
    raw persisted vectors straight to the collection. A reopened collection
    returned un-normalized vectors, scored differently from the same collection
    before it was closed, and got silently normalized in place by the first
    search that touched it.
    """
    vector_size = 32
    with tempfile.TemporaryDirectory() as tmpdir:
        client = ingest_dense_vector_data(vector_size=vector_size, path=tmpdir)
        query = np.random.randn(vector_size).tolist()

        ids = sorted(point.id for point in client.scroll(default_collection_name, limit=10)[0])
        retrieve = lambda c: [
            point.vector
            for point in sorted(
                c.retrieve(default_collection_name, ids, with_vectors=True),
                key=lambda point: point.id,
            )
        ]
        search = lambda c: [
            (point.id, point.score)
            for point in c.query_points(default_collection_name, query=query, limit=10).points
        ]

        before_vectors, before_scores = retrieve(client), search(client)
        before_searched_vectors = retrieve(client)
        client.close()

        client = QdrantClient(path=tmpdir)
        after_vectors = retrieve(client)
        after_scores = search(client)
        after_searched_vectors = retrieve(client)

        assert after_vectors == before_vectors
        assert after_scores == before_scores
        assert after_searched_vectors == before_searched_vectors
        for vector in after_vectors:
            assert np.isclose(np.linalg.norm(vector), 1.0)
        client.close()


def test_multivector_cosine_persistence_is_stable():
    """Same as the dense case, for multivectors."""
    vector_size, tokens = 32, 8
    with tempfile.TemporaryDirectory() as tmpdir:
        client = ingest_multivector_data(vector_size=vector_size, tokens=tokens, path=tmpdir)
        query = np.random.randn(3, vector_size).tolist()

        ids = list(range(10))
        retrieve = lambda c: [
            point.vector
            for point in sorted(
                c.retrieve(default_collection_name, ids, with_vectors=True),
                key=lambda point: point.id,
            )
        ]
        search = lambda c: [
            (point.id, point.score)
            for point in c.query_points(default_collection_name, query=query, limit=10).points
        ]

        before_vectors, before_scores = retrieve(client), search(client)
        before_searched_vectors = retrieve(client)
        client.close()

        client = QdrantClient(path=tmpdir)
        after_vectors = retrieve(client)
        after_scores = search(client)
        after_searched_vectors = retrieve(client)

        assert after_vectors == before_vectors
        assert after_scores == before_scores
        assert after_searched_vectors == before_searched_vectors
        for multivector in after_vectors:
            assert np.allclose(np.linalg.norm(multivector, axis=-1), 1.0)
        client.close()


def test_upsert_over_an_existing_point_matches_a_fresh_insert():
    """Overwriting a point must store exactly what inserting it fresh would store.

    `_add_point` casts to float32 before normalizing, `_update_point` normalized
    first and cast after, so the same vector landed in the collection with
    different values depending on which path wrote it.
    """
    vector_size, tokens = 32, 8
    vector = {
        "dense": np.random.randn(vector_size).tolist(),
        "multi": np.random.randn(tokens, vector_size).tolist(),
    }
    with tempfile.TemporaryDirectory() as tmpdir:
        client = QdrantClient(path=tmpdir)
        client.create_collection(
            default_collection_name,
            vectors_config={
                "dense": rest.VectorParams(size=vector_size, distance=rest.Distance.COSINE),
                "multi": rest.VectorParams(
                    size=vector_size,
                    distance=rest.Distance.COSINE,
                    multivector_config=rest.MultiVectorConfig(
                        comparator=rest.MultiVectorComparator.MAX_SIM
                    ),
                ),
            },
        )
        # id 1 is inserted once, id 2 is inserted and then overwritten
        client.upsert(
            default_collection_name,
            points=[rest.PointStruct(id=i, vector=vector) for i in (1, 2)],
        )
        client.upsert(default_collection_name, points=[rest.PointStruct(id=2, vector=vector)])

        inserted, updated = sorted(
            client.retrieve(default_collection_name, [1, 2], with_vectors=True),
            key=lambda point: point.id,
        )
        assert inserted.vector == updated.vector
        client.close()


def test_zero_norm_cosine_vector_survives_reload():
    """A cosine vector of zero norm must come back as it went in.

    Normalizing it would divide by zero, so the write path leaves it alone. The
    reload path has to make the same exception.
    """
    vector_size = 32
    with tempfile.TemporaryDirectory() as tmpdir:
        client = QdrantClient(path=tmpdir)
        client.create_collection(
            default_collection_name,
            vectors_config=rest.VectorParams(size=vector_size, distance=rest.Distance.COSINE),
        )
        client.upsert(
            default_collection_name,
            points=[
                rest.PointStruct(id=1, vector=[0.0] * vector_size),
                rest.PointStruct(id=2, vector=np.random.randn(vector_size).tolist()),
            ],
        )
        before = client.retrieve(default_collection_name, [1], with_vectors=True)[0].vector
        client.close()

        client = QdrantClient(path=tmpdir)
        after = client.retrieve(default_collection_name, [1], with_vectors=True)[0].vector
        assert after == before
        assert not np.isnan(after).any()
        # the point must still be searchable, not poison the whole collection
        assert (
            len(client.query_points(default_collection_name, query=[1.0] * vector_size).points)
            == 2
        )
        client.close()
