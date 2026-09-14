"""Stored sparse vectors must share nothing with the caller's objects.

Unlike dense vectors, they used to cross the storage boundary by reference in both directions,
so mutating a submitted or returned vector rewrote the collection. The write path also sorted
the caller's `SparseVector` in place, which remote mode does not do.
"""

import uuid
from pathlib import Path
from typing import Callable, Iterator

import pytest

from qdrant_client import QdrantClient, models

COLLECTION = "test"
POINT_ID = 1
STORED = {"indices": [0], "values": [1.0]}


def make_client(path: str | None = None) -> QdrantClient:
    client = QdrantClient(path=path) if path is not None else QdrantClient(":memory:")
    client.create_collection(
        COLLECTION,
        vectors_config={},
        sparse_vectors_config={"text": models.SparseVectorParams()},
    )
    return client


@pytest.fixture
def client() -> Iterator[QdrantClient]:
    client = make_client()
    try:
        yield client
    finally:
        client.close()


def stored_vector() -> models.SparseVector:
    return models.SparseVector(**STORED)


def mutate(vector: models.SparseVector) -> None:
    vector.indices[0] = 1
    vector.values[0] = 99.0


def assert_collection_untouched(client: QdrantClient) -> None:
    record = client.retrieve(COLLECTION, [POINT_ID], with_vectors=True)[0]
    assert record.vector == {"text": stored_vector()}

    results = client.query_points(COLLECTION, query=stored_vector(), using="text").points
    assert [(point.id, point.score) for point in results] == [(POINT_ID, 1.0)]


def seed_other_vector(client: QdrantClient) -> None:
    client.upsert(
        COLLECTION,
        [
            models.PointStruct(
                id=POINT_ID, vector={"text": models.SparseVector(indices=[2], values=[2.0])}
            )
        ],
    )


# Each writer returns the object the collection was handed.


def write_insert(client: QdrantClient) -> models.SparseVector:
    point = models.PointStruct(id=POINT_ID, vector={"text": stored_vector()})
    client.upsert(COLLECTION, [point])
    return point.vector["text"]


def write_replacing_upsert(client: QdrantClient) -> models.SparseVector:
    seed_other_vector(client)
    return write_insert(client)


def write_batch(client: QdrantClient) -> models.SparseVector:
    batch = models.Batch(ids=[POINT_ID], vectors={"text": [stored_vector()]})
    client.upsert(COLLECTION, batch)
    return batch.vectors["text"][0]


def write_update_vectors(client: QdrantClient) -> models.SparseVector:
    seed_other_vector(client)
    point = models.PointVectors(id=POINT_ID, vector={"text": stored_vector()})
    client.update_vectors(COLLECTION, [point])
    return point.vector["text"]


def write_batch_update_points(client: QdrantClient) -> models.SparseVector:
    point = models.PointStruct(id=POINT_ID, vector={"text": stored_vector()})
    client.batch_update_points(
        COLLECTION,
        [models.UpsertOperation(upsert=models.PointsList(points=[point]))],
    )
    return point.vector["text"]


@pytest.mark.parametrize(
    "write",
    [
        pytest.param(write_insert, id="insert"),
        pytest.param(write_replacing_upsert, id="replacing_upsert"),
        pytest.param(write_batch, id="batch"),
        pytest.param(write_update_vectors, id="update_vectors"),
        pytest.param(write_batch_update_points, id="batch_update_points"),
    ],
)
def test_submitted_vectors_are_detached(
    client: QdrantClient, write: Callable[[QdrantClient], models.SparseVector]
) -> None:
    submitted = write(client)

    mutate(submitted)

    assert_collection_untouched(client)


def read_retrieve(client: QdrantClient) -> models.SparseVector:
    return client.retrieve(COLLECTION, [POINT_ID], with_vectors=True)[0].vector["text"]


def read_scroll(client: QdrantClient) -> models.SparseVector:
    return client.scroll(COLLECTION, with_vectors=True)[0][0].vector["text"]


def read_query(client: QdrantClient) -> models.SparseVector:
    points = client.query_points(
        COLLECTION, query=stored_vector(), using="text", with_vectors=True
    ).points
    return points[0].vector["text"]


@pytest.mark.parametrize(
    "read",
    [
        pytest.param(read_retrieve, id="retrieve"),
        pytest.param(read_scroll, id="scroll"),
        pytest.param(read_query, id="query"),
    ],
)
def test_returned_vectors_are_detached(
    client: QdrantClient, read: Callable[[QdrantClient], models.SparseVector]
) -> None:
    write_insert(client)

    mutate(read(client))

    assert_collection_untouched(client)


def test_upsert_leaves_the_submitted_point_alone(client: QdrantClient) -> None:
    point_id = uuid.uuid4()
    vector = models.SparseVector(indices=[3, 1], values=[3.0, 1.0])
    point = models.PointStruct(id=point_id, vector={"text": vector})

    client.upsert(COLLECTION, [point])

    assert point.id == point_id
    # by value, not identity: pydantic v1 copies submodels on validation, v2 does not
    assert point.vector["text"].indices == [3, 1]
    assert point.vector["text"].values == [3.0, 1.0]
    assert vector.indices == [3, 1]
    assert vector.values == [3.0, 1.0]


def test_unsorted_vectors_are_stored_sorted(tmp_path: Path) -> None:
    """Scoring assumes sorted indices, including after a reload from disk."""
    path = str(tmp_path / "db")
    client = make_client(path)
    try:
        client.upsert(
            COLLECTION,
            [
                models.PointStruct(
                    id=POINT_ID,
                    vector={"text": models.SparseVector(indices=[3, 1], values=[3.0, 1.0])},
                )
            ],
        )
        expected = models.SparseVector(indices=[1, 3], values=[1.0, 3.0])
        assert client.retrieve(COLLECTION, [POINT_ID], with_vectors=True)[0].vector == {
            "text": expected
        }
    finally:
        client.close()

    reopened = QdrantClient(path=path)
    try:
        assert reopened.retrieve(COLLECTION, [POINT_ID], with_vectors=True)[0].vector == {
            "text": expected
        }
        results = reopened.query_points(COLLECTION, query=expected, using="text").points
        assert [(point.id, point.score) for point in results] == [(POINT_ID, 10.0)]
    finally:
        reopened.close()
