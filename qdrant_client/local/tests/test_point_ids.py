from uuid import UUID

import pytest

from qdrant_client import QdrantClient, models


@pytest.mark.parametrize("persistent", [False, True])
@pytest.mark.parametrize("as_batch", [False, True])
@pytest.mark.parametrize("invalid_id", [-1, 2**64])
def test_invalid_integer_id_rejects_entire_upsert(tmp_path, persistent, as_batch, invalid_id):
    options = {"path": str(tmp_path / "db")} if persistent else {"location": ":memory:"}
    client = QdrantClient(**options)
    try:
        client.create_collection(
            "points", vectors_config=models.VectorParams(size=2, distance=models.Distance.DOT)
        )
        client.upsert(
            "points",
            [models.PointStruct(id=1, vector=[1.0, 2.0], payload={"version": "original"})],
        )
        before, _ = client.scroll("points", with_vectors=True)
        ids = [1, 2, invalid_id]
        vectors = [[3.0, 4.0]] * 3
        payloads = [{"version": "changed"}] * 3
        if as_batch:
            points = models.Batch(ids=ids, vectors=vectors, payloads=payloads)
        else:
            points = [
                models.PointStruct(id=id_, vector=vector, payload=payload)
                for id_, vector, payload in zip(ids, vectors, payloads)
            ]

        with pytest.raises(ValueError, match="unsigned 64-bit integer"):
            client.upsert("points", points)

        after, _ = client.scroll("points", with_vectors=True)
        assert after == before
        if persistent:
            client.close()
            client = QdrantClient(**options)
            restored, _ = client.scroll("points", with_vectors=True)
            assert restored == before
    finally:
        client.close()


@pytest.mark.parametrize("persistent", [False, True])
@pytest.mark.parametrize(
    "point_id",
    [
        0,
        2**63 - 1,
        2**63,
        2**64 - 1,
        "550e8400-e29b-41d4-a716-446655440000",
        UUID("550e8400-e29b-41d4-a716-446655440000"),
    ],
)
def test_valid_point_ids_round_trip(tmp_path, persistent, point_id):
    options = {"path": str(tmp_path / "db")} if persistent else {"location": ":memory:"}
    client = QdrantClient(**options)
    try:
        client.create_collection(
            "points", vectors_config=models.VectorParams(size=2, distance=models.Distance.DOT)
        )
        client.upsert("points", [models.PointStruct(id=point_id, vector=[1.0, 2.0])])
        if persistent:
            client.close()
            client = QdrantClient(**options)
        records = client.retrieve("points", [point_id], with_vectors=True)
        assert len(records) == 1
        assert records[0].id == (str(point_id) if isinstance(point_id, UUID) else point_id)
        assert records[0].vector == [1.0, 2.0]
    finally:
        client.close()
