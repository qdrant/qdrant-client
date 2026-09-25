"""Offline regression cases for issue #1477; no provider or server required."""

import uuid
import copy
import pickle
import sqlite3
from contextlib import closing

import pytest

from qdrant_client import AsyncQdrantClient, QdrantClient, models
from qdrant_client.local.persistence import CollectionPersistence


CANONICAL = "936da01f-9abd-4d9d-80c7-02af85c822a8"
SPELLINGS = [
    CANONICAL,
    CANONICAL.upper(),
    CANONICAL.replace("-", ""),
    "urn:uuid:" + CANONICAL,
    uuid.UUID(CANONICAL),
]


def create_collection(client):
    client.create_collection(
        "points", vectors_config=models.VectorParams(size=2, distance=models.Distance.DOT)
    )


@pytest.fixture
def client():
    with closing(QdrantClient(":memory:")) as client:
        create_collection(client)
        client.upsert("points", [models.PointStruct(id=CANONICAL, vector=[1.0, 0.0])])
        yield client


@pytest.mark.parametrize("spelling", SPELLINGS)
def test_upsert_same_identity(client, spelling):
    client.upsert("points", [models.PointStruct(id=spelling, vector=[0.0, 1.0])])
    assert client.count("points").count == 1
    assert client.retrieve("points", [CANONICAL], with_vectors=True)[0].vector == [0.0, 1.0]


@pytest.mark.parametrize("spelling", SPELLINGS)
def test_retrieve_same_identity(client, spelling):
    assert len(client.retrieve("points", [spelling])) == 1


@pytest.mark.parametrize("spelling", SPELLINGS)
def test_delete_same_identity(client, spelling):
    client.delete("points", models.PointIdsList(points=[spelling]))
    assert client.count("points").count == 0


@pytest.mark.parametrize("spelling", SPELLINGS)
def test_filter_same_identity(client, spelling):
    condition = models.Filter(must=[models.HasIdCondition(has_id=[spelling])])
    assert client.count("points", count_filter=condition).count == 1


@pytest.mark.parametrize("spelling", SPELLINGS)
def test_payload_same_identity(client, spelling):
    client.set_payload("points", {"updated": True}, points=[spelling])
    assert client.retrieve("points", [CANONICAL])[0].payload == {"updated": True}


def test_scroll_uuid_object_offset(client):
    client.upsert("points", [models.PointStruct(id=str(uuid.UUID(int=1)), vector=[1.0, 0.0])])
    _, offset = client.scroll("points", limit=1)
    assert offset is not None
    page, _ = client.scroll("points", limit=1, offset=uuid.UUID(offset))
    assert [point.id for point in page] == [CANONICAL]


@pytest.mark.parametrize("spelling", SPELLINGS)
def test_identity_survives_reopen(tmp_path, spelling):
    with closing(QdrantClient(path=str(tmp_path))) as client:
        create_collection(client)
        client.upsert("points", [models.PointStruct(id=spelling, vector=[1.0, 0.0])])
    with closing(QdrantClient(path=str(tmp_path))) as client:
        assert len(client.retrieve("points", [CANONICAL])) == 1
        client.delete("points", models.PointIdsList(points=[CANONICAL]))
    with closing(QdrantClient(path=str(tmp_path))) as client:
        assert client.count("points").count == 0


@pytest.mark.parametrize("spelling", SPELLINGS)
def test_update_and_delete_vectors(client, spelling):
    point = models.PointVectors(id=spelling, vector={"": [0.0, 1.0]})
    original = copy.deepcopy(point)
    client.update_vectors("points", [point])
    assert point == original
    assert client.retrieve("points", [CANONICAL], with_vectors=True)[0].vector == [0.0, 1.0]
    client.delete_vectors("points", vectors=[""], points=[spelling])
    assert client.retrieve("points", [CANONICAL], with_vectors=True)[0].vector == {}


@pytest.mark.parametrize("spelling", SPELLINGS)
def test_query_by_id_excludes_itself(client, spelling):
    client.upsert("points", [models.PointStruct(id=1, vector=[0.0, 1.0])])
    result = client.query_points("points", query=spelling).points
    assert [point.id for point in result] == [1]


@pytest.mark.parametrize("spelling", SPELLINGS)
def test_inputs_unchanged_and_nested_filter(client, spelling):
    point = models.PointStruct(id=spelling, vector=[1.0, 0.0])
    condition = models.Filter(
        must=[models.Filter(should=[models.HasIdCondition(has_id=[spelling])])]
    )
    original = copy.deepcopy((point, condition))
    client.upsert("points", [point])
    assert client.count("points", count_filter=condition).count == 1
    assert (point, condition) == original


def test_integer_id_is_not_uuid(client):
    zero_uuid = str(uuid.UUID(int=1))
    client.upsert(
        "points",
        [
            models.PointStruct(id=1, vector=[1.0, 0.0]),
            models.PointStruct(id=zero_uuid, vector=[0.0, 1.0]),
        ],
    )
    assert client.count("points").count == 3
    client.delete("points", [1])
    assert len(client.retrieve("points", [zero_uuid])) == 1


def test_invalid_uuid_still_rejected_on_upsert(client):
    with pytest.raises(ValueError, match="not a valid UUID"):
        client.upsert("points", [models.PointStruct(id="not-a-uuid", vector=[1.0, 0.0])])
    assert client.count("points").count == 1


def test_group_lookup_non_uuid_payload_remains_missing(client):
    client.set_payload("points", {"group": "not-a-uuid"}, points=[CANONICAL])
    result = client.query_points_groups(
        "points", query=[1.0, 0.0], group_by="group", with_lookup="points"
    )
    assert len(result.groups) == 1
    assert result.groups[0].lookup is None


def seed_legacy_storage(tmp_path, ids):
    """Write the pre-fix on-disk format, bypassing the new normalization."""
    with closing(QdrantClient(path=str(tmp_path))) as client:
        create_collection(client)
    database = tmp_path / "collection" / "points" / "storage.sqlite"
    with closing(sqlite3.connect(database)) as storage, storage:
        for index, point_id in enumerate(ids):
            point = models.PointStruct(id=point_id, vector=[1.0, 0.0], payload={"index": index})
            storage.execute(
                "INSERT INTO points VALUES (?, ?)",
                (
                    CollectionPersistence.encode_key(point_id),
                    pickle.dumps(point),
                ),
            )
    return database


def read_rows(database):
    with closing(sqlite3.connect(database)) as storage:
        return storage.execute("SELECT id, point FROM points ORDER BY id").fetchall()


@pytest.mark.parametrize("spelling", SPELLINGS[1:])
@pytest.mark.parametrize("operation", ["delete", "update", "upsert"])
def test_legacy_storage_keys_do_not_resurrect(tmp_path, spelling, operation):
    database = seed_legacy_storage(tmp_path, [spelling])
    before = read_rows(database)
    with closing(QdrantClient(path=str(tmp_path))) as client:
        assert client.retrieve("points", [CANONICAL])[0].id == CANONICAL
        assert read_rows(database) == before  # Opening is not a destructive migration.
        if operation == "delete":
            client.delete("points", [CANONICAL])
        elif operation == "update":
            client.set_payload("points", {"updated": True}, points=[CANONICAL])
        else:
            client.upsert("points", [models.PointStruct(id=CANONICAL, vector=[0.0, 1.0])])
    with closing(QdrantClient(path=str(tmp_path))) as client:
        assert client.count("points").count == (0 if operation == "delete" else 1)
        if operation == "update":
            assert client.retrieve("points", [CANONICAL])[0].payload["updated"] is True
        elif operation == "upsert":
            assert client.retrieve("points", [CANONICAL], with_vectors=True)[0].vector == [
                0.0,
                1.0,
            ]
    rows = read_rows(database)
    if operation != "delete":
        assert rows[0][0] == CollectionPersistence.encode_key(CANONICAL)
    assert len(rows) == (0 if operation == "delete" else 1)


@pytest.mark.parametrize("ids", [[CANONICAL, CANONICAL.upper()], [CANONICAL.upper(), CANONICAL]])
def test_conflicting_legacy_ids_leave_storage_untouched(tmp_path, ids):
    database = seed_legacy_storage(tmp_path, ids)
    before = read_rows(database)
    with pytest.raises(ValueError, match="Duplicate UUID identity"):
        QdrantClient(path=str(tmp_path))
    assert read_rows(database) == before


def test_legacy_key_migration_is_transactional(tmp_path):
    database = seed_legacy_storage(tmp_path, [CANONICAL.upper()])
    before = read_rows(database)
    storage = CollectionPersistence(str(database.parent))
    try:
        point = list(storage.load())[0]
        storage.storage.execute(
            "CREATE TRIGGER reject_insert BEFORE INSERT ON points "
            "BEGIN SELECT RAISE(ABORT, 'injected failure'); END"
        )
        with pytest.raises(sqlite3.IntegrityError, match="injected failure"):
            storage.persist(point)
        assert read_rows(database) == before
        storage.storage.execute("DROP TRIGGER reject_insert")
        storage.delete(CANONICAL)
        assert read_rows(database) == []
    finally:
        storage.close()


@pytest.mark.asyncio
async def test_async_client_shares_uuid_identity():
    client = AsyncQdrantClient(":memory:")
    try:
        await client.create_collection(
            "points", vectors_config=models.VectorParams(size=2, distance=models.Distance.DOT)
        )
        await client.upsert(
            "points", [models.PointStruct(id=CANONICAL.upper(), vector=[1.0, 0.0])]
        )
        assert len(await client.retrieve("points", [CANONICAL])) == 1
        await client.delete("points", [CANONICAL.replace("-", "")])
        assert (await client.count("points")).count == 0
    finally:
        await client.close()
