from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.local.local_collection import DEFAULT_VECTOR_NAME, LocalCollection

COLLECTION_NAME = "test_groups_bool_keys"


def _points() -> list[models.PointStruct]:
    return [
        models.PointStruct(id=1, vector=[1.0, 0.0], payload={"g": True}),
        models.PointStruct(id=2, vector=[1.0, 0.0], payload={"g": 1}),
        models.PointStruct(id=3, vector=[1.0, 0.0], payload={"g": False}),
        models.PointStruct(id=4, vector=[1.0, 0.0], payload={"g": 0}),
        models.PointStruct(id=5, vector=[1.0, 0.0], payload={"g": "1"}),
        models.PointStruct(id=6, vector=[1.0, 0.0], payload={"g": 1.5}),
        models.PointStruct(id=7, vector=[1.0, 0.0], payload={"other": "x"}),
    ]


def test_query_groups_skips_bool_keys():
    """Bool payload values must not form groups, mirroring the server.

    The server's `GroupId::try_from` rejects `Bool` (`BadKeyType`, point ignored),
    so `True`/`1` and `False`/`0` are never merged. Local mode used
    `isinstance(v, (str, int))`, which admits bools, and `set()` then collapsed
    them with the equal int. Same root cause as #1259 (filters) and #1389 (facet).
    """
    client = QdrantClient(location=":memory:")
    client.create_collection(
        COLLECTION_NAME,
        vectors_config=models.VectorParams(size=2, distance=models.Distance.COSINE),
    )
    client.upsert(COLLECTION_NAME, points=_points())

    res = client.query_points_groups(
        COLLECTION_NAME, group_by="g", query=[1.0, 0.0], limit=10, group_size=10
    )

    by_id = {grp.id: sorted(p.id for p in grp.hits) for grp in res.groups}
    assert by_id == {1: [2], 0: [4], "1": [5]}


def test_search_groups_skips_bool_keys():
    """Same contract for the legacy `search_groups` path (twin hunk)."""
    collection = LocalCollection(
        models.CreateCollection(
            vectors=models.VectorParams(size=2, distance=models.Distance.COSINE)
        )
    )
    collection.upsert(points=_points())

    res = collection.search_groups(
        query_vector=(DEFAULT_VECTOR_NAME, [1.0, 0.0]),
        group_by="g",
        limit=10,
        group_size=10,
    )

    by_id = {grp.id: sorted(p.id for p in grp.hits) for grp in res.groups}
    assert by_id == {1: [2], 0: [4], "1": [5]}
