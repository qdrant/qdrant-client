from qdrant_client.http import models
from qdrant_client.local.qdrant_local import QdrantLocal

COLLECTION_NAME = "test_facet"


def facet_of(points: list[models.PointStruct]) -> list[tuple[type, models.FacetValue, int]]:
    client = QdrantLocal(":memory:")
    client.create_collection(COLLECTION_NAME, vectors_config={})
    client.upsert(COLLECTION_NAME, points=points)

    hits = client.facet(COLLECTION_NAME, key="a").hits

    return [(type(hit.value), hit.value, hit.count) for hit in hits]


def test_facet_keeps_scalar_types_distinct():
    """`False == 0` and `True == 1` in python, but they are distinct facet values.

    A facet on the server reads a single payload index, so a bool index only ever
    returns bools and an integer index only ever returns ints. Local mode facets the
    raw payload, so it has to keep the types apart on its own.
    """
    # every bucket ties at one point, so this also pins the tie-break order
    assert facet_of(
        [
            models.PointStruct(id=1, vector={}, payload={"a": [False, 0]}),
            models.PointStruct(id=2, vector={}, payload={"a": True}),
            models.PointStruct(id=3, vector={}, payload={"a": 1}),
            models.PointStruct(id=4, vector={}, payload={"a": "0"}),
        ]
    ) == [
        (bool, False, 1),
        (bool, True, 1),
        (int, 0, 1),
        (int, 1, 1),
        (str, "0", 1),
    ]


def test_facet_does_not_compare_values_of_different_types():
    """Ties on count fall through to the value, and `"a" < 1` raises in python."""
    assert facet_of(
        [
            models.PointStruct(id=1, vector={}, payload={"a": 1}),
            models.PointStruct(id=2, vector={}, payload={"a": "a"}),
        ]
    ) == [
        (int, 1, 1),
        (str, "a", 1),
    ]


def test_facet_normalizes_uuids_before_deduplicating():
    """Both spellings of one uuid are a single value, counted once for the point.

    This matches a uuid index on the server. Local mode has no index awareness, so it
    normalizes unconditionally; a keyword index on the server would instead keep the
    two spellings as two separate values.
    """
    assert facet_of(
        [
            models.PointStruct(
                id=1,
                vector={},
                payload={
                    "a": [
                        "550e8400e29b41d4a716446655440000",
                        "550e8400-e29b-41d4-a716-446655440000",
                    ]
                },
            ),
        ]
    ) == [(str, "550e8400-e29b-41d4-a716-446655440000", 1)]
