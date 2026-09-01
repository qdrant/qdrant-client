"""Local mode must mirror the server's IsNull semantics: a value is null if it
is null itself, or is an array containing a null element, one level deep
(qdrant/qdrant#10101)."""

from qdrant_client import QdrantClient
from qdrant_client.http import models


def count_is_null(client: QdrantClient, key: str) -> int:
    return client.count(
        "test",
        count_filter=models.Filter(must=[models.IsNullCondition(is_null=models.PayloadField(key=key))]),
    ).count


def test_is_null_matches_null_inside_arrays() -> None:
    client = QdrantClient(location=":memory:")
    client.create_collection(
        "test", vectors_config=models.VectorParams(size=2, distance=models.Distance.COSINE)
    )
    client.upsert(
        "test",
        points=[
            models.PointStruct(id=1, vector=[0.0, 1.0], payload={"a": [None, 1]}),
            models.PointStruct(id=2, vector=[0.0, 1.0], payload={"a": [1, None]}),
            models.PointStruct(id=3, vector=[0.0, 1.0], payload={"a": [1, 2]}),
            models.PointStruct(id=4, vector=[0.0, 1.0], payload={"a": None}),
            models.PointStruct(id=5, vector=[0.0, 1.0], payload={"a": [[None]]}),  # too deep
            models.PointStruct(
                id=6,
                vector=[0.0, 1.0],
                payload={"nested": [{"empty": [None]}, {"empty": [None]}]},
            ),
        ],
    )
    assert count_is_null(client, "a") == 3  # ids 1, 2, 4
    assert count_is_null(client, "nested[].empty") == 1  # id 6
