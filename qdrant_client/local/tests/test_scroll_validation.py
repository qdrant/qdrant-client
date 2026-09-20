import pytest

from qdrant_client import QdrantClient, models


def _client(with_points: bool) -> QdrantClient:
    client = QdrantClient(":memory:")
    client.create_collection(
        "test", vectors_config=models.VectorParams(size=2, distance=models.Distance.DOT)
    )
    if with_points:
        client.upsert(
            "test",
            points=[models.PointStruct(id=1, vector=[0.1, 0.2], payload={"k": 1})],
        )
    return client


@pytest.mark.parametrize("with_points", [False, True])
def test_scroll_rejects_offset_with_order_by(with_points: bool) -> None:
    """`offset` combined with `order_by` is rejected whether or not data exists yet.

    The server rejects the combination before it looks at the stored points, and local
    mode did the same, but only once the collection held at least one point: the
    empty-collection fast path returned an empty page instead. A caller therefore sees a
    result on an empty collection and the documented error on the very same call after
    the first upsert.
    """
    client = _client(with_points)

    with pytest.raises(ValueError, match="Offset is not supported"):
        client.scroll("test", limit=3, order_by=models.OrderBy(key="k"), offset=1)
