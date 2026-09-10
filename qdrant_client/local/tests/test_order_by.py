from qdrant_client.http import models
from qdrant_client.local.order_by import to_order_value
from qdrant_client.local.qdrant_local import QdrantLocal

COLLECTION_NAME = "test_order_by_bool"


def make_client(points: list[models.PointStruct]) -> QdrantLocal:
    client = QdrantLocal(":memory:")
    client.create_collection(COLLECTION_NAME, vectors_config={})
    client.upsert(COLLECTION_NAME, points=points)
    return client


def bool_mixed_points() -> list[models.PointStruct]:
    return [
        models.PointStruct(id=1, vector={}, payload={"k": True}),
        models.PointStruct(id=2, vector={}, payload={"k": 1}),
        models.PointStruct(id=3, vector={}, payload={"k": False}),
        models.PointStruct(id=4, vector={}, payload={"k": 0}),
    ]


def test_to_order_value_rejects_bool():
    """`isinstance(True, int)` is True, but a bool payload is never an OrderValue.

    The server reads order-by values exclusively from the numeric index
    (`filtered_read_by_index_ordered` / `filtered_read_by_value_stream` via
    `numeric_index_for`), where bool payloads have no entries, and
    `OrderValue::try_from(Value)` only accepts `as_i64` / `as_f64` (both None
    for `Bool`). The REST model agrees: `OrderValue` is `StrictInt | StrictFloat`
    and rejects bools at validation.
    """
    assert to_order_value(True) is None
    assert to_order_value(False) is None
    assert to_order_value(1) == 1
    assert to_order_value(0) == 0
    assert to_order_value(1.5) == 1.5
    assert to_order_value(None) is None


def test_scroll_order_by_skips_bool_points():
    """Bool points carry no ordering value, so they are left out, not ordered as 0/1.

    Pre-fix this crashes local mode with a pydantic ValidationError, because the
    bool order value fails `Record.order_value` (`StrictInt | StrictFloat`) validation.
    """
    client = make_client(bool_mixed_points())
    records, _ = client.scroll(COLLECTION_NAME, limit=10, order_by=models.OrderBy(key="k"))
    assert [(r.id, r.order_value) for r in records] == [(4, 0), (2, 1)]


def test_scroll_order_by_desc_skips_bool_points():
    client = make_client(bool_mixed_points())
    records, _ = client.scroll(
        COLLECTION_NAME,
        limit=10,
        order_by=models.OrderBy(key="k", direction=models.Direction.DESC),
    )
    assert [(r.id, r.order_value) for r in records] == [(2, 1), (4, 0)]
