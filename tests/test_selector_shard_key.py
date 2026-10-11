import pytest

from qdrant_client import models
from qdrant_client.http.api.points_api import jsonable_encoder
from qdrant_client.qdrant_remote import QdrantRemote


@pytest.mark.parametrize(
    "embedded, explicit, expected",
    [
        ("tenant-a", "tenant-b", "tenant-b"),
        ("tenant-a", None, "tenant-a"),
        (None, None, None),
        (1, 0, 0),  # falsy explicit key still wins
        (0, None, 0),  # falsy embedded key is kept
    ],
)
def test_rest_selector_shard_key(embedded, explicit, expected):
    selector = models.PointIdsList(points=[1, 2], shard_key=embedded)

    converted = QdrantRemote._try_argument_to_rest_selector(selector, explicit)

    assert converted.shard_key == expected
    assert converted.points == [1, 2]
    assert selector.shard_key == embedded  # caller's selector is not modified


def test_rest_selector_request_body():
    # MatchExcept has an aliased field (`except_` -> `except`), which is lost if nested models
    # are turned into dicts
    filter_ = models.Filter(
        must=[models.FieldCondition(key="color", match=models.MatchExcept(**{"except": ["x"]}))]
    )
    selector = models.FilterSelector(filter=filter_, shard_key="tenant-a")

    converted = QdrantRemote._try_argument_to_rest_selector(selector, "tenant-b")

    expected = models.FilterSelector(filter=filter_, shard_key="tenant-b")
    assert jsonable_encoder(converted) == jsonable_encoder(expected)
