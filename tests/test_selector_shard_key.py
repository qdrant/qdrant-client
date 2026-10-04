import pytest

from qdrant_client import models
from qdrant_client.async_qdrant_remote import AsyncQdrantRemote
from qdrant_client.qdrant_remote import QdrantRemote


@pytest.mark.parametrize(
    "selector",
    [
        pytest.param(models.PointIdsList(points=[1, 2, 3], shard_key="tenant-a"), id="point_ids"),
        pytest.param(
            models.FilterSelector(filter=models.Filter(), shard_key="tenant-a"), id="filter"
        ),
    ],
)
@pytest.mark.parametrize("converter", [QdrantRemote, AsyncQdrantRemote], ids=["sync", "async"])
def test_rest_selector_conversion_does_not_mutate_the_caller(
    converter: type, selector: models.PointsSelector
) -> None:
    """A caller's selector must come back unchanged.

    An application that holds a selector as a constant and passes ``shard_key_selector``
    per request would otherwise have that constant rewritten, so the next request would
    go to whichever shard the previous call used.
    """
    converted = converter._try_argument_to_rest_selector(selector, "tenant-b")

    assert converted.shard_key == "tenant-b", "the request must use the explicit shard key"
    assert selector.shard_key == "tenant-a", "the caller's selector must not be rewritten"
    assert converted is not selector, "the conversion must not hand back the caller's object"


@pytest.mark.parametrize("converter", [QdrantRemote, AsyncQdrantRemote], ids=["sync", "async"])
def test_rest_selector_conversion_keeps_the_embedded_key_when_none_is_given(
    converter: type,
) -> None:
    """With no explicit key the embedded one must still be used, not dropped."""
    selector = models.PointIdsList(points=[1, 2, 3], shard_key="tenant-a")

    converted = converter._try_argument_to_rest_selector(selector, None)

    assert converted.shard_key == "tenant-a"


@pytest.mark.parametrize("converter", [QdrantRemote, AsyncQdrantRemote], ids=["sync", "async"])
def test_rest_selector_conversion_handles_a_selector_without_a_key(converter: type) -> None:
    """A selector with no shard key at all must still convert."""
    selector = models.PointIdsList(points=[1, 2, 3])

    converted = converter._try_argument_to_rest_selector(selector, None)

    assert converted.shard_key is None
    assert selector.shard_key is None


def test_grpc_selector_conversion_already_leaves_the_caller_alone() -> None:
    """The gRPC path is the contrast: it returns the embedded key rather than writing it."""
    selector = models.PointIdsList(points=[1, 2, 3], shard_key="tenant-a")

    _converted, embedded = QdrantRemote._try_argument_to_grpc_selector(selector)

    assert selector.shard_key == "tenant-a"
    assert embedded is not None, "the embedded key is surfaced, not applied to the input"


def test_conversion_preserves_the_rest_of_the_selector() -> None:
    """Copying must not drop the fields that identify which points to delete."""
    selector = models.PointIdsList(points=[4, 5, 6], shard_key="tenant-a")

    converted = QdrantRemote._try_argument_to_rest_selector(selector, "tenant-b")

    assert list(converted.points) == [4, 5, 6]


def test_a_falsy_shard_key_still_wins_over_the_embedded_one() -> None:
    """`0` is a valid shard key, so the guard must not treat it as missing.

    `test_shard_key_from_points_selector` asserts this against a live cluster; this
    pins the same rule without needing one.
    """
    selector = models.PointIdsList(points=[1, 2], shard_key=1)

    converted = QdrantRemote._try_argument_to_rest_selector(selector, 0)

    assert converted.shard_key == 0, "an explicit falsy shard key must win"
    assert selector.shard_key == 1, "and the caller's selector is still untouched"


def test_a_falsy_embedded_shard_key_is_preserved() -> None:
    selector = models.PointIdsList(points=[1], shard_key=0)

    converted = QdrantRemote._try_argument_to_rest_selector(selector, None)

    assert converted.shard_key == 0
