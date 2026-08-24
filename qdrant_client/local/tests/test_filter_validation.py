import numpy as np
import pytest

from qdrant_client.http import models
from qdrant_client.local.payload_filters import calculate_payload_mask, validate_filter

F = models.FieldCondition
M = models.MatchValue
COND = [F(key="a", match=M(value=1))]


def mask(payload_filter):
    """Run a filter the way local mode does, over one point."""
    return calculate_payload_mask(
        payloads=[{"a": 1}],
        payload_filter=payload_filter,
        ids_inv=[1],
        deleted_per_vector={"": np.array([False])},
    )


def min_should(min_count, conditions=None):
    return models.MinShould(conditions=conditions or COND, min_count=min_count)


@pytest.mark.parametrize("min_count", [0, -1, -5])
def test_min_count_below_one_is_rejected(min_count):
    """The server refuses these - 422 for 0, 400 for negatives.

    Local mode evaluated `matches >= min_count`, so any value <= 0 was
    trivially true for every point and the whole collection came back. A filter
    that silently matches everything is the worst way to be wrong.
    """
    with pytest.raises(ValueError, match="min_count"):
        mask(models.Filter(min_should=min_should(min_count)))


@pytest.mark.parametrize("min_count", [1, 2, 10])
def test_valid_min_count_is_untouched(min_count):
    mask(models.Filter(min_should=min_should(min_count)))


@pytest.mark.parametrize("clause", ["must", "should", "must_not"])
def test_nested_filters_are_validated(clause):
    """A bad min_count inside a nested clause is just as invalid."""
    nested = models.Filter(min_should=min_should(0))
    with pytest.raises(ValueError, match="min_count"):
        mask(models.Filter(**{clause: [nested]}))


def test_nested_inside_min_should_conditions():
    inner = models.Filter(min_should=min_should(0))
    outer = models.Filter(min_should=min_should(1, conditions=[inner]))
    with pytest.raises(ValueError, match="min_count"):
        mask(outer)


def test_deeply_nested():
    bad = models.Filter(min_should=min_should(-2))
    with pytest.raises(ValueError, match="min_count"):
        mask(models.Filter(must=[models.Filter(should=[models.Filter(must=[bad])])]))


def test_filters_without_min_should_are_unaffected():
    validate_filter(models.Filter(must=COND))
    validate_filter(models.Filter(should=[], must=[], must_not=[]))
    validate_filter(models.Filter(must=[models.Filter(must=COND)]))


def test_error_message_names_the_value():
    with pytest.raises(ValueError) as exc:
        validate_filter(models.Filter(min_should=min_should(0)))
    assert "min_count value 0 is invalid" in str(exc.value)
    assert "Must be 1 or larger" in str(exc.value)


def _nested(inner: models.Filter, key: str = "arr") -> models.NestedCondition:
    return models.NestedCondition(nested=models.Nested(key=key, filter=inner))


@pytest.mark.parametrize("clause", ["must", "should", "must_not"])
def test_nested_condition_hides_a_filter(clause: str):
    """A NestedCondition carries a filter, so it has to be walked too.

    check_condition evaluates condition.nested.filter like any other filter, so
    an invalid min_count reached through one is as live as a top-level one.
    Filter and NestedCondition are the only condition types that wrap a filter.
    """
    bad = _nested(models.Filter(min_should=min_should(0)))
    with pytest.raises(ValueError, match="min_count"):
        mask(models.Filter(**{clause: [bad]}))


def test_nested_condition_inside_min_should_conditions():
    bad = _nested(models.Filter(min_should=min_should(-1)))
    with pytest.raises(ValueError, match="min_count"):
        mask(models.Filter(min_should=min_should(1, conditions=[bad])))


def test_nested_conditions_chained():
    bad = models.Filter(min_should=min_should(0))
    with pytest.raises(ValueError, match="min_count"):
        mask(models.Filter(must=[_nested(models.Filter(must=[_nested(bad, "b")]), "a")]))


def test_valid_nested_condition_is_unaffected():
    validate_filter(models.Filter(must=[_nested(models.Filter(must=COND))]))
    validate_filter(models.Filter(must=[_nested(models.Filter(min_should=min_should(1)))]))
