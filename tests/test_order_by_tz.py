from datetime import datetime, timedelta, timezone, tzinfo

from qdrant_client.local.order_by import to_order_value


def test_naive_datetime_is_utc_like_a_naive_string() -> None:
    """A naive datetime means UTC, matching how a naive datetime *string* is parsed.

    Reading it as local time made the order value depend on the client machine's
    timezone, so the same wall clock sorted differently on different machines.
    """
    from_string = to_order_value("2024-06-15 12:30:45")
    from_naive = to_order_value(datetime(2024, 6, 15, 12, 30, 45))
    from_aware = to_order_value(datetime(2024, 6, 15, 12, 30, 45, tzinfo=timezone.utc))

    assert from_string == from_naive == from_aware


def test_aware_datetime_offset_is_respected() -> None:
    ist = timezone(timedelta(hours=5, minutes=30))
    assert to_order_value(datetime(2024, 6, 15, 18, 0, 45, tzinfo=ist)) == to_order_value(
        datetime(2024, 6, 15, 12, 30, 45, tzinfo=timezone.utc)
    )


def test_non_datetime_values_are_unchanged() -> None:
    assert to_order_value(None) is None
    assert to_order_value(42) == 42
    assert to_order_value(1.5) == 1.5
    assert to_order_value("not a date") is None


def test_tzinfo_with_none_utcoffset_is_treated_as_naive() -> None:
    """A tzinfo whose utcoffset() returns None is naive by Python's definition.

    Checking only `tzinfo is None` let such a value through to timestamp(), which
    raised TypeError instead of ordering it.
    """

    class NoOffset(tzinfo):
        def utcoffset(self, dt):
            return None

        def dst(self, dt):
            return None

    value = datetime(2024, 6, 15, 12, 30, 45, tzinfo=NoOffset())
    assert value.tzinfo is not None and value.utcoffset() is None  # naive per datetime docs

    assert to_order_value(value) == to_order_value(
        datetime(2024, 6, 15, 12, 30, 45, tzinfo=timezone.utc)
    )
