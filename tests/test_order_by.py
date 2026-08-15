import os
import time
from datetime import datetime, timedelta, timezone

import pytest

from qdrant_client.local.order_by import to_order_value

# 2024-06-15 12:30:45 UTC
WALL_CLOCK = (2024, 6, 15, 12, 30, 45)
EXPECTED_MICROS = 1718454645000000


def test_naive_datetime_object_is_interpreted_as_utc() -> None:
    """A datetime with no tzinfo means UTC, matching how qdrant core reads a
    datetime string with no offset."""
    assert to_order_value(datetime(*WALL_CLOCK)) == EXPECTED_MICROS


def test_naive_object_string_and_aware_utc_agree() -> None:
    """The same wall-clock time must order identically however it is spelled."""
    naive_object = to_order_value(datetime(*WALL_CLOCK))
    naive_string = to_order_value("2024-06-15 12:30:45")
    aware_utc = to_order_value(datetime(*WALL_CLOCK, tzinfo=timezone.utc))

    assert naive_object == naive_string == aware_utc


def test_aware_non_utc_datetime_is_converted_not_stripped() -> None:
    """An explicit offset is still honoured -- naive-means-UTC must not turn
    into treating every datetime as UTC."""
    aware = datetime(*WALL_CLOCK, tzinfo=timezone(timedelta(hours=5, minutes=30)))

    assert to_order_value(aware) == EXPECTED_MICROS - int(5.5 * 3600 * 1_000_000)


@pytest.mark.skipif(
    not hasattr(time, "tzset"),
    reason="process timezone is not settable on this platform",
)
def test_order_value_does_not_depend_on_the_client_timezone() -> None:
    """Without this, the naive-object path went through datetime.timestamp(),
    which reads the machine's local time, so the same call produced different
    order values on differently-configured clients."""
    original = os.environ.get("TZ")
    try:
        values = []
        for zone in ("UTC", "Asia/Kolkata", "America/Los_Angeles"):
            os.environ["TZ"] = zone
            time.tzset()
            values.append(to_order_value(datetime(*WALL_CLOCK)))
        assert values == [EXPECTED_MICROS] * 3
    finally:
        if original is None:
            os.environ.pop("TZ", None)
        else:
            os.environ["TZ"] = original
        time.tzset()
