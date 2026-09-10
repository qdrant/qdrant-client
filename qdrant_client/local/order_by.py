from datetime import datetime, timezone

from qdrant_client.http.models import OrderValue
from qdrant_client.local.datetime_utils import parse

MICROS_PER_SECOND = 1_000_000


def datetime_to_microseconds(dt: datetime) -> int:
    if dt.utcoffset() is None:
        # A naive datetime is assumed to be UTC, same as `datetime_utils.parse` does for a
        # datetime string without an offset, and same as qdrant core does. Otherwise
        # `timestamp()` would read it as the client machine's local time, making order
        # values depend on where the client runs.
        # Note: `utcoffset() is None` is the canonical naive check, it also covers a tzinfo
        # whose `utcoffset()` returns None, which `timestamp()` raises on.
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * MICROS_PER_SECOND)


def to_order_value(value: str | datetime | OrderValue | None) -> OrderValue | None:
    if value is None:
        return None

    # check if OrderValue
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return value

    if isinstance(value, datetime):
        return datetime_to_microseconds(value)

    if isinstance(value, str):
        dt = parse(value)
        if dt is not None:
            return datetime_to_microseconds(dt)

    return None
