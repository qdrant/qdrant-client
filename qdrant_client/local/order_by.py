from datetime import datetime, timezone

from qdrant_client.http.models import OrderValue
from qdrant_client.local.datetime_utils import parse

MICROS_PER_SECOND = 1_000_000


def datetime_to_microseconds(dt: datetime) -> int:
    return int(dt.timestamp() * MICROS_PER_SECOND)


def to_order_value(value: str | datetime | OrderValue | None) -> OrderValue | None:
    if value is None:
        return None

    # check if OrderValue
    if isinstance(value, (int, float)):
        return value

    if isinstance(value, datetime):
        if value.tzinfo is None:
            # A naive datetime means UTC — the same assumption `parse()` makes for a
            # datetime string with no offset, and the one qdrant core makes. Without
            # this, `timestamp()` reads it as local time, so the same wall clock
            # sorts differently depending on the client machine's timezone.
            value = value.replace(tzinfo=timezone.utc)
        return datetime_to_microseconds(value)

    if isinstance(value, str):
        dt = parse(value)
        if dt is not None:
            return datetime_to_microseconds(dt)

    return None
