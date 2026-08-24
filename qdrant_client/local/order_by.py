from datetime import datetime, timezone

from qdrant_client.http.models import OrderValue
from qdrant_client.local.datetime_utils import parse

MICROS_PER_SECOND = 1_000_000


def datetime_to_microseconds(dt: datetime) -> int:
    if dt.tzinfo is None:
        # Assume UTC if no timezone is provided, matching `datetime_utils.parse`
        # and qdrant core. `datetime.timestamp()` would otherwise read a naive
        # datetime as local time, making order values depend on where the
        # client runs.
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * MICROS_PER_SECOND)


def to_order_value(value: str | datetime | OrderValue | None) -> OrderValue | None:
    if value is None:
        return None

    # check if OrderValue
    if isinstance(value, (int, float)):
        return value

    if isinstance(value, datetime):
        return datetime_to_microseconds(value)

    if isinstance(value, str):
        dt = parse(value)
        if dt is not None:
            return datetime_to_microseconds(dt)

    return None
