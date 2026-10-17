from datetime import date, datetime, timezone

from qdrant_client.http.models import OrderValue
from qdrant_client.local.datetime_utils import parse

MICROS_PER_SECOND = 1_000_000
MICROS_PER_DAY = 24 * 60 * 60 * MICROS_PER_SECOND

_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)


def datetime_to_microseconds(dt: datetime) -> int:
    # `utcoffset() is None`, not `tzinfo is None`: a tzinfo can be attached and still report
    # no offset, and the subtraction below rejects a naive datetime.
    if dt.utcoffset() is None:
        # A naive datetime means UTC, same as `datetime_utils.parse` and qdrant core. Reading
        # it as local time, which `timestamp()` and `astimezone()` both do, would make order
        # values depend on where the client runs.
        dt = dt.replace(tzinfo=timezone.utc)

    # Exact, unlike `int(dt.timestamp() * MICROS_PER_SECOND)`, which truncates a float that
    # has already lost precision - by up to ~32us near year 9999.
    delta = dt - _EPOCH
    return delta.days * MICROS_PER_DAY + delta.seconds * MICROS_PER_SECOND + delta.microseconds


def to_order_value(value: str | date | datetime | OrderValue | None) -> OrderValue | None:
    if value is None:
        return None

    # check if OrderValue
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return value

    if isinstance(value, datetime):
        return datetime_to_microseconds(value)

    if isinstance(value, date):
        # Must stay below the datetime branch: datetime is a subclass of date. Midnight is
        # naive, so `datetime_to_microseconds` reads it as UTC, like core reads "%Y-%m-%d".
        return datetime_to_microseconds(datetime.combine(value, datetime.min.time()))

    if isinstance(value, str):
        dt = parse(value)
        if dt is not None:
            return datetime_to_microseconds(dt)

    return None
