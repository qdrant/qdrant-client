from datetime import datetime, timezone

from qdrant_client.http.models import OrderValue
from qdrant_client.local.datetime_utils import parse

MICROS_PER_SECOND = 1_000_000

_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)


def datetime_to_microseconds(dt: datetime) -> int:
    # Not `int(dt.timestamp() * MICROS_PER_SECOND)`: `timestamp()` returns a float, and
    # both the float itself and multiplying it by 1e6 before truncating lose precision -
    # off by one microsecond for a large fraction of timestamps in testing. The error grows
    # with distance from 1970 (float64 has 52 mantissa bits, so the representable precision
    # shrinks as the magnitude grows): dates near year 9999 can be off by up to ~32
    # microseconds, though never anywhere near a full second.
    #
    # `datetime - datetime` (both aware) is exact integer arithmetic internally (no
    # floats), so route through that instead. A naive `dt` is first attached to the
    # system's local timezone via `astimezone()` - the same "assume local time" behavior
    # `timestamp()` has for naive input - which itself returns a datetime, not a lossy
    # float, so precision is preserved all the way through.
    #
    # `utcoffset() is None` (not `tzinfo is None`) is the correct naive-datetime check: a
    # tzinfo subclass can be attached and still report no offset, and `dt - _EPOCH` raises
    # TypeError if naive and aware datetimes are mixed.
    if dt.utcoffset() is None:
        dt = dt.astimezone()
    delta = dt - _EPOCH
    return (
        delta.days * 86400 * MICROS_PER_SECOND
        + delta.seconds * MICROS_PER_SECOND
        + delta.microseconds
    )


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
