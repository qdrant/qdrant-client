from datetime import datetime

from qdrant_client.http.models import OrderValue
from qdrant_client.local.datetime_utils import parse

MICROS_PER_SECOND = 1_000_000


def datetime_to_microseconds(dt: datetime) -> int:
    return int(dt.timestamp() * MICROS_PER_SECOND)


def to_order_value(value: str | datetime | OrderValue | None) -> OrderValue | None:
    if value is None:
        return None

    # check if OrderValue
    # bool is a subclass of int in Python, but bools are never order values on
    # the server: order-by reads exclusively from the numeric index, where bool
    # payloads have no entries, and OrderValue::try_from only accepts integers
    # and floats. The REST model agrees (OrderValue is StrictInt | StrictFloat).
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
