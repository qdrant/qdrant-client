from datetime import datetime
from typing import Any, Optional, Union, get_args

from qdrant_client.local.datetime_utils import parse

MICROS_PER_SECOND = 1_000_000

OrderingValue = Union[int, float]


def datetime_to_microseconds(dt: datetime) -> int:
    return int(dt.timestamp() * MICROS_PER_SECOND)


def to_ordering_value(value: Optional[Any]) -> Optional[OrderingValue]:
    if value is None:
        return None

    if isinstance(value, get_args(OrderingValue)):
        return value

    if isinstance(value, datetime):
        return datetime_to_microseconds(value)

    if isinstance(value, str):
        dt = parse(value)
        if dt is not None:
            return datetime_to_microseconds(dt)

    return None
