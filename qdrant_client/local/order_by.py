from datetime import datetime
from typing import Any, Optional, Union

from dateutil.parser import parse

MICROS_PER_SECOND = 1_000_000


class OrderingValue:
    def __init__(self, value: Union[int, float]):
        self.value = value

    def __lt__(self, other: "OrderingValue") -> bool:
        return self.value < other.value

    def __gt__(self, other: "OrderingValue") -> bool:
        return self.value > other.value

    def __ge__(self, other: "OrderingValue") -> bool:
        return self.value >= other.value

    def __le__(self, other: "OrderingValue") -> bool:
        return self.value <= other.value


def datetime_to_microseconds(dt: datetime) -> int:
    return int(dt.timestamp() * MICROS_PER_SECOND)


def to_ordering_value(value: Optional[Any]) -> Optional[OrderingValue]:
    if value is None:
        return None

    if isinstance(value, (int, float)):
        return OrderingValue(value)

    if isinstance(value, str):
        # dateutil parser also parses "now", but qdrant core does not
        if value == "now":
            return None

        try:
            dt = parse(value)
            return OrderingValue(datetime_to_microseconds(dt))
        except:
            # ignore unparsable datetime
            pass

    return None
