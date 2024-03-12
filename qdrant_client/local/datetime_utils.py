from datetime import datetime, timezone
from typing import Optional

# These are the formats accepted by qdrant core
available_formats = [
    "%Y-%m-%dT%H:%M:%S.%f%z",
    "%Y-%m-%d %H:%M:%S.%f%z",
    "%Y-%m-%dT%H:%M:%S%z",
    "%Y-%m-%d %H:%M:%S%z",
    "%Y-%m-%dT%H:%M:%S.%f",
    "%Y-%m-%d %H:%M:%S.%f",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d %H:%M",
    "%Y-%m-%d",
]


def parse(date_str: str) -> Optional[datetime]:
    """Parses one section of the date string at a time.

    Args:
        date_str (str): Accepts any of the formats in qdrant core (see https://github.com/qdrant/qdrant/blob/0ed86ce0575d35930268db19e1f7680287072c58/lib/segment/src/types.rs#L1388-L1410)

    Returns:
        Optional[datetime]: the datetime if the string is valid, otherwise None
    """

    for fmt in available_formats:
        try:
            dt = datetime.strptime(date_str, fmt)
            if dt.tzinfo is None:
                # Assume UTC if no timezone is provided
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except ValueError:
            pass

    return None
