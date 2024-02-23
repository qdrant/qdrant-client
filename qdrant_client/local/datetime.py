import re
from datetime import datetime
from typing import Optional

# These are the formats that should work with just datetime.strptime,
# but it does not handle a variable number of digits in the microseconds section,
# so we have to handle it manually to match what the qdrant core does.
# _expected_formats = [
#     "%Y-%m-%dT%H:%M:%S.%f%z",
#     "%Y-%m-%d %H:%M:%S.%f%z",
#     "%Y-%m-%dT%H:%M:%S.%f%:z",
#     "%Y-%m-%d %H:%M:%S.%f%:z",
#     "%Y-%m-%dT%H:%M:%S.%f",
#     "%Y-%m-%d %H:%M:%S.%f",
#     "%Y-%m-%dT%H:%M:%S%z",
#     "%Y-%m-%d %H:%M:%S%z",
#     "%Y-%m-%dT%H:%M:%S",
#     "%Y-%m-%d %H:%M:%S",
#     "%Y-%m-%d %H:%M",
#     "%Y-%m-%d"
# ]

sep_formats = ["T", " "]

tz_formats = ["%z", "%:z"]


def parse_fmts(date_str: str, fmts: list[str]) -> Optional[datetime]:
    for fmt in fmts:
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            pass
    return None


def parse(date_str: str) -> Optional[datetime]:
    """Parses one section of the date string at a time.

    Args:
        date_str (str): Accepts any of the formats in qdrant core (see https://github.com/qdrant/qdrant/blob/0ed86ce0575d35930268db19e1f7680287072c58/lib/segment/src/types.rs#L1388-L1410)

    Returns:
        Optional[datetime]: the datetime if the string is valid, otherwise None
    """

    dt = None

    # try to parse date section
    if len(date_str) < 10:
        return None

    dt = parse_fmts(date_str[:10], ["%Y-%m-%d"])

    if dt is None:
        return None

    remaining = date_str[10:]
    if len(remaining) == 0:
        return dt

    # try to parse separator
    sep = None
    for s in sep_formats:
        if remaining.startswith(s):
            remaining = remaining[1:]
            sep = s
            break

    if sep is None or len(remaining) == 0:
        # invalid date formats: invalid separator, or separator without hour
        return None

    # try to parse hours and minutes section
    if len(remaining) < 5:
        return None
    hour_dt = parse_fmts(remaining[:5], ["%H:%M"])

    if hour_dt is None:
        return None

    dt = dt.replace(hour=hour_dt.hour, minute=hour_dt.minute)

    remaining = remaining[5:]
    if len(remaining) == 0:
        return dt

    # try to parse seconds section
    if len(remaining) < 3:
        return None
    if remaining[0] == ":":
        secs_dt = parse_fmts(remaining[1:3], ["%S"])
        if secs_dt is None:
            return None
        dt = dt.replace(second=secs_dt.second)
        remaining = remaining[3:]
        if len(remaining) == 0:
            return dt

    # try to parse decimals section
    if remaining[0] == ".":
        remaining = remaining[1:]
        if len(remaining) == 0:
            return dt

        match = re.match(r"(\d{1,6})", remaining)
        if match is None:
            return None

        micros_str = match.group(1).ljust(6, "0")  # pad to 6 digits
        micros = int(micros_str)
        dt = dt.replace(microsecond=micros)

        remaining = remaining[match.end() :]  # skip past the matched digits
        if len(remaining) == 0:
            return dt

    # try to parse timezone
    tz_dt = parse_fmts(remaining, tz_formats)
    if tz_dt is None:
        return None

    dt = dt.replace(tzinfo=tz_dt.tzinfo)

    return dt
