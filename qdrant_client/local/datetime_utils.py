from datetime import datetime, timezone

# These are the formats accepted by qdrant core
available_formats = [
    "%Y-%m-%dT%H:%M:%S.%f%z",
    "%Y-%m-%d %H:%M:%S.%f%z",
    "%Y-%m-%dT%H:%M:%S%z",
    "%Y-%m-%d %H:%M:%S%z",
    # core reads back its own Display output, which puts a space before the offset
    "%Y-%m-%dT%H:%M:%S.%f %z",
    "%Y-%m-%d %H:%M:%S.%f %z",
    "%Y-%m-%dT%H:%M:%S %z",
    "%Y-%m-%d %H:%M:%S %z",
    "%Y-%m-%dT%H:%M:%S.%f",
    "%Y-%m-%d %H:%M:%S.%f",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%dT%H:%M",
    "%Y-%m-%d %H:%M",
    "%Y-%m-%d",
]


def normalize(date_str: str) -> str:
    """Rewrites the leniencies of core's parser that strptime does not share.

    Core skips whitespace ahead of the date, takes a lowercase "z", spells UTC out the way
    chrono's Display does, and keeps up to nanoseconds where "%f" stops at microseconds.
    """
    date_str = date_str.lstrip()

    if date_str.endswith(" UTC"):
        date_str = date_str[: -len(" UTC")] + "+00:00"
    elif date_str.endswith("z"):
        date_str = date_str[:-1] + "Z"

    # microseconds are as fine as datetime gets, so a fraction longer than 6 digits is cut
    # short rather than rejected. The value is then core's, rounded down by less than a
    # microsecond. Anything after the fraction, such as an offset, is kept.
    # dt examples to handle:
    # "2021-01-01T00:00:00.123456789" -> "2021-01-01T00:00:00.123456"
    # "2021-01-01 00:00:00.1234567+01:00" -> "2021-01-01 00:00:00.123456+01:00"
    dot = date_str.find(".")
    if dot != -1:
        end = dot + 1
        while end < len(date_str) and date_str[end].isdigit():
            end += 1
        if end - dot > 7:  # the dot plus more than 6 digits
            date_str = date_str[: dot + 7] + date_str[end:]

    return date_str


def parse(date_str: str) -> datetime | None:
    """Parses one section of the date string at a time.

    Args:
        date_str (str): Accepts any of the formats in qdrant core (see https://github.com/qdrant/qdrant/blob/81d27d9baf13ea43b8c9398b914d36ee160cf60e/lib/segment/src/types.rs#L114-L149)

    Returns:
        Optional[datetime]: the datetime if the string is valid, otherwise None
    """

    def parse_available_formats(datetime_str: str) -> datetime | None:
        for fmt in available_formats:
            try:
                dt = datetime.strptime(datetime_str, fmt)
                if dt.tzinfo is None:
                    # Assume UTC if no timezone is provided
                    dt = dt.replace(tzinfo=timezone.utc)
                return dt
            except ValueError:
                pass
        return None

    date_str = normalize(date_str)

    parsed_dt = parse_available_formats(date_str)
    if parsed_dt is not None:
        return parsed_dt

    # Python can't parse timezones containing only hours (+HH), but it can parse timezones with hours and minutes
    # So we add :00 to the assumed timezone and try parsing it again
    # dt examples to handle:
    # "2021-01-01 00:00:00.000+01"
    # "2021-01-01 00:00:00.000-10"
    #
    # Only strings ending in an hour-only offset get the retry. Appending ":00"
    # unconditionally also completed truncated datetimes, e.g. "2024-06-15 12"
    # became "2024-06-15 12:00" and "2024-06-15T12:30" became
    # "2024-06-15T12:30:00", so local mode accepted values qdrant core rejects.
    offset = date_str[-3:]
    if len(offset) == 3 and offset[0] in "+-" and offset[1:].isdigit():
        return parse_available_formats(date_str + ":00")

    return None
