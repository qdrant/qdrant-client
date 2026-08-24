import re
from datetime import datetime, timezone

# An hour-only UTC offset at the end of the string, e.g. the "+01" in
# "2021-01-01 00:00:00.000+01". Python can parse "+HH:MM" but not "+HH", so
# these are completed with ":00" and parsed again.
hour_only_offset = re.compile(r"\d{2}:\d{2}(:\d{2})?([.,]\d+)?[+-]\d{2}$")

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


def parse(date_str: str) -> datetime | None:
    """Parses one section of the date string at a time.

    Args:
        date_str (str): Accepts any of the formats in qdrant core (see https://github.com/qdrant/qdrant/blob/0ed86ce0575d35930268db19e1f7680287072c58/lib/segment/src/types.rs#L1388-L1410)

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

    parsed_dt = parse_available_formats(date_str)
    if parsed_dt is not None:
        return parsed_dt

    # Python can't parse timezones containing only hours (+HH), but it can parse timezones with hours and minutes
    # So we add :00 to the assumed timezone and try parsing it again
    # dt examples to handle:
    # "2021-01-01 00:00:00.000+01"
    # "2021-01-01 00:00:00.000-10"
    #
    # Only strings that actually end in an hour-only offset get the retry.
    # Appending ":00" unconditionally also completed truncated datetimes --
    # "2024-06-15 12" became "2024-06-15 12:00" and "2024-06-15T12:30" became
    # "2024-06-15T12:30:00" -- so local mode accepted values qdrant core rejects.
    if hour_only_offset.search(date_str):
        return parse_available_formats(date_str + ":00")

    return None
