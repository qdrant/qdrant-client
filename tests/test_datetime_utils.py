from datetime import datetime, timezone

import pytest

from qdrant_client.local.datetime_utils import parse


@pytest.mark.parametrize(
    "date_str, expected",
    [
        ("2024-06-15", datetime(2024, 6, 15, tzinfo=timezone.utc)),
        ("2024-06-15 12:30", datetime(2024, 6, 15, 12, 30, tzinfo=timezone.utc)),
        ("2024-06-15 12:30:45", datetime(2024, 6, 15, 12, 30, 45, tzinfo=timezone.utc)),
        ("2024-06-15T12:30:45", datetime(2024, 6, 15, 12, 30, 45, tzinfo=timezone.utc)),
        (
            "2024-06-15T12:30:45.500",
            datetime(2024, 6, 15, 12, 30, 45, 500000, tzinfo=timezone.utc),
        ),
        (
            "2024-06-15T12:30:45+05:00",
            datetime(2024, 6, 15, 7, 30, 45, tzinfo=timezone.utc),
        ),
    ],
)
def test_accepted_formats_still_parse(date_str: str, expected: datetime) -> None:
    parsed = parse(date_str)
    assert parsed is not None
    assert parsed == expected


@pytest.mark.parametrize(
    "date_str",
    [
        "2021-01-01 00:00:00.000+01",
        "2021-01-01 00:00:00.000-10",
        "2024-06-15T12:30:45+05",
    ],
)
def test_hour_only_offset_is_completed(date_str: str) -> None:
    """The ``+HH`` completion the fallback exists for must keep working."""
    parsed = parse(date_str)
    assert parsed is not None
    assert parsed.utcoffset() is not None


@pytest.mark.parametrize(
    "date_str",
    [
        # `%Y-%m-%d %H` is not an accepted qdrant format; the `:00` retry turned
        # it into `2024-06-15 12:00`, which is.
        "2024-06-15 12",
        # `T` with minute precision is not accepted either; the retry turned it
        # into `2024-06-15T12:30:00`, which matches `%Y-%m-%dT%H:%M:%S`.
        "2024-06-15T12:30",
    ],
)
def test_incomplete_datetimes_are_rejected(date_str: str) -> None:
    """The ``:00`` retry is only for hour-only UTC offsets. Appending it to any
    unparsed string made the local client accept datetimes qdrant core rejects,
    so a value that errors server-side silently succeeded in local mode."""
    assert parse(date_str) is None


def test_garbage_is_still_rejected() -> None:
    assert parse("not a date") is None
    assert parse("") is None
