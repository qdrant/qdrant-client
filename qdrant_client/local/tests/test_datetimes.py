from datetime import datetime, timedelta, timezone, tzinfo

import pytest

from qdrant_client.local.datetime_utils import parse
from qdrant_client.local.order_by import datetime_to_microseconds


@pytest.mark.parametrize(  # type: ignore
    "date_str, expected",
    [
        ("2021-01-01T00:00:00", datetime(2021, 1, 1, 0, 0, 0, tzinfo=timezone.utc)),
        ("2021-01-01T00:00:00Z", datetime(2021, 1, 1, 0, 0, 0, tzinfo=timezone.utc)),
        ("2021-01-01T00:00:00+00:00", datetime(2021, 1, 1, 0, 0, 0, tzinfo=timezone.utc)),
        ("2021-01-01T00:00:00.000000", datetime(2021, 1, 1, 0, 0, 0, tzinfo=timezone.utc)),
        ("2021-01-01T00:00:00.000000Z", datetime(2021, 1, 1, 0, 0, 0, tzinfo=timezone.utc)),
        (
            "2021-01-01T00:00:00.000000+01:00",
            datetime(2021, 1, 1, 0, 0, 0, tzinfo=timezone(timedelta(hours=1))),
        ),
        (
            "2021-01-01T00:00:00.000000-10:00",
            datetime(2021, 1, 1, 0, 0, 0, tzinfo=timezone(timedelta(hours=-10))),
        ),
        ("2021-01-01", datetime(2021, 1, 1, 0, 0, 0, tzinfo=timezone.utc)),
        ("2021-01-01 00:00:00", datetime(2021, 1, 1, 0, 0, 0, tzinfo=timezone.utc)),
        ("2021-01-01 00:00:00Z", datetime(2021, 1, 1, 0, 0, 0, tzinfo=timezone.utc)),
        (
            "2021-01-01 00:00:00+0200",
            datetime(2021, 1, 1, 0, 0, 0, tzinfo=timezone(timedelta(hours=2))),
        ),
        ("2021-01-01 00:00:00.000000", datetime(2021, 1, 1, 0, 0, 0, tzinfo=timezone.utc)),
        ("2021-01-01 00:00:00.000000Z", datetime(2021, 1, 1, 0, 0, 0, tzinfo=timezone.utc)),
        (
            "2021-01-01 00:00:00.000000+00:30",
            datetime(2021, 1, 1, 0, 0, 0, tzinfo=timezone(timedelta(minutes=30))),
        ),
        (
            "2021-01-01 00:00:00.000009+00:30",
            datetime(2021, 1, 1, 0, 0, 0, 9, tzinfo=timezone(timedelta(minutes=30))),
        ),
        # this is accepted in core but not here, there is no specifier for only-hour offset
        (
            "2021-01-01 00:00:00.000+01",
            datetime(2021, 1, 1, 0, 0, 0, tzinfo=timezone(timedelta(hours=1))),
        ),
        (
            "2021-01-01 00:00:00.000-10",
            datetime(2021, 1, 1, 0, 0, 0, tzinfo=timezone(timedelta(hours=-10))),
        ),
        ("2021-01-01T00:00", datetime(2021, 1, 1, 0, 0, 0, tzinfo=timezone.utc)),
        # core reads these too: a lowercase "z", whitespace ahead of the date, and the two
        # shapes chrono's Display writes
        ("2021-01-01T00:00:00z", datetime(2021, 1, 1, 0, 0, 0, tzinfo=timezone.utc)),
        (" 2021-01-01T00:00:00", datetime(2021, 1, 1, 0, 0, 0, tzinfo=timezone.utc)),
        ("2021-01-01 00:00:00 UTC", datetime(2021, 1, 1, 0, 0, 0, tzinfo=timezone.utc)),
        (
            "2021-01-01 00:00:00.123 +01:00",
            datetime(2021, 1, 1, 0, 0, 0, 123000, tzinfo=timezone(timedelta(hours=1))),
        ),
        (
            "2021-01-01 00:00:00 +0530",
            datetime(2021, 1, 1, 0, 0, 0, tzinfo=timezone(timedelta(hours=5, minutes=30))),
        ),
        # core keeps nanoseconds, datetime stops at microseconds, so a longer fraction is
        # cut short rather than rejected
        (
            "2021-01-01T00:00:00.123456789",
            datetime(2021, 1, 1, 0, 0, 0, 123456, tzinfo=timezone.utc),
        ),
        (
            "2021-01-01 00:00:00.1234567+01:00",
            datetime(2021, 1, 1, 0, 0, 0, 123456, tzinfo=timezone(timedelta(hours=1))),
        ),
        (
            "2021-01-01T00:00:00+05",
            datetime(2021, 1, 1, 0, 0, 0, tzinfo=timezone(timedelta(hours=5))),
        ),
        (
            "2021-01-01 00:00:00-03:00",
            datetime(2021, 1, 1, 0, 0, 0, tzinfo=timezone(timedelta(hours=-3))),
        ),
    ],
)
def test_parse_dates(date_str: str, expected: datetime):
    assert parse(date_str) == expected


@pytest.mark.parametrize(  # type: ignore
    "date_str",
    [
        # an hour on its own is not an accepted format in core, but the
        # hour-only offset fallback used to complete it into one
        "2021-01-01 00",
        "not a date",
        "",
    ],
)
def test_parse_unsupported_dates(date_str: str):
    assert parse(date_str) is None


def test_tzinfo_without_an_offset_counts_as_naive() -> None:
    """A datetime is aware only when `utcoffset()` returns an offset, so a tzinfo returning
    None is naive and gets UTC too. `timestamp()` raised TypeError on these."""

    class NoOffset(tzinfo):
        def utcoffset(self, dt: datetime | None) -> timedelta | None:
            return None

        def dst(self, dt: datetime | None) -> timedelta | None:
            return None

        def tzname(self, dt: datetime | None) -> str | None:
            return None

    assert datetime_to_microseconds(
        datetime(2024, 6, 15, 12, 30, 45, tzinfo=NoOffset())
    ) == datetime_to_microseconds(datetime(2024, 6, 15, 12, 30, 45, tzinfo=timezone.utc))


def test_the_whole_datetime_range_converts() -> None:
    """Read as UTC, a naive datetime converts by subtraction from the epoch, over the full
    datetime range.

    Read as local time it went through the platform's local-time conversion instead, which
    `datetime.min` falls outside of whatever the machine's timezone is.
    """
    assert datetime_to_microseconds(datetime.min) == datetime_to_microseconds(
        datetime.min.replace(tzinfo=timezone.utc)
    )
