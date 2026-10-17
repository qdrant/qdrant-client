from datetime import date, datetime, timedelta, timezone, tzinfo

import pytest

from qdrant_client.local.datetime_utils import parse
from qdrant_client.local.order_by import datetime_to_microseconds, to_order_value


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


@pytest.mark.parametrize(  # type: ignore
    "dt, microseconds",
    [
        (datetime(1970, 7, 21, 14, 9, 16, 146413, tzinfo=timezone.utc), 17417356146413),
        (datetime(2024, 6, 15, 12, 30, 45, 123456, tzinfo=timezone.utc), 1718454645123456),
        (datetime(2100, 1, 1, 0, 0, 0, 1, tzinfo=timezone.utc), 4102444800000001),
        (datetime.min.replace(tzinfo=timezone.utc), -62135596800000000),
        (datetime.max.replace(tzinfo=timezone.utc), 253402300799999999),
    ],
)
def test_datetime_to_microseconds_is_exact(dt: datetime, microseconds: int) -> None:
    """The old float path read 146412 for the first of these, and put `datetime.max` a
    microsecond above its own value."""
    assert datetime_to_microseconds(dt) == microseconds


def test_every_microsecond_is_distinguishable_far_from_the_epoch() -> None:
    """A fix that replaced only the multiplication, still taking whole seconds from
    `timestamp()`, was still wrong this far out."""
    base = datetime(2100, 6, 15, 12, 30, 45, tzinfo=timezone.utc)
    start = datetime_to_microseconds(base)

    for microsecond in range(0, 1_000_000, 997):
        assert (
            datetime_to_microseconds(base.replace(microsecond=microsecond)) == start + microsecond
        )


def test_tzinfo_without_an_offset_counts_as_naive() -> None:
    """A tzinfo reporting no offset is naive per the datetime docs, so it gets UTC too."""

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


def test_to_order_value_reads_a_bare_date_as_utc_midnight() -> None:
    """A `date` is a member of the StartFrom union, and means the same instant as the
    "%Y-%m-%d" string REST serializes it to. Local midnight would shift the window, and is
    only visible on a client outside UTC."""
    assert to_order_value(date(2021, 1, 1)) == 1609459200000000  # 2021-01-01T00:00:00Z
    assert to_order_value("2021-01-01") == 1609459200000000
