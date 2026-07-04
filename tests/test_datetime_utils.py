from datetime import datetime, timedelta, timezone

import pytest

from qdrant_client.local.datetime_utils import parse

UTC = timezone.utc


@pytest.mark.parametrize(
    ("date_str", "expected"),
    [
        # Full precision with an explicit zero offset.
        (
            "2021-01-01T00:00:00.123456+0000",
            datetime(2021, 1, 1, 0, 0, 0, 123456, tzinfo=UTC),
        ),
        # Space separator instead of "T".
        ("2021-01-01 00:00:00+0000", datetime(2021, 1, 1, 0, 0, 0, tzinfo=UTC)),
        # Fractional seconds without a timezone.
        (
            "2021-01-01 00:00:00.500000",
            datetime(2021, 1, 1, 0, 0, 0, 500000, tzinfo=UTC),
        ),
        # Minute precision.
        ("2021-01-01 12:30", datetime(2021, 1, 1, 12, 30, 0, tzinfo=UTC)),
    ],
)
def test_parse_supported_formats(date_str: str, expected: datetime) -> None:
    assert parse(date_str) == expected


@pytest.mark.parametrize(
    ("date_str", "expected"),
    [
        ("2021-01-01T00:00:00", datetime(2021, 1, 1, 0, 0, 0, tzinfo=UTC)),
        ("2021-01-01 00:00:00", datetime(2021, 1, 1, 0, 0, 0, tzinfo=UTC)),
        # Date-only inputs become midnight UTC.
        ("2021-01-01", datetime(2021, 1, 1, 0, 0, 0, tzinfo=UTC)),
    ],
)
def test_parse_assumes_utc_when_timezone_missing(
    date_str: str, expected: datetime
) -> None:
    parsed = parse(date_str)
    assert parsed == expected
    assert parsed.tzinfo == UTC


@pytest.mark.parametrize(
    ("date_str", "offset_hours"),
    [
        ("2021-01-01 00:00:00.000+01", 1),
        ("2021-06-15 12:00:00-10", -10),
        ("2021-06-15T12:00:00+05", 5),
    ],
)
def test_parse_hour_only_timezone_offset(date_str: str, offset_hours: int) -> None:
    # Python's strptime cannot parse a bare "+HH" offset, so parse() retries with
    # ":00" appended. The resulting datetime must carry the expected offset.
    parsed = parse(date_str)
    assert parsed is not None
    assert parsed.utcoffset() == timedelta(hours=offset_hours)


def test_parse_hour_minute_timezone_offset() -> None:
    parsed = parse("2021-01-01T00:00:00+0530")
    assert parsed is not None
    assert parsed.utcoffset() == timedelta(hours=5, minutes=30)


@pytest.mark.parametrize(
    "date_str",
    [
        "",
        "not a date",
        "2021-13-01",  # invalid month
        "2021-01-32",  # invalid day
        "01-01-2021",  # wrong field order
        "2021/01/01",  # unsupported separator
    ],
)
def test_parse_returns_none_for_invalid_input(date_str: str) -> None:
    assert parse(date_str) is None
