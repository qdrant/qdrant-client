from datetime import datetime, timezone

import pytest

from qdrant_client.local.datetime import parse


@pytest.mark.parametrize(  # type: ignore
    "date_str, expected",
    [
        ("2021-01-01T00:00:00", datetime(2021, 1, 1, 0, 0, 0)),
        ("2021-01-01T00:00:00Z", datetime(2021, 1, 1, 0, 0, 0, tzinfo=timezone.utc)),
        ("2021-01-01T00:00:00+00:00", datetime(2021, 1, 1, 0, 0, 0, tzinfo=timezone.utc)),
        ("2021-01-01T00:00:00.000000", datetime(2021, 1, 1, 0, 0, 0)),
        ("2021-01-01T00:00:00.000000Z", datetime(2021, 1, 1, 0, 0, 0, tzinfo=timezone.utc)),
        ("2021-01-01T00:00:00.000000+00:00", datetime(2021, 1, 1, 0, 0, 0, tzinfo=timezone.utc)),
        ("2021-01-01T00:00:00.000000+00:00", datetime(2021, 1, 1, 0, 0, 0, tzinfo=timezone.utc)),
        ("2021-01-01 00:00:00", datetime(2021, 1, 1, 0, 0, 0)),
        ("2021-01-01 00:00:00Z", datetime(2021, 1, 1, 0, 0, 0, tzinfo=timezone.utc)),
        ("2021-01-01 00:00:00+00:00", datetime(2021, 1, 1, 0, 0, 0, tzinfo=timezone.utc)),
        ("2021-01-01 00:00:00.000000", datetime(2021, 1, 1, 0, 0, 0)),
        ("2021-01-01 00:00:00.000000Z", datetime(2021, 1, 1, 0, 0, 0, tzinfo=timezone.utc)),
        ("2021-01-01 00:00:00.000000+00:00", datetime(2021, 1, 1, 0, 0, 0, tzinfo=timezone.utc)),
        ("2021-01-01 00:00:00.000-00:00", datetime(2021, 1, 1, 0, 0, 0, tzinfo=timezone.utc)),
    ],
)
def test_parse_dates(date_str: str, expected: datetime):
    assert parse(date_str) == expected
