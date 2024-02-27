from datetime import datetime, timedelta, timezone

import pytest

from qdrant_client.conversions.conversion import GrpcToRest, RestToGrpc
from qdrant_client.local.datetime_utils import parse


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
        ("2021-01-01", datetime(2021, 1, 1, 0, 0, 0)),
        ("2021-01-01 00:00:00", datetime(2021, 1, 1, 0, 0, 0)),
        ("2021-01-01 00:00:00Z", datetime(2021, 1, 1, 0, 0, 0, tzinfo=timezone.utc)),
        ("2021-01-01 00:00:00+00:00", datetime(2021, 1, 1, 0, 0, 0, tzinfo=timezone.utc)),
        ("2021-01-01 00:00:00.000000", datetime(2021, 1, 1, 0, 0, 0)),
        ("2021-01-01 00:00:00.000000Z", datetime(2021, 1, 1, 0, 0, 0, tzinfo=timezone.utc)),
        ("2021-01-01 00:00:00.000000+00:00", datetime(2021, 1, 1, 0, 0, 0, tzinfo=timezone.utc)),
        ("2021-01-01 00:00:00.000-00:00", datetime(2021, 1, 1, 0, 0, 0, tzinfo=timezone.utc)),
        (
            "2021-01-01 00:00:00-03:00",
            datetime(2021, 1, 1, 0, 0, 0, tzinfo=timezone(timedelta(days=-1, seconds=75600))),
        ),
    ],
)
def test_parse_dates(date_str: str, expected: datetime):
    assert parse(date_str) == expected


@pytest.mark.parametrize(
    "dt",
    [
        datetime(2021, 1, 1, 0, 0, 0, tzinfo=timezone.utc),
        datetime(2021, 1, 1, 0, 0, 0, tzinfo=timezone(timedelta(hours=5))),
        datetime(2021, 1, 1, 0, 0, 0),
        datetime.utcnow(),
        datetime.now(),
    ],
)
def test_datetime_to_timestamp(dt: datetime):
    rest_to_grpc = RestToGrpc.convert_datetime(dt)
    grpc_to_rest = GrpcToRest.convert_timestamp(rest_to_grpc)

    print(f"dt: {dt}, rest_to_grpc: {rest_to_grpc}, grpc_to_rest: {grpc_to_rest}")
    assert (
        dt.utctimetuple() == grpc_to_rest.utctimetuple()
    ), f"Failed for {dt}, should be equal to {grpc_to_rest}"
