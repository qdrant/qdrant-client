from datetime import datetime, timezone

from qdrant_client.local.order_by import datetime_to_microseconds


def _exact_microseconds_since_epoch(dt: datetime) -> int:
    """Reference implementation using only exact integer arithmetic (no floats),
    to cross-check `datetime_to_microseconds` against."""
    epoch = datetime(1970, 1, 1, tzinfo=timezone.utc)
    delta = dt - epoch
    return delta.days * 86400 * 1_000_000 + delta.seconds * 1_000_000 + delta.microseconds


def test_datetime_to_microseconds_matches_exact_arithmetic():
    # Regression test: the previous implementation computed
    # `int(dt.timestamp() * 1_000_000)`, which truncates a float64 that has
    # already accumulated rounding error from the multiplication - off by one
    # microsecond for a large fraction of timestamps (worse the further the
    # date is from 1970, where the float has less precision to spare).
    # These specific values are known to trigger that mismatch.
    known_mismatching_datetimes = [
        datetime(1970, 7, 21, 14, 9, 16, 146413, tzinfo=timezone.utc),
        datetime(1970, 4, 9, 5, 50, 44, 113346, tzinfo=timezone.utc),
        datetime(2024, 6, 15, 12, 30, 45, 123456, tzinfo=timezone.utc),
        datetime(2100, 1, 1, 0, 0, 0, 1, tzinfo=timezone.utc),
        datetime(9999, 12, 31, 23, 59, 59, 999999, tzinfo=timezone.utc),
    ]
    for dt in known_mismatching_datetimes:
        assert datetime_to_microseconds(dt) == _exact_microseconds_since_epoch(dt)


def test_datetime_to_microseconds_no_mismatch_across_range():
    # Broader sweep: every microsecond value in one second, at a date far enough
    # from 1970 that the float-multiplication bug reliably reproduces.
    for micro in range(
        0, 1_000_000, 997
    ):  # every ~1000th microsecond, full coverage would be slow
        dt = datetime(2100, 6, 15, 12, 30, 45, micro, tzinfo=timezone.utc)
        assert datetime_to_microseconds(dt) == _exact_microseconds_since_epoch(dt)
