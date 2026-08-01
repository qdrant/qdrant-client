"""Unit tests for local-mode payload filter helpers.

These run without a server, so they can cover the multi-value paths that the
congruence tests in ``tests/congruence_tests`` don't reach.
"""

import pytest

from qdrant_client import models
from qdrant_client.local.payload_filters import check_values_count, get_value_counts


class TestGetValueCounts:
    @pytest.mark.parametrize(
        ("values", "expected"),
        [
            ([], [0]),
            ([None], [0]),
            ([None, None], [0]),
            (["a"], [1]),
            ([[1, 2, 3]], [3]),
            ([[1], [1, 2, 3]], [1, 3]),
            ([None, [1, 2]], [0, 2]),
        ],
    )
    def test_counts(self, values, expected):
        assert get_value_counts(values) == expected


class TestCheckValuesCount:
    def test_none_values_never_match(self):
        assert check_values_count(models.ValuesCount(gt=0), None) is False

    @pytest.mark.parametrize(
        ("condition", "expected"),
        [
            (models.ValuesCount(lt=4), True),
            (models.ValuesCount(lt=3), False),
            (models.ValuesCount(lte=3), True),
            (models.ValuesCount(lte=2), False),
            (models.ValuesCount(gt=2), True),
            (models.ValuesCount(gt=3), False),
            (models.ValuesCount(gte=3), True),
            (models.ValuesCount(gte=4), False),
            (models.ValuesCount(gt=2, lt=4), True),
            (models.ValuesCount(gte=3, lte=3), True),
        ],
    )
    def test_single_count(self, condition, expected):
        """One value, one count — the case the congruence tests already cover."""
        assert check_values_count(condition, [[1, 2, 3]]) is expected

    def test_bounds_must_be_satisfied_by_the_same_count(self):
        """Regression for #1292.

        Counts are ``[1, 10]``. ``1`` satisfies ``lt=9`` and ``10`` satisfies
        ``gt=2``, but neither count is inside ``(2, 9)``, so the point must not
        match. The bounds used to be evaluated independently, which let two
        different values combine to satisfy the range.
        """
        values = [[1], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]]
        assert get_value_counts(values) == [1, 10]

        assert check_values_count(models.ValuesCount(gt=2, lt=9), values) is False

    def test_multi_value_matches_when_one_count_satisfies_all_bounds(self):
        """The same shape, but with a count that genuinely falls in range."""
        values = [[1], [1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]]
        assert get_value_counts(values) == [1, 5, 10]

        assert check_values_count(models.ValuesCount(gt=2, lt=9), values) is True

    @pytest.mark.parametrize(
        ("condition", "expected"),
        [
            # 1 satisfies lte=1, 10 satisfies gte=10 — but no single count does both.
            (models.ValuesCount(gte=10, lte=1), False),
            # Both bounds satisfied by the count 10.
            (models.ValuesCount(gte=10, lte=10), True),
            # Both bounds satisfied by the count 1.
            (models.ValuesCount(gte=1, lte=1), True),
        ],
    )
    def test_inverted_bounds_across_values(self, condition, expected):
        values = [[1], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]]
        assert check_values_count(condition, values) is expected

    def test_all_four_bounds_against_one_count(self):
        values = [[1, 2, 3]]
        assert (
            check_values_count(models.ValuesCount(gt=2, gte=3, lt=4, lte=3), values) is True
        )
        assert (
            check_values_count(models.ValuesCount(gt=2, gte=3, lt=4, lte=2), values) is False
        )

    def test_no_bounds_matches_any_count(self):
        assert check_values_count(models.ValuesCount(), [[1, 2, 3]]) is True
