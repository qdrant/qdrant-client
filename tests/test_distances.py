import numpy as np
import pytest

from qdrant_client.http import models
from qdrant_client.local.distances import (
    DistanceOrder,
    distance_to_order,
    dot_product,
    euclidean_distance,
    fast_sigmoid,
    manhattan_distance,
    scaled_fast_sigmoid,
)


def _query():
    return np.array([1.0, 0.0], dtype=np.float32)


def _vectors():
    return np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=np.float32)


def test_dot_product():
    assert dot_product(_query(), _vectors()).tolist() == [1.0, 0.0, 1.0]


def test_euclidean_distance():
    result = euclidean_distance(_query(), _vectors())
    assert result == pytest.approx([0.0, np.sqrt(2), 1.0], abs=1e-5)


def test_manhattan_distance():
    assert manhattan_distance(_query(), _vectors()).tolist() == [0.0, 2.0, 1.0]


def test_distance_to_order():
    assert distance_to_order(models.Distance.COSINE) == DistanceOrder.BIGGER_IS_BETTER
    assert distance_to_order(models.Distance.DOT) == DistanceOrder.BIGGER_IS_BETTER
    assert distance_to_order(models.Distance.EUCLID) == DistanceOrder.SMALLER_IS_BETTER
    assert (
        distance_to_order(models.Distance.MANHATTAN) == DistanceOrder.SMALLER_IS_BETTER
    )


def test_fast_sigmoid_maps_to_open_interval():
    assert fast_sigmoid(np.float32(0.0)) == 0.0
    assert fast_sigmoid(np.float32(1.0)) == pytest.approx(0.5)
    assert fast_sigmoid(np.float32(-1.0)) == pytest.approx(-0.5)


def test_fast_sigmoid_passes_non_finite_through():
    # NaN/inf are returned unchanged to avoid invalid divisions.
    assert np.isnan(fast_sigmoid(np.float32("nan")))
    assert fast_sigmoid(np.float32("inf")) == np.float32("inf")


def test_scaled_fast_sigmoid_shifts_into_unit_range():
    # 0.5 * (fast_sigmoid(x) + 1): x=0 -> 0.5, and stays within (0, 1).
    assert scaled_fast_sigmoid(np.float32(0.0)) == pytest.approx(0.5)
    assert 0.0 < scaled_fast_sigmoid(np.float32(5.0)) < 1.0
    assert 0.0 < scaled_fast_sigmoid(np.float32(-5.0)) < 1.0
