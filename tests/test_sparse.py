import pytest

from qdrant_client.http.models import SparseVector
from qdrant_client.local.sparse import (
    empty_sparse_vector,
    is_sorted,
    sort_sparse_vector,
    validate_sparse_vector,
)


def test_empty_sparse_vector_has_no_entries():
    vector = empty_sparse_vector()
    assert vector.indices == []
    assert vector.values == []


def test_validate_accepts_a_well_formed_vector():
    # Matching lengths, no NaNs, unique indices -> no error.
    validate_sparse_vector(SparseVector(indices=[1, 2, 3], values=[0.1, 0.2, 0.3]))


def test_validate_rejects_length_mismatch():
    with pytest.raises(AssertionError, match="same length"):
        validate_sparse_vector(SparseVector(indices=[1, 2], values=[0.1]))


def test_validate_rejects_nan_values():
    with pytest.raises(AssertionError, match="NaN"):
        validate_sparse_vector(SparseVector(indices=[1], values=[float("nan")]))


def test_validate_rejects_duplicate_indices():
    with pytest.raises(AssertionError, match="unique"):
        validate_sparse_vector(SparseVector(indices=[1, 1], values=[0.1, 0.2]))


def test_is_sorted_on_empty_and_single_element():
    assert is_sorted(SparseVector(indices=[], values=[]))
    assert is_sorted(SparseVector(indices=[5], values=[0.5]))


def test_is_sorted_distinguishes_sorted_from_unsorted():
    assert is_sorted(SparseVector(indices=[1, 2, 3], values=[0.1, 0.2, 0.3]))
    assert not is_sorted(SparseVector(indices=[1, 3, 2], values=[0.1, 0.3, 0.2]))


def test_sort_orders_indices_and_carries_values_along():
    result = sort_sparse_vector(SparseVector(indices=[3, 1, 2], values=[0.3, 0.1, 0.2]))
    assert result.indices == [1, 2, 3]
    assert result.values == [0.1, 0.2, 0.3]


def test_sort_returns_same_object_when_already_sorted():
    vector = SparseVector(indices=[1, 2], values=[0.1, 0.2])
    # Already sorted: the function short-circuits and returns the input.
    assert sort_sparse_vector(vector) is vector
