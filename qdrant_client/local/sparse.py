import numpy as np

from qdrant_client._pydantic_compat import construct
from qdrant_client.http.models import SparseVector


def empty_sparse_vector() -> SparseVector:
    return SparseVector(
        indices=[],
        values=[],
    )


def copy_sparse_vector(vector: SparseVector) -> SparseVector:
    """Return a sparse vector sharing no lists with `vector`.

    Two flat lists, so `deepcopy` is an order of magnitude slower for the same result, and
    this runs per sparse vector per point on every read.
    """
    return construct(SparseVector, indices=list(vector.indices), values=list(vector.values))


def validate_sparse_vector(vector: SparseVector) -> None:
    # these validate user input, so they must not be `assert`s: python -O strips those,
    # which would let a malformed vector into the collection
    if len(vector.indices) != len(vector.values):
        raise ValueError("Indices and values must have the same length")
    if np.isnan(vector.values).any():
        raise ValueError("Values must not contain NaN")
    if len(vector.indices) != len(set(vector.indices)):
        raise ValueError("Indices must be unique")


def is_sorted(vector: SparseVector) -> bool:
    for i in range(1, len(vector.indices)):
        if vector.indices[i] < vector.indices[i - 1]:
            return False
    return True


def sort_sparse_vector(vector: SparseVector) -> SparseVector:
    if is_sorted(vector):
        return vector

    sorted_indices = np.argsort(vector.indices)
    return SparseVector(
        indices=[vector.indices[i] for i in sorted_indices],
        values=[vector.values[i] for i in sorted_indices],
    )
