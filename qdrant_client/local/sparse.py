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
    """Validate sparse vector structure and reject NaN values.

    This runs on both write and query paths (search/recommend queries validate
    their inputs here too), so it keeps the pre-existing query-time contract:
    +-inf values are accepted at query time, where they simply score +-inf.
    Stored values get the stricter check in `validate_sparse_vector_at_write`.
    """
    # these validate user input, so they must not be `assert`s: python -O strips those,
    # which would let a malformed vector into the collection
    if len(vector.indices) != len(vector.values):
        raise ValueError("Indices and values must have the same length")
    if np.isnan(vector.values).any():
        raise ValueError("Values must not contain NaN")
    if len(vector.indices) != len(set(vector.indices)):
        raise ValueError("Indices must be unique")


def validate_sparse_vector_at_write(vector: SparseVector) -> None:
    """`validate_sparse_vector`, plus the finite check for stored values.

    The collection persists `values` as float32, so anything finite for Python
    but not for float32 (1e40) silently becomes infinity on cast and poisons
    every distance it takes part in. Write paths call this variant; query paths
    deliberately keep using `validate_sparse_vector`.
    """
    # the finite check runs before validate_sparse_vector so that a NaN input
    # gets the write-path message, not the query-path one; the cast may overflow
    # (1e40 -> inf) or be invalid (None -> nan), and neither should reach the
    # user as a numpy warning
    with np.errstate(over="ignore", invalid="ignore"):
        finite = bool(np.isfinite(np.asarray(vector.values, dtype=np.float32)).all())
    if not finite:
        raise ValueError("Values must not contain NaN or infinite values")
    validate_sparse_vector(vector)


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
