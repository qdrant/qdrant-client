import pytest

from qdrant_client.http.models import SparseVector
from qdrant_client.local.sparse import validate_sparse_vector


def test_validate_sparse_vector_accepts_valid() -> None:
    validate_sparse_vector(SparseVector(indices=[], values=[]))
    validate_sparse_vector(SparseVector(indices=[1, 2, 3], values=[0.1, 0.2, 0.3]))
    # indices do not have to be sorted to be valid
    validate_sparse_vector(SparseVector(indices=[3, 1], values=[0.1, 0.2]))


@pytest.mark.parametrize(
    ("vector", "message"),
    [
        (SparseVector(indices=[1, 2], values=[0.1]), "same length"),
        (SparseVector(indices=[1], values=[float("nan")]), "NaN"),
        (SparseVector(indices=[1, 1], values=[0.1, 0.2]), "unique"),
    ],
)
def test_validate_sparse_vector_rejects_invalid(vector: SparseVector, message: str) -> None:
    # ValueError rather than AssertionError, so the check survives `python -O`
    with pytest.raises(ValueError, match=message):
        validate_sparse_vector(vector)
