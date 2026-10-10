"""Write-time validation has to reject values the collection cannot store.

NaN was always rejected. +-inf slipped through and corrupted the point instead: under
COSINE the stored vector normalised into NaN and then sorted to the top with a NaN
score; under DOT and EUCLID it scored +-inf; an inf in a sparse vector's values scored
inf. In each case the damage was silent and permanent.
"""

import pytest

from qdrant_client import models
from qdrant_client.http.models import SparseVector
from qdrant_client.local.local_collection import (
    VECTOR_MUST_BE_FINITE,
    LocalCollection,
    validate_dense_vector,
    validate_multivector,
)
from qdrant_client.local.sparse import (
    validate_sparse_vector,
    validate_sparse_vector_at_write,
)

INF = float("inf")

# 1e40 is finite as a Python float but has no float32 representative, so it becomes
# infinity exactly where it hurts: in the array that gets stored.
NON_FINITE_VALUES = [float("nan"), INF, -INF, 1e40]


@pytest.mark.parametrize("value", NON_FINITE_VALUES)
def test_dense_vector_rejects_non_finite(value: float) -> None:
    """Dense write validation must refuse every value float32 cannot store."""
    with pytest.raises(ValueError, match=VECTOR_MUST_BE_FINITE):
        validate_dense_vector([1.0, value, 3.0], "d")


@pytest.mark.parametrize("value", NON_FINITE_VALUES)
def test_multivector_rejects_non_finite(value: float) -> None:
    """Multivector write validation must refuse every value float32 cannot store."""
    with pytest.raises(ValueError, match=VECTOR_MUST_BE_FINITE):
        validate_multivector([[1.0, value, 3.0], [4.0, 5.0, 6.0]], "m")


@pytest.mark.parametrize("value", NON_FINITE_VALUES)
def test_sparse_vector_rejects_non_finite_at_write(value: float) -> None:
    """Sparse write validation must refuse every value float32 cannot store."""
    with pytest.raises(ValueError, match="NaN or infinite"):
        validate_sparse_vector_at_write(SparseVector(indices=[1, 2], values=[0.5, value]))


def test_sparse_query_paths_still_accept_infinity() -> None:
    """Query-time validation is deliberately unchanged: +-inf stays legal there.

    `validate_sparse_vector` runs on search/recommend inputs too, so rejecting
    infinity in it would change query behaviour this PR does not touch. An inf
    value in a query simply scores +-inf against the stored points.
    """
    validate_sparse_vector(SparseVector(indices=[1, 2], values=[0.5, INF]))  # must not raise

    collection = LocalCollection(
        models.CreateCollection(
            vectors={},
            sparse_vectors={"s": models.SparseVectorParams()},
        )
    )
    collection.upsert(
        [
            models.PointStruct(
                id=1,
                vector={"s": models.SparseVector(indices=[1], values=[1.0])},
            )
        ]
    )
    hits = collection.search(
        query_vector=("s", models.SparseVector(indices=[1], values=[INF])), limit=5
    )
    assert [hit.id for hit in hits] == [1]


def test_rejected_write_leaves_later_search_clean() -> None:
    """The point that used to turn into NaN after a rejected write is simply not there."""
    collection = LocalCollection(
        models.CreateCollection(
            vectors={"d": models.VectorParams(size=3, distance=models.Distance.COSINE)}
        )
    )
    collection.upsert([models.PointStruct(id=1, vector={"d": [1.0, 0.0, 0.0]})])

    with pytest.raises(ValueError, match=VECTOR_MUST_BE_FINITE):
        collection.upsert([models.PointStruct(id=2, vector={"d": [INF, 0.0, 0.0]})])

    hits = collection.search(query_vector=("d", [1.0, 0.0, 0.0]), limit=10)
    assert [hit.id for hit in hits] == [1]
    # the corrupted version of this point scored NaN and sorted first
    assert hits[0].score == pytest.approx(1.0, abs=1e-6)


def test_rejected_rewrite_keeps_the_stored_vector() -> None:
    """Overwriting an existing point validates too - that path has its own normalisation step."""
    collection = LocalCollection(
        models.CreateCollection(
            vectors={"d": models.VectorParams(size=3, distance=models.Distance.COSINE)}
        )
    )
    collection.upsert([models.PointStruct(id=1, vector={"d": [1.0, 0.0, 0.0]})])

    with pytest.raises(ValueError, match=VECTOR_MUST_BE_FINITE):
        collection.upsert([models.PointStruct(id=1, vector={"d": [INF, 0.0, 0.0]})])

    stored = collection.retrieve([1], with_vectors=True)[0].vector["d"]
    assert stored == pytest.approx([1.0, 0.0, 0.0], abs=1e-6)
