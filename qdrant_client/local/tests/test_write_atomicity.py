"""A rejected write must not leave the collection's internal arrays skewed.

The observable half of this — local and remote agreeing after a refused write — lives in
`tests/congruence_tests/test_updates.py`. What can only be checked from the inside is that
`payload`, `deleted` and `deleted_per_vector` stay aligned with `ids_inv`. They used to
drift whenever validation ran after the point had already been added, and the next
successful upsert then broke every query with a shape mismatch.
"""

import pytest

from qdrant_client import models
from qdrant_client.local.local_collection import LocalCollection

NAN_VECTOR = [1.0, float("nan"), 3.0]
GOOD_VECTOR = [1.0, 2.0, 3.0]


def assert_internally_consistent(collection: LocalCollection) -> None:
    assert len(collection.ids) == len(collection.ids_inv)
    assert len(collection.payload) == len(collection.ids_inv)
    assert len(collection.deleted) == len(collection.ids_inv)
    for vector_name, deleted in collection.deleted_per_vector.items():
        assert len(deleted) == len(collection.ids_inv), vector_name


def test_rejected_add_keeps_internal_arrays_aligned() -> None:
    collection = LocalCollection(
        models.CreateCollection(
            vectors={"d": models.VectorParams(size=3, distance=models.Distance.DOT)}
        )
    )
    collection.upsert([models.PointStruct(id=1, vector={"d": GOOD_VECTOR})])

    with pytest.raises(ValueError, match="Vector contains NaN values"):
        collection.upsert([models.PointStruct(id=2, vector={"d": NAN_VECTOR})])

    assert_internally_consistent(collection)

    # the skew used to surface only here, once a healthy point arrived
    collection.upsert([models.PointStruct(id=3, vector={"d": [1.0, 1.0, 1.0]})])
    assert len(collection.search(query_vector=("d", GOOD_VECTOR), limit=10)) == 2


def test_rejected_multivector_add_keeps_internal_arrays_aligned() -> None:
    collection = LocalCollection(
        models.CreateCollection(
            vectors={
                "m": models.VectorParams(
                    size=3,
                    distance=models.Distance.DOT,
                    multivector_config=models.MultiVectorConfig(
                        comparator=models.MultiVectorComparator.MAX_SIM
                    ),
                )
            }
        )
    )
    collection.upsert([models.PointStruct(id=1, vector={"m": [GOOD_VECTOR]})])

    with pytest.raises(ValueError, match="Vector contains NaN values"):
        collection.upsert([models.PointStruct(id=2, vector={"m": [NAN_VECTOR]})])

    assert_internally_consistent(collection)

    collection.upsert([models.PointStruct(id=3, vector={"m": [[1.0, 1.0, 1.0]]})])
    assert len(collection.search(query_vector=("m", [GOOD_VECTOR]), limit=10)) == 2


def test_rejected_delete_vectors_deletes_nothing() -> None:
    """An unknown vector name must not take the recognized names down with it.

    This one stays local-only: the server always rejects the request, but whether it
    deletes the name it recognized before noticing the unknown one varies from run to run,
    so there is no server behavior for a congruence test to pin down.
    """
    collection = LocalCollection(
        models.CreateCollection(
            vectors={
                "a": models.VectorParams(size=3, distance=models.Distance.DOT),
                "b": models.VectorParams(size=3, distance=models.Distance.DOT),
            }
        )
    )
    collection.upsert([models.PointStruct(id=1, vector={"a": GOOD_VECTOR, "b": GOOD_VECTOR})])

    with pytest.raises(ValueError, match="Not existing vector name error: nope"):
        collection.delete_vectors(vectors=["a", "nope"], selector=[1])

    assert sorted(collection._get_vectors(idx=0, with_vectors=True)) == ["a", "b"]
