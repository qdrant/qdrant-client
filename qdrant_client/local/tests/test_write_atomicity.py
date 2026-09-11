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


def batch_collection() -> LocalCollection:
    collection = LocalCollection(
        models.CreateCollection(
            vectors={
                "a": models.VectorParams(size=3, distance=models.Distance.DOT),
                "b": models.VectorParams(size=3, distance=models.Distance.DOT),
            }
        )
    )
    collection.upsert(
        [
            models.PointStruct(
                id=1, vector={"a": GOOD_VECTOR, "b": GOOD_VECTOR}, payload={"p": "orig"}
            )
        ]
    )
    return collection


def touch_payload() -> models.SetPayloadOperation:
    return models.SetPayloadOperation(
        set_payload=models.SetPayload(payload={"p": "changed"}, points=[1])
    )


def test_rejected_batch_operation_applies_nothing() -> None:
    """An unknown vector name in a later operation must not keep an earlier one applied.

    The server errors too, but whether it deletes the name it recognized varies run to
    run, so this is asserted here rather than in the congruence tests.
    """
    collection = batch_collection()
    bad_operation = models.DeleteVectorsOperation(
        delete_vectors=models.DeleteVectors(points=[1], vector=["a", "nope"])
    )

    with pytest.raises(ValueError):
        collection.batch_update_points([touch_payload(), bad_operation])

    assert collection.payload[0] == {"p": "orig"}
    assert sorted(collection._get_vectors(idx=0, with_vectors=True)) == ["a", "b"]


@pytest.mark.parametrize(
    ("vectors_config", "good", "bad"),
    [
        (
            {"d": models.VectorParams(size=3, distance=models.Distance.DOT)},
            {"d": GOOD_VECTOR},
            {"d": [1.0, 2.0, 3.0, 4.0, 5.0]},
        ),
        (
            {
                "m": models.VectorParams(
                    size=3,
                    distance=models.Distance.DOT,
                    multivector_config=models.MultiVectorConfig(
                        comparator=models.MultiVectorComparator.MAX_SIM
                    ),
                )
            },
            {"m": [GOOD_VECTOR]},
            {"m": [[1.0, 2.0, 3.0, 4.0, 5.0]]},
        ),
    ],
    ids=["dense", "multivector"],
)
def test_wrong_vector_dimension_is_rejected_before_writing(vectors_config, good, bad) -> None:
    """A wrong-size vector used to reach numpy: dense died mid-write, multivector was stored."""
    collection = LocalCollection(models.CreateCollection(vectors=vectors_config))
    collection.upsert([models.PointStruct(id=1, vector=good)])

    with pytest.raises(ValueError, match="expected dim: 3, got 5"):
        collection.upsert([models.PointStruct(id=2, vector=bad)])

    assert len(collection.ids) == 1
    assert_internally_consistent(collection)

    with pytest.raises(ValueError, match="expected dim: 3, got 5"):
        collection.update_vectors([models.PointVectors(id=1, vector=bad)])

    assert collection._get_vectors(idx=0, with_vectors=True) == good
