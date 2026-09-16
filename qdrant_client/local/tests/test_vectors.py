import random

import pytest

from qdrant_client import models
from qdrant_client.local.local_collection import LocalCollection, DEFAULT_VECTOR_NAME


def test_get_vectors():
    collection = LocalCollection(
        models.CreateCollection(
            vectors=models.VectorParams(size=2, distance=models.Distance.MANHATTAN)
        )
    )
    collection.upsert(
        points=[
            models.PointStruct(id=i, vector=[random.random(), random.random()]) for i in range(10)
        ]
    )

    assert collection._get_vectors(idx=1, with_vectors=[DEFAULT_VECTOR_NAME])
    assert collection._get_vectors(idx=2, with_vectors=True)
    assert collection._get_vectors(idx=3, with_vectors=False) is None


def test_query_vector_kind_mismatch_error():
    """A query vector of the wrong kind against an existing named vector must
    produce an accurate error, not a misleading 'not found'."""
    multivector_config = models.MultiVectorConfig(
        comparator=models.MultiVectorComparator.MAX_SIM
    )
    collection = LocalCollection(
        models.CreateCollection(
            vectors={
                "dense_vec": models.VectorParams(
                    size=2, distance=models.Distance.COSINE
                ),
                "multi_vec": models.VectorParams(
                    size=2,
                    distance=models.Distance.COSINE,
                    multivector_config=multivector_config,
                ),
            }
        )
    )

    # 1-D query against a multivector
    with pytest.raises(ValueError, match="is a multivector vector"):
        collection.search(("multi_vec", [1.0, 0.0]))

    # 2-D query against a dense vector
    with pytest.raises(ValueError, match="is a dense vector"):
        collection.search(("dense_vec", [[1.0, 0.0]]))

    # truly missing names keep the original-style error
    with pytest.raises(ValueError, match="Dense vector missing is not found"):
        collection.search(("missing", [1.0, 0.0]))

    # correct kinds still search fine
    collection.upsert(
        points=[models.PointStruct(id=1, vector={"dense_vec": [1.0, 0.0]})]
    )
    assert collection.search(("dense_vec", [1.0, 0.0]))[0].id == 1


def test_query_vector_kind_mismatch_sparse():
    """Sparse queries routed to a dense/multivector name (or a missing name)
    produce the accurate mismatch / not-found messages."""
    multivector_config = models.MultiVectorConfig(
        comparator=models.MultiVectorComparator.MAX_SIM
    )
    collection = LocalCollection(
        models.CreateCollection(
            vectors={
                "dense_vec": models.VectorParams(
                    size=2, distance=models.Distance.COSINE
                ),
                "multi_vec": models.VectorParams(
                    size=2,
                    distance=models.Distance.COSINE,
                    multivector_config=multivector_config,
                ),
            },
            sparse_vectors={"sparse_vec": models.SparseVectorParams()},
        )
    )
    sparse = models.SparseVector(indices=[0], values=[1.0])

    with pytest.raises(ValueError, match="is a dense vector"):
        collection.search(("dense_vec", sparse))

    with pytest.raises(ValueError, match="is a multivector vector"):
        collection.search(("multi_vec", sparse))

    with pytest.raises(ValueError, match="Sparse vector missing is not found"):
        collection.search(("missing", sparse))

    # sparse queries against a configured sparse name still work
    collection.upsert(
        points=[
            models.PointStruct(
                id=1,
                vector={"sparse_vec": models.SparseVector(indices=[0], values=[1.0])},
            )
        ]
    )
    assert collection.search(("sparse_vec", sparse))[0].id == 1


def test_query_vector_kind_mismatch_unnamed():
    """The unnamed default vector is rendered explicitly in the error message."""
    collection = LocalCollection(
        models.CreateCollection(
            vectors=models.VectorParams(size=2, distance=models.Distance.COSINE)
        )
    )
    with pytest.raises(ValueError, match="unnamed vector"):
        collection.search([[1.0, 0.0]])
