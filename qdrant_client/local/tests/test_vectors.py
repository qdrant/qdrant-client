import random

from qdrant_client import models
from qdrant_client.local.local_collection import LocalCollection, DEFAULT_VECTOR_NAME


def test_get_vectors():
    """Test retrieving vectors from a local collection."""
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

    assert collection._get_vectors(idx=1, with_vectors=DEFAULT_VECTOR_NAME)
    assert collection._get_vectors(idx=2, with_vectors=True)
    assert collection._get_vectors(idx=3, with_vectors=False) is None


def test_multivector_search_skips_points_without_a_vector():
    """Test that missing named multivectors are skipped during local search."""
    collection = LocalCollection(
        models.CreateCollection(
            vectors={
                "dense": models.VectorParams(size=4, distance=models.Distance.COSINE),
                "multi": models.VectorParams(
                    size=4,
                    distance=models.Distance.COSINE,
                    multivector_config=models.MultiVectorConfig(
                        comparator=models.MultiVectorComparator.MAX_SIM
                    ),
                ),
            }
        )
    )
    collection.upsert(
        points=[
            models.PointStruct(id=1, vector={}),
            models.PointStruct(id=2, vector={"dense": [1.0] * 4, "multi": [[1.0] * 4]}),
        ]
    )

    result = collection.search(("multi", [[1.0] * 4]), limit=5)

    assert [point.id for point in result] == [2]
