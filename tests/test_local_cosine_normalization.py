import numpy as np

from qdrant_client import QdrantClient, models


def test_repeated_cosine_upsert_stores_identical_vector() -> None:
    """Re-upserting the same point must store bit-identical normalized vectors
    for both the insert path and the update path (#1411)."""
    client = QdrantClient(":memory:")
    client.create_collection(
        "test",
        vectors_config={
            "dense": models.VectorParams(size=3, distance=models.Distance.COSINE),
            "multi": models.VectorParams(
                size=3,
                distance=models.Distance.COSINE,
                multivector_config=models.MultiVectorConfig(
                    comparator=models.MultiVectorComparator.MAX_SIM
                ),
            ),
        },
    )
    point = models.PointStruct(
        id=1,
        vector={
            "dense": [0.1, 0.2, 0.30000001],
            "multi": [[0.1, 0.2, 0.30000001], [0.7, 0.11, 0.13]],
        },
    )

    client.upsert("test", [point])  # insert path
    first = client.retrieve("test", [1], with_vectors=True)[0].vector

    client.upsert("test", [point])  # update path
    second = client.retrieve("test", [1], with_vectors=True)[0].vector

    assert np.array_equal(
        np.asarray(first["dense"], dtype=np.float32),
        np.asarray(second["dense"], dtype=np.float32),
    )
    assert np.array_equal(
        np.asarray(first["multi"], dtype=np.float32),
        np.asarray(second["multi"], dtype=np.float32),
    )
