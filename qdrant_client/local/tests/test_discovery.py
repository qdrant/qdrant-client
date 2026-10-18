import pytest

from qdrant_client import QdrantClient, models


@pytest.mark.parametrize(
    "vector_kind,distance",
    [(kind, distance) for kind in ("dense", "multi") for distance in models.Distance]
    + [("sparse", models.Distance.DOT)],
)
def test_discovery_on_empty_collection(vector_kind: str, distance: models.Distance) -> None:
    """Check discovery before insertion, after insertion, and after deleting all points."""
    client = QdrantClient(":memory:")
    collection_name = "empty-discovery"
    positive: models.SparseVector | list[float] | list[list[float]]
    negative: models.SparseVector | list[float] | list[list[float]]

    try:
        if vector_kind == "sparse":
            client.create_collection(
                collection_name,
                vectors_config={},
                sparse_vectors_config={"vector": models.SparseVectorParams()},
            )
            positive = models.SparseVector(indices=[0], values=[1.0])
            negative = models.SparseVector(indices=[1], values=[1.0])
        else:
            client.create_collection(
                collection_name,
                vectors_config={
                    "vector": models.VectorParams(
                        size=2,
                        distance=distance,
                        multivector_config=models.MultiVectorConfig(
                            comparator=models.MultiVectorComparator.MAX_SIM
                        )
                        if vector_kind == "multi"
                        else None,
                    )
                },
            )
            positive = [[1.0, 0.0]] if vector_kind == "multi" else [1.0, 0.0]
            negative = [[0.0, 1.0]] if vector_kind == "multi" else [0.0, 1.0]

        query = models.DiscoverQuery(
            discover=models.DiscoverInput(
                target=positive,
                context=models.ContextPair(positive=positive, negative=negative),
            )
        )

        def discover() -> list[models.ScoredPoint]:
            """Run the prepared discovery query against the current collection contents."""
            return client.query_points(collection_name, query=query, using="vector").points

        assert discover() == []

        client.upsert(
            collection_name,
            points=[models.PointStruct(id=1, vector={"vector": positive})],
        )
        assert [point.id for point in discover()] == [1]

        client.delete(collection_name, points_selector=models.PointIdsList(points=[1]))
        assert discover() == []
    finally:
        client.close()
