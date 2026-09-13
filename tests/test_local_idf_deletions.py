import math

import pytest

from qdrant_client import QdrantClient, models


@pytest.mark.parametrize("persistent", [False, True])
@pytest.mark.parametrize("operation", ["points", "vectors_upsert", "vectors_update"])
def test_idf_statistics_after_deletion(tmp_path, persistent: bool, operation: str) -> None:
    client = QdrantClient(path=str(tmp_path)) if persistent else QdrantClient(":memory:")
    client.create_collection(
        "test",
        vectors_config={},
        sparse_vectors_config={"text": models.SparseVectorParams(modifier=models.Modifier.IDF)},
    )
    vector = models.SparseVector(indices=[0], values=[1.0])
    client.upsert(
        "test",
        [models.PointStruct(id=i, vector={"text": vector}) for i in range(3)]
        + [models.PointStruct(id=3, vector={})],
    )

    def assert_scores(num_vectors: int) -> None:
        for idf in (models.IdfScope.GLOBAL, models.IdfCorpusParams(corpus=models.Filter())):
            points = client.query_points(
                "test", using="text", query=vector, search_params=models.SearchParams(idf=idf)
            ).points
            assert len(points) == num_vectors
            assert [point.score for point in points] == pytest.approx(
                [math.log((num_vectors + 1) / (num_vectors + 0.5))] * num_vectors
            )

    try:
        assert_scores(3)
        # Repeating a deletion must not subtract document frequencies a second time.
        for _ in range(2):
            if operation == "points":
                client.delete("test", points_selector=[1, 2, 3])
            else:
                client.delete_vectors("test", vectors=["text"], points=[1, 2, 3])
            assert_scores(1)

        if persistent:
            client.close()
            client = QdrantClient(path=str(tmp_path))
            assert_scores(1)

        if operation != "vectors_update":
            client.upsert(
                "test", [models.PointStruct(id=i, vector={"text": vector}) for i in [1, 2]]
            )
        else:
            client.update_vectors(
                "test", [models.PointVectors(id=i, vector={"text": vector}) for i in [1, 2]]
            )
        assert_scores(3)
    finally:
        client.close()
