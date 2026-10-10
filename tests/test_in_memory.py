import math

import pytest

from qdrant_client import QdrantClient, models


@pytest.fixture
def qdrant() -> QdrantClient:
    return QdrantClient(":memory:")


def test_dense_in_memory_key_filter_returns_results(qdrant: QdrantClient):
    qdrant.create_collection(
        collection_name="test_collection",
        vectors_config=models.VectorParams(size=4, distance=models.Distance.DOT),
    )

    operation_info = qdrant.upsert(
        collection_name="test_collection",
        wait=True,
        points=[
            models.PointStruct(id=1, vector=[0.05, 0.61, 0.76, 0.74], payload={"city": "Berlin"}),
            models.PointStruct(
                id=2,
                vector=[0.19, 0.81, 0.75, 0.11],
                payload={"city": ["Berlin", "London"]},
            ),
            models.PointStruct(
                id=3,
                vector=[0.36, 0.55, 0.47, 0.94],
                payload={"city": ["Berlin", "Moscow"]},
            ),
            models.PointStruct(
                id=4,
                vector=[0.18, 0.01, 0.85, 0.80],
                payload={"city": ["London", "Moscow"]},
            ),
            models.PointStruct(id=5, vector=[0.24, 0.18, 0.22, 0.44], payload={"count": [0]}),
            models.PointStruct(id=6, vector=[0.35, 0.08, 0.11, 0.44]),
        ],
    )

    assert operation_info.operation_id == 0
    assert operation_info.status == models.UpdateStatus.COMPLETED

    search_result = qdrant.query_points(
        collection_name="test_collection",
        query=[0.2, 0.1, 0.9, 0.7],
        query_filter=models.Filter(
            must=[models.FieldCondition(key="city", match=models.MatchValue(value="London"))]
        ),
        limit=3,
    ).points

    assert [r.id for r in search_result] == [4, 2]


def test_sparse_in_memory_key_filter_returns_results(qdrant: QdrantClient):
    qdrant.create_collection(
        collection_name="test_collection",
        vectors_config={},
        sparse_vectors_config={"text": models.SparseVectorParams()},
    )

    operation_info = qdrant.upsert(
        collection_name="test_collection",
        wait=True,
        points=[
            models.PointStruct(
                id=1,
                vector={
                    "text": models.SparseVector(
                        indices=[0, 1, 2, 3], values=[0.05, 0.61, 0.76, 0.74]
                    )
                },
                payload={"city": "Berlin"},
            ),
            models.PointStruct(
                id=2,
                vector={
                    "text": models.SparseVector(
                        indices=[0, 1, 2, 3], values=[0.19, 0.81, 0.75, 0.11]
                    )
                },
                payload={"city": ["Berlin", "London"]},
            ),
            models.PointStruct(
                id=3,
                vector={
                    "text": models.SparseVector(
                        indices=[0, 1, 2, 3], values=[0.36, 0.55, 0.47, 0.94]
                    )
                },
                payload={"city": ["Berlin", "Moscow"]},
            ),
            models.PointStruct(
                id=4,
                vector={
                    "text": models.SparseVector(
                        indices=[0, 1, 2, 3], values=[0.18, 0.01, 0.85, 0.80]
                    )
                },
                payload={"city": ["London", "Moscow"]},
            ),
        ],
    )

    assert operation_info.operation_id == 0
    assert operation_info.status == models.UpdateStatus.COMPLETED

    search_result = qdrant.query_points(
        collection_name="test_collection",
        using="text",
        query=models.SparseVector(indices=[0, 1, 2, 3], values=[0.2, 0.1, 0.9, 0.7]),
        query_filter=models.Filter(
            must=[models.FieldCondition(key="city", match=models.MatchValue(value="London"))]
        ),
        limit=3,
    ).points

    assert [r.id for r in search_result] == [4, 2]


def test_fusion_rrf_score_threshold(qdrant: QdrantClient):
    """Test that RRF fusion with score_threshold correctly filters results.

    RRF scores in local mode are normalized and for 5 points we get roughly:
    - ID 1: 1.0
    - ID 2: 0.667
    - ID 3: 0.5
    - ID 5: 0.4
    - ID 4: 0.333

    A threshold of 0.45 should filter out IDs 4 and 5.
    """
    qdrant.create_collection(
        collection_name="test_collection",
        vectors_config={
            "text": models.VectorParams(size=4, distance=models.Distance.COSINE),
            "image": models.VectorParams(size=4, distance=models.Distance.COSINE),
        },
    )

    qdrant.upsert(
        collection_name="test_collection",
        wait=True,
        points=[
            models.PointStruct(
                id=1,
                vector={"text": [1.0, 0.0, 0.0, 0.0], "image": [1.0, 0.0, 0.0, 0.0]},
            ),
            models.PointStruct(
                id=2,
                vector={"text": [0.9, 0.1, 0.0, 0.0], "image": [0.9, 0.1, 0.0, 0.0]},
            ),
            models.PointStruct(
                id=3,
                vector={"text": [0.5, 0.5, 0.0, 0.0], "image": [0.5, 0.5, 0.0, 0.0]},
            ),
            models.PointStruct(
                id=4,
                vector={"text": [0.0, 1.0, 0.0, 0.0], "image": [0.0, 1.0, 0.0, 0.0]},
            ),
            models.PointStruct(
                id=5,
                vector={"text": [0.0, 0.0, 1.0, 0.0], "image": [0.0, 0.0, 1.0, 0.0]},
            ),
        ],
    )

    query_vector = [1.0, 0.0, 0.0, 0.0]

    # Without score_threshold - should return all 5 points
    result_no_threshold = qdrant.query_points(
        collection_name="test_collection",
        prefetch=[
            models.Prefetch(query=query_vector, using="text", limit=10),
            models.Prefetch(query=query_vector, using="image", limit=10),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=10,
    )
    assert len(result_no_threshold.points) == 5

    # Find points with scores below 0.45 - IDs 4 (0.333) and 5 (0.4) should be filtered
    low_score_count = sum(1 for p in result_no_threshold.points if p.score < 0.45)
    assert low_score_count == 2, f"Expected 2 low-scoring points, got {low_score_count}"

    # With a threshold of 0.45, points with scores below should be filtered
    result_with_threshold = qdrant.query_points(
        collection_name="test_collection",
        prefetch=[
            models.Prefetch(query=query_vector, using="text", limit=10),
            models.Prefetch(query=query_vector, using="image", limit=10),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        score_threshold=0.45,
        limit=10,
    )

    # Verify all returned points have score >= threshold
    for point in result_with_threshold.points:
        assert point.score >= 0.45, f"Score {point.score} is below threshold 0.45"

    # Key assertion: filtering should reduce the count from 5 to 3
    assert len(result_with_threshold.points) == 3, (
        f"Expected 3 points after filtering (threshold 0.45), got {len(result_with_threshold.points)}. "
        f"Scores: {[p.score for p in result_no_threshold.points]}"
    )


def test_fusion_dbsf_score_threshold(qdrant: QdrantClient):
    """Test that DBSF fusion with score_threshold correctly filters results.

    DBSF scores for the test data:
    - ID 1: ~1.30
    - ID 2: ~1.30
    - ID 3: ~1.11
    - ID 4: ~0.64
    - ID 5: ~0.64

    A threshold of 1.0 should filter out IDs 4 and 5.
    """
    qdrant.create_collection(
        collection_name="test_collection",
        vectors_config={
            "text": models.VectorParams(size=4, distance=models.Distance.COSINE),
            "image": models.VectorParams(size=4, distance=models.Distance.COSINE),
        },
    )

    qdrant.upsert(
        collection_name="test_collection",
        wait=True,
        points=[
            models.PointStruct(
                id=1,
                vector={"text": [1.0, 0.0, 0.0, 0.0], "image": [1.0, 0.0, 0.0, 0.0]},
            ),
            models.PointStruct(
                id=2,
                vector={"text": [0.9, 0.1, 0.0, 0.0], "image": [0.9, 0.1, 0.0, 0.0]},
            ),
            models.PointStruct(
                id=3,
                vector={"text": [0.5, 0.5, 0.0, 0.0], "image": [0.5, 0.5, 0.0, 0.0]},
            ),
            models.PointStruct(
                id=4,
                vector={"text": [0.0, 1.0, 0.0, 0.0], "image": [0.0, 1.0, 0.0, 0.0]},
            ),
            models.PointStruct(
                id=5,
                vector={"text": [0.0, 0.0, 1.0, 0.0], "image": [0.0, 0.0, 1.0, 0.0]},
            ),
        ],
    )

    query_vector = [1.0, 0.0, 0.0, 0.0]

    # Without score_threshold - should return all 5 points
    result_no_threshold = qdrant.query_points(
        collection_name="test_collection",
        prefetch=[
            models.Prefetch(query=query_vector, using="text", limit=10),
            models.Prefetch(query=query_vector, using="image", limit=10),
        ],
        query=models.FusionQuery(fusion=models.Fusion.DBSF),
        limit=10,
    )
    assert len(result_no_threshold.points) == 5

    # Find points with scores below 1.0 - IDs 4 and 5 (~0.64) should be filtered
    low_score_count = sum(1 for p in result_no_threshold.points if p.score < 1.0)
    assert low_score_count == 2, f"Expected 2 low-scoring points, got {low_score_count}"

    # With score_threshold of 1.0, points below should be filtered
    result_with_threshold = qdrant.query_points(
        collection_name="test_collection",
        prefetch=[
            models.Prefetch(query=query_vector, using="text", limit=10),
            models.Prefetch(query=query_vector, using="image", limit=10),
        ],
        query=models.FusionQuery(fusion=models.Fusion.DBSF),
        score_threshold=1.0,
        limit=10,
    )

    # Verify all returned points have score >= threshold
    for point in result_with_threshold.points:
        assert point.score >= 1.0, f"Score {point.score} is below threshold 1.0"

    # Key assertion: filtering should reduce the count from 5 to 3
    assert len(result_with_threshold.points) == 3, (
        f"Expected 3 points after filtering (threshold 1.0), got {len(result_with_threshold.points)}. "
        f"Scores: {[p.score for p in result_no_threshold.points]}"
    )


@pytest.mark.parametrize("operation", ["points", "vectors_upsert", "vectors_update"])
def test_idf_statistics_after_deletion(qdrant: QdrantClient, operation: str):
    """Deleting points or their sparse vectors has to take them out of the IDF statistics.

    Only the local client can be held to this. `IdfScope.GLOBAL` reads the sparse index on
    the server, which goes on counting deleted points for as long as they sit in it, so the
    two are expected to differ here and a congruence test cannot pin it down. What can be
    pinned down is that the global scope agrees with a corpus selecting every live point,
    and that both match the IDF formula.
    """
    qdrant.create_collection(
        collection_name="test_collection",
        vectors_config={},
        sparse_vectors_config={"text": models.SparseVectorParams(modifier=models.Modifier.IDF)},
    )
    vector = models.SparseVector(indices=[0], values=[1.0])
    qdrant.upsert(
        collection_name="test_collection",
        points=[models.PointStruct(id=i, vector={"text": vector}) for i in range(3)]
        # a point without the sparse vector is not part of the corpus to begin with
        + [models.PointStruct(id=3, vector={})],
    )

    def assert_scores(corpus_size: int):
        # ((n - df + 0.5) / (df + 0.5) + 1).ln() with every document holding the one term
        expected = math.log((corpus_size + 1) / (corpus_size + 0.5))
        for idf in (models.IdfScope.GLOBAL, models.IdfCorpusParams(corpus=models.Filter())):
            points = qdrant.query_points(
                collection_name="test_collection",
                using="text",
                query=vector,
                search_params=models.SearchParams(idf=idf),
            ).points
            assert len(points) == corpus_size
            assert [point.score for point in points] == pytest.approx([expected] * corpus_size)

    assert_scores(3)

    # repeating a deletion must not take the same document frequencies out twice
    for _ in range(2):
        if operation == "points":
            qdrant.delete("test_collection", points_selector=[1, 2, 3])
        else:
            qdrant.delete_vectors("test_collection", vectors=["text"], points=[1, 2, 3])
        assert_scores(1)

    if operation == "vectors_update":
        qdrant.update_vectors(
            "test_collection",
            [models.PointVectors(id=i, vector={"text": vector}) for i in (1, 2)],
        )
    else:
        qdrant.upsert(
            "test_collection",
            [models.PointStruct(id=i, vector={"text": vector}) for i in (1, 2)],
        )
    assert_scores(3)


@pytest.mark.parametrize("distance", [models.Distance.EUCLID, models.Distance.MANHATTAN])
def test_dbsf_fusion_respects_score_direction(qdrant: QdrantClient, distance: models.Distance):
    """DBSF must rank nearest-first on metrics where a lower score is a better match.

    Core normalizes the internal similarity, which is oriented "bigger is better" for
    every metric, so the fused order has to stay nearest-first on Euclid/Manhattan.
    """
    qdrant.create_collection(
        collection_name="test_collection",
        vectors_config={
            "dense": models.VectorParams(size=2, distance=distance),
            "cosine": models.VectorParams(size=2, distance=models.Distance.COSINE),
        },
    )
    qdrant.upsert(
        collection_name="test_collection",
        points=[
            models.PointStruct(id=i, vector={"dense": [float(i), 0.0], "cosine": [1.0, float(i)]})
            for i in range(5)
        ],
    )

    # a plain search is nearest-first: ids 0, 1, 2, 3, 4
    plain = qdrant.query_points(
        collection_name="test_collection", query=[0.0, 0.0], using="dense", limit=5
    ).points
    assert [point.id for point in plain] == [0, 1, 2, 3, 4]

    def fuse(prefetch: list[models.Prefetch]) -> list[int]:
        points = qdrant.query_points(
            collection_name="test_collection",
            prefetch=prefetch,
            query=models.FusionQuery(fusion=models.Fusion.DBSF),
            limit=5,
        ).points
        return [point.id for point in points]

    # both sources agree on nearest-first, so the fused order must agree too
    assert fuse(
        [
            models.Prefetch(query=[0.0, 0.0], using="dense", limit=5),
            models.Prefetch(query=[0.1, 0.0], using="dense", limit=5),
        ]
    ) == [0, 1, 2, 3, 4]

    # mixing with a bigger-is-better source must not flip the smaller-is-better one either
    assert (
        fuse(
            [
                models.Prefetch(query=[0.0, 0.0], using="dense", limit=5),
                models.Prefetch(query=[1.0, 0.0], using="cosine", limit=5),
            ]
        )[0]
        == 0
    )
