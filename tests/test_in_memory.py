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


def test_recommend_euclid_score_threshold(qdrant: QdrantClient):
    """Test that RecommendQuery on a EUCLID collection correctly filters by score_threshold.

    Recommend queries use synthetic sigmoid scores where higher is better.
    score_threshold should filter out points with scores below the threshold.
    """
    qdrant.create_collection(
        "recommend_euclid",
        vectors_config=models.VectorParams(size=2, distance=models.Distance.EUCLID),
    )
    qdrant.upsert(
        "recommend_euclid",
        points=[
            models.PointStruct(id=1, vector=[0.0, 0.0]),
            models.PointStruct(id=2, vector=[1.0, 0.0]),
            models.PointStruct(id=3, vector=[2.0, 0.0]),
            models.PointStruct(id=4, vector=[3.0, 0.0]),
            models.PointStruct(id=5, vector=[10.0, 0.0]),
        ],
    )
    query = models.RecommendQuery(
        recommend=models.RecommendInput(
            positive=[1], strategy=models.RecommendStrategy.BEST_SCORE
        )
    )

    no_threshold = qdrant.query_points("recommend_euclid", query=query, limit=10).points
    assert len(no_threshold) == 4
    assert [p.id for p in no_threshold] == [2, 3, 4, 5]

    # Scores: id 2 (~0.25), id 3 (~0.10), id 4 (~0.05), id 5 (~0.005)
    filtered = qdrant.query_points(
        "recommend_euclid", query=query, limit=10, score_threshold=0.05
    ).points
    assert len(filtered) == 3
    assert [p.id for p in filtered] == [2, 3, 4]
    for p in filtered:
        assert p.score >= 0.05


def test_discover_manhattan_score_threshold(qdrant: QdrantClient):
    """Test that DiscoverQuery on a MANHATTAN collection correctly filters by score_threshold."""
    qdrant.create_collection(
        "discover_manhattan",
        vectors_config=models.VectorParams(size=2, distance=models.Distance.MANHATTAN),
    )
    qdrant.upsert(
        "discover_manhattan",
        points=[
            models.PointStruct(id=1, vector=[0.0, 0.0]),
            models.PointStruct(id=2, vector=[1.0, 0.0]),
            models.PointStruct(id=3, vector=[5.0, 0.0]),
            models.PointStruct(id=4, vector=[1.2, 0.0]),
            models.PointStruct(id=5, vector=[2.0, 0.0]),
        ],
    )
    query = models.DiscoverQuery(
        discover=models.DiscoverInput(
            target=1, context=[models.ContextPair(positive=2, negative=3)]
        )
    )

    no_threshold = qdrant.query_points("discover_manhattan", query=query, limit=10).points
    assert len(no_threshold) > 0

    best_score = no_threshold[0].score
    worst_score = no_threshold[-1].score
    assert best_score >= worst_score

    # Threshold set between worst and best score
    mid_threshold = (best_score + worst_score) / 2.0
    filtered = qdrant.query_points(
        "discover_manhattan", query=query, limit=10, score_threshold=mid_threshold
    ).points
    assert len(filtered) > 0
    assert len(filtered) <= len(no_threshold)
    for p in filtered:
        assert p.score >= mid_threshold


def test_context_euclid_score_threshold(qdrant: QdrantClient):
    """Test that ContextQuery on a EUCLID collection correctly filters by score_threshold.

    Context queries compute synthetic scores where higher is better.
    """
    qdrant.create_collection(
        "context_euclid",
        vectors_config=models.VectorParams(size=2, distance=models.Distance.EUCLID),
    )
    qdrant.upsert(
        "context_euclid",
        points=[
            models.PointStruct(id=1, vector=[0.0, 0.0]),
            models.PointStruct(id=2, vector=[1.0, 0.0]),
            models.PointStruct(id=3, vector=[5.0, 0.0]),
            models.PointStruct(id=4, vector=[1.2, 0.0]),
            models.PointStruct(id=5, vector=[2.0, 0.0]),
        ],
    )
    query = models.ContextQuery(
        context=[models.ContextPair(positive=2, negative=3)]
    )

    no_threshold = qdrant.query_points("context_euclid", query=query, limit=10).points
    assert len(no_threshold) > 0

    best_score = no_threshold[0].score
    worst_score = no_threshold[-1].score
    assert best_score >= worst_score

    mid_threshold = (best_score + worst_score) / 2.0
    filtered = qdrant.query_points(
        "context_euclid", query=query, limit=10, score_threshold=mid_threshold
    ).points
    assert len(filtered) > 0
    assert len(filtered) <= len(no_threshold)
    for p in filtered:
        assert p.score >= mid_threshold


def test_multivector_queries_euclid_score_threshold(qdrant: QdrantClient):
    """Test that multivector Recommend, Discover, and Context queries on EUCLID apply score_threshold correctly."""
    qdrant.create_collection(
        "multi_euclid",
        vectors_config=models.VectorParams(
            size=2,
            distance=models.Distance.EUCLID,
            multivector_config=models.MultiVectorConfig(
                comparator=models.MultiVectorComparator.MAX_SIM
            ),
        ),
    )
    qdrant.upsert(
        "multi_euclid",
        points=[
            models.PointStruct(id=1, vector=[[0.0, 0.0], [0.1, 0.1]]),
            models.PointStruct(id=2, vector=[[1.0, 0.0], [1.1, 0.0]]),
            models.PointStruct(id=3, vector=[[2.0, 0.0], [2.1, 0.0]]),
            models.PointStruct(id=4, vector=[[5.0, 0.0], [5.1, 0.0]]),
        ],
    )
    # MultiRecoQuery
    reco_query = models.RecommendQuery(
        recommend=models.RecommendInput(positive=[1], strategy=models.RecommendStrategy.BEST_SCORE)
    )
    reco_no_th = qdrant.query_points("multi_euclid", query=reco_query, limit=10).points
    assert len(reco_no_th) == 3
    threshold = (reco_no_th[0].score + reco_no_th[-1].score) / 2.0
    reco_filtered = qdrant.query_points("multi_euclid", query=reco_query, limit=10, score_threshold=threshold).points
    expected_reco_ids = {p.id for p in reco_no_th if p.score >= threshold}
    assert {p.id for p in reco_filtered} == expected_reco_ids
    for p in reco_filtered:
        assert p.score >= threshold

    # MultiDiscoveryQuery
    disc_query = models.DiscoverQuery(
        discover=models.DiscoverInput(target=1, context=[models.ContextPair(positive=2, negative=4)])
    )
    disc_no_th = qdrant.query_points("multi_euclid", query=disc_query, limit=10).points
    assert len(disc_no_th) > 0
    disc_th = (disc_no_th[0].score + disc_no_th[-1].score) / 2.0
    disc_filtered = qdrant.query_points("multi_euclid", query=disc_query, limit=10, score_threshold=disc_th).points
    expected_disc_ids = {p.id for p in disc_no_th if p.score >= disc_th}
    assert {p.id for p in disc_filtered} == expected_disc_ids
    for p in disc_filtered:
        assert p.score >= disc_th

    # MultiContextQuery
    ctx_query = models.ContextQuery(
        context=[models.ContextPair(positive=2, negative=4)]
    )
    ctx_no_th = qdrant.query_points("multi_euclid", query=ctx_query, limit=10).points
    assert len(ctx_no_th) > 0
    ctx_th = (ctx_no_th[0].score + ctx_no_th[-1].score) / 2.0
    ctx_filtered = qdrant.query_points("multi_euclid", query=ctx_query, limit=10, score_threshold=ctx_th).points
    expected_ctx_ids = {p.id for p in ctx_no_th if p.score >= ctx_th}
    assert {p.id for p in ctx_filtered} == expected_ctx_ids
    for p in ctx_filtered:
        assert p.score >= ctx_th
