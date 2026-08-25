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


def test_facet_with_mixed_type_values_on_count_tie(qdrant: QdrantClient):
    # A payload key can hold different scalar types across points. When two
    # facet values have the same count, the tie-break used to compare the
    # values directly, which raises TypeError for e.g. int vs str.
    qdrant.create_collection(
        collection_name="test_collection",
        vectors_config=models.VectorParams(size=2, distance=models.Distance.DOT),
    )
    qdrant.upsert(
        collection_name="test_collection",
        points=[
            models.PointStruct(id=1, vector=[0.1, 0.2], payload={"k": 5}),
            models.PointStruct(id=2, vector=[0.2, 0.3], payload={"k": "x"}),
            models.PointStruct(id=3, vector=[0.3, 0.4], payload={"k": False}),
        ],
    )

    response = qdrant.facet(collection_name="test_collection", key="k")

    # Each value has count 1, so all three are a mutual tie. The call must not
    # raise, and the ordering must be deterministic: values are grouped by type
    # (numbers before strings) and ordered within their group.
    assert {(hit.value, hit.count) for hit in response.hits} == {(5, 1), ("x", 1), (False, 1)}
    assert [hit.value for hit in response.hits] == [False, 5, "x"]


def test_facet_tie_break_is_deterministic_across_colliding_types(qdrant: QdrantClient):
    # `""` (str) and `0` (int) are distinct facet values that an id-style key
    # collides onto the same tie-break key, which would make the order of
    # equal-count values depend on the upsert order. The facet ordering must stay
    # deterministic regardless of the order the points are upserted.
    def facet_order(point_order):
        qdrant.delete_collection(collection_name="test_collection")
        qdrant.create_collection(
            collection_name="test_collection",
            vectors_config=models.VectorParams(size=2, distance=models.Distance.DOT),
        )
        qdrant.upsert(collection_name="test_collection", points=point_order)
        response = qdrant.facet(collection_name="test_collection", key="k")
        assert {(hit.value, hit.count) for hit in response.hits} == {("", 1), (0, 1)}
        return [hit.value for hit in response.hits]

    points = [
        models.PointStruct(id=1, vector=[0.1, 0.2], payload={"k": ""}),
        models.PointStruct(id=2, vector=[0.2, 0.3], payload={"k": 0}),
    ]

    # The order must be identical no matter how the points were inserted.
    assert facet_order(points) == facet_order(list(reversed(points)))


def test_search_matrix_with_mixed_int_and_uuid_ids(qdrant: QdrantClient):
    # Qdrant allows integer and UUID (string) point ids to coexist in one
    # collection. Sorting the sampled points by raw id used to raise TypeError
    # when comparing an int id against a UUID string id.
    qdrant.create_collection(
        collection_name="test_collection",
        vectors_config=models.VectorParams(size=2, distance=models.Distance.DOT),
    )
    qdrant.upsert(
        collection_name="test_collection",
        points=[
            models.PointStruct(id=1, vector=[0.1, 0.9]),
            models.PointStruct(id="550e8400-e29b-41d4-a716-446655440000", vector=[0.9, 0.1]),
            models.PointStruct(id=7, vector=[0.2, 0.8]),
        ],
    )

    pairs_response = qdrant.search_matrix_pairs(
        collection_name="test_collection", sample=10, limit=1
    )
    assert len(pairs_response.pairs) > 0

    offsets_response = qdrant.search_matrix_offsets(
        collection_name="test_collection", sample=10, limit=1
    )
    assert len(offsets_response.ids) == 3
