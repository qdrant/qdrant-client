import pytest

from qdrant_client import AsyncQdrantClient, QdrantClient, models
from qdrant_client.hybrid.fusion import reciprocal_rank_fusion


@pytest.mark.parametrize(
    "query",
    [models.FusionQuery(fusion=models.Fusion.RRF), models.RrfQuery(rrf=models.Rrf(k=2))],
)
@pytest.mark.parametrize(
    "threshold, offset, expected_ids",
    [
        (0.3333333432674408, 0, [1, 2]),
        (0.3333333432674408, 1, [2]),
        (0.0, 0, [1, 2, 3]),
        (0.3333334, 0, [1]),
        (0.5, 0, [1]),
    ],
)
def test_rrf_score_threshold_uses_float32_scores(
    query: models.Query, threshold: float, offset: int, expected_ids: list[int]
) -> None:
    client = QdrantClient(":memory:")
    try:
        client.create_collection(
            "precision", vectors_config=models.VectorParams(size=2, distance=models.Distance.DOT)
        )
        client.upsert(
            "precision",
            points=[models.PointStruct(id=i, vector=[float(4 - i), 0.0]) for i in (1, 2, 3)],
        )
        # The second rank scores 1/3, rounded to float32 by the server.
        result = client.query_points(
            "precision",
            prefetch=models.Prefetch(query=[1.0, 0.0], limit=3),
            query=query,
            score_threshold=threshold,
            offset=offset,
        ).points
        assert [point.id for point in result] == expected_ids
        if 2 in expected_ids:
            assert next(point.score for point in result if point.id == 2) == 0.3333333432674408
    finally:
        client.close()


@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.asyncio
async def test_rrf_accumulates_float32_scores(nested: bool) -> None:
    client = AsyncQdrantClient(":memory:")
    try:
        await client.create_collection(
            "precision", vectors_config=models.VectorParams(size=2, distance=models.Distance.DOT)
        )
        await client.upsert(
            "precision",
            points=[
                models.PointStruct(id=1, vector=[1.0, 0.0]),
                models.PointStruct(id=2, vector=[0.0, 1.0]),
            ],
        )
        query = models.FusionQuery(fusion=models.Fusion.RRF)
        prefetch = [
            models.Prefetch(query=[1.0, 0.0], limit=2),
            models.Prefetch(query=[0.0, 1.0], limit=2),
        ]
        # f32(0.5) + f32(1/3), not a final-only rounding of 0.5 + 1/3.
        threshold = 0.8333333730697632
        if nested:
            prefetch = [
                models.Prefetch(prefetch=prefetch, query=query, score_threshold=threshold, limit=2)
            ]
        result = await client.query_points(
            "precision",
            prefetch=prefetch,
            query=query,
            score_threshold=0.0 if nested else threshold,
        )
        assert sorted(point.id for point in result.points) == [1, 2]
        if not nested:
            assert all(point.score == threshold for point in result.points)
    finally:
        await client.close()


@pytest.mark.parametrize(
    "weights, expected_scores",
    [
        (None, [0.8333333730697632, 0.8333333730697632]),
        ([3.0, 1.0], [1.0999999046325684, 1.0833333730697632]),
        ([0.0, 1.0], [0.5, 0.3333333432674408]),
    ],
)
def test_rrf_score_precision_with_multiple_sources(
    weights: list[float] | None, expected_scores: list[float]
) -> None:
    responses = [
        [models.ScoredPoint(id=i, score=1.0, version=0) for i in (1, 2)],
        [models.ScoredPoint(id=i, score=1.0, version=0) for i in (2, 1)],
    ]
    result = reciprocal_rank_fusion(responses, weights=weights)
    assert [point.score for point in result] == expected_scores
