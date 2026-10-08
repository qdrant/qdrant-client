import pytest

from qdrant_client import QdrantClient
from qdrant_client.http import models


def _make_euclid_like(distance) -> QdrantClient:
    client = QdrantClient(":memory:")
    client.create_collection("c", vectors_config=models.VectorParams(size=2, distance=distance))
    client.upsert(
        "c",
        points=[
            models.PointStruct(id=1, vector=[0.0, 0.0]),
            models.PointStruct(id=2, vector=[1.0, 0.0]),
            models.PointStruct(id=3, vector=[2.0, 0.0]),
            models.PointStruct(id=4, vector=[3.0, 0.0]),
            models.PointStruct(id=5, vector=[10.0, 0.0]),
        ],
    )
    return client


@pytest.mark.parametrize("distance", [models.Distance.EUCLID, models.Distance.MANHATTAN])
def test_recommend_best_score_threshold_not_inverted_on_smaller_is_better(distance):
    """https://github.com/qdrant/qdrant-client/issues/1370

    `best_score` Recommend scores are a bigger-is-better sigmoid even on a Euclidean/Manhattan
    collection, so `score_threshold` must keep points above the threshold instead of dropping the
    whole result set on the first (best-scoring) point.
    """
    client = _make_euclid_like(distance)
    query = models.RecommendQuery(
        recommend=models.RecommendInput(positive=[1], strategy=models.RecommendStrategy.BEST_SCORE)
    )

    points = client.query_points("c", query=query, limit=10, score_threshold=0.1).points
    ids = [p.id for p in points]

    # Closest point (id=2) has the highest synthetic score (~0.25) and must survive the threshold.
    assert 2 in ids, f"score_threshold dropped the best-scoring point: {ids}"
    # Farthest point (id=5) has the lowest synthetic score (~0.005) and must be excluded.
    assert 5 not in ids, f"score_threshold kept a sub-threshold point: {ids}"


@pytest.mark.parametrize("distance", [models.Distance.EUCLID, models.Distance.MANHATTAN])
def test_recommend_sum_scores_threshold_is_bigger_is_better(distance):
    client = _make_euclid_like(distance)
    query = models.RecommendQuery(
        recommend=models.RecommendInput(positive=[1], strategy=models.RecommendStrategy.SUM_SCORES)
    )
    # sum_scores yields negative distances (closer = larger). Threshold -5 keeps id=2 and id=3
    # (score -1 / -4 or -2) but drops id=4 and id=5.
    points = client.query_points("c", query=query, limit=10, score_threshold=-5).points
    ids = [p.id for p in points]
    assert 2 in ids and 3 in ids, ids
    assert 5 not in ids, ids


@pytest.mark.parametrize("distance", [models.Distance.EUCLID, models.Distance.MANHATTAN])
def test_context_query_threshold_not_inverted_on_smaller_is_better(distance):
    client = _make_euclid_like(distance)
    # Context scoring is also a bigger-is-better synthetic score, independent of the raw distance.
    query = models.ContextQuery(context=[models.ContextPair(positive=1, negative=5)])
    points = client.query_points("c", query=query, limit=10, score_threshold=-0.5).points
    ids = [p.id for p in points]
    # Regression: on a smaller-is-better collection the cut-off used to break on the first point
    # and return an empty result set. With the fix, above-threshold points are retained.
    assert ids, f"context query with threshold returned an empty result on {distance}"
    assert 2 in ids, ids


def test_cosine_recommend_threshold_unaffected():
    # Control: the existing bigger-is-better (cosine) path must keep its semantics.
    client = QdrantClient(":memory:")
    client.create_collection(
        "c", vectors_config=models.VectorParams(size=2, distance=models.Distance.COSINE)
    )
    client.upsert(
        "c",
        points=[
            models.PointStruct(id=1, vector=[1.0, 0.0]),
            models.PointStruct(id=2, vector=[0.0, 1.0]),
            models.PointStruct(id=3, vector=[1.0, 0.0]),
        ],
    )
    query = models.RecommendQuery(
        recommend=models.RecommendInput(positive=[1], strategy=models.RecommendStrategy.BEST_SCORE)
    )
    points = client.query_points("c", query=query, limit=10, score_threshold=0.6).points
    ids = [p.id for p in points]
    # id=3 is parallel to the positive (synthetic score ~0.75) and must survive; id=2 is
    # orthogonal (~0.5) and must be excluded. The positive example id=1 is not returned.
    assert set(ids) == {3}, ids


def test_raw_nearest_threshold_smaller_is_better_unaffected():
    # Control: an ordinary nearest (raw vector) query on Euclid still uses smaller-is-better,
    # proving the non-reco cut-off branch is untouched.
    client = _make_euclid_like(models.Distance.EUCLID)
    points = client.query_points("c", query=[0.0, 0.0], limit=10, score_threshold=1.5).points
    ids = [p.id for p in points]
    # Distances from origin: id=1 -> 0, id=2 -> 1, id=3 -> 2. Only id=1 (0) and id=2 (1) are <= 1.5.
    assert set(ids) == {1, 2}, ids
