import numpy as np
import pytest

from qdrant_client.http import models
from qdrant_client.hybrid.fusion import reciprocal_rank_fusion, distribution_based_score_fusion


def test_reciprocal_rank_fusion() -> None:
    responses = [
        [
            models.ScoredPoint(id="1", score=0.1, version=1),
            models.ScoredPoint(id="2", score=0.2, version=1),
            models.ScoredPoint(id="3", score=0.3, version=1),
        ],
        [
            models.ScoredPoint(id="5", score=12.0, version=1),
            models.ScoredPoint(id="6", score=8.0, version=1),
            models.ScoredPoint(id="7", score=5.0, version=1),
            models.ScoredPoint(id="2", score=3.0, version=1),
        ],
    ]

    fused = reciprocal_rank_fusion(responses)

    assert fused[0].id == "2"
    assert fused[1].id in ["1", "5"]
    assert np.isclose(fused[1].score, 1 / 2)
    assert fused[2].id in ["1", "5"]
    assert np.isclose(fused[2].score, 1 / 2)


def test_distribution_based_score_fusion() -> None:
    responses = [
        [
            models.ScoredPoint(id=1, version=0, score=85.0),
            models.ScoredPoint(id=0, version=0, score=76.0),
            models.ScoredPoint(id=5, version=0, score=68.0),
        ],
        [
            models.ScoredPoint(id=1, version=0, score=62.0),
            models.ScoredPoint(id=0, version=0, score=61.0),
            models.ScoredPoint(id=4, version=0, score=57.0),
            models.ScoredPoint(id=3, version=0, score=51.0),
            models.ScoredPoint(id=2, version=0, score=44.0),
        ],
    ]

    fused = distribution_based_score_fusion(responses, limit=3)

    assert fused[0].id == 1
    assert fused[1].id == 0
    assert fused[2].id == 4


@pytest.mark.parametrize(
    "k, expected_scores",
    [
        (None, [1.0, 3 / 5 + 1 / 3, 3 / 4, 1 / 4]),
        (2, [1.0, 3 / 5 + 1 / 3, 3 / 4, 1 / 4]),
        (60, [1 / 30, 3 / 179 + 1 / 61, 3 / 178, 1 / 62]),
    ],
)
@pytest.mark.parametrize("limit", [0, 2, 10])
def test_weighted_rrf_scores_and_limit(
    k: int | None, expected_scores: list[float], limit: int
) -> None:
    responses = [
        [models.ScoredPoint(id=i, score=100.0 - i, version=0) for i in (1, 2, 3)],
        [models.ScoredPoint(id=i, score=float(i), version=0) for i in (3, 2, 4)],
    ]

    fused = reciprocal_rank_fusion(
        responses, limit=limit, ranking_constant_k=k, weights=[3.0, 1.0]
    )

    # Weights scale ranks, not the original scores or the final reciprocal score.
    # Shared IDs accumulate contributions from both sources.
    assert [point.id for point in fused] == [3, 2, 1, 4][:limit]
    assert [point.score for point in fused] == pytest.approx(expected_scores[:limit])


def test_weighted_rrf_zero_weight() -> None:
    responses = [
        [models.ScoredPoint(id=i, score=10.0, version=0) for i in (1, 2)],
        [models.ScoredPoint(id=i, score=20.0, version=0) for i in (2, 3)],
    ]

    fused = reciprocal_rank_fusion(responses, weights=[0.0, 1.0])

    assert [point.id for point in fused] == [2, 3, 1]
    assert [point.score for point in fused] == pytest.approx([1 / 2, 1 / 3, 0.0])


def test_weighted_rrf_empty_source_preserves_weight_alignment() -> None:
    responses = [[], [models.ScoredPoint(id=1, score=99.0, version=0)]]

    fused = reciprocal_rank_fusion(responses, weights=[100.0, 3.0])

    assert [point.id for point in fused] == [1]
    assert fused[0].score == pytest.approx(3 / 4)


@pytest.mark.parametrize("weights", [[], [1.0], [1.0, 2.0, 3.0]])
def test_weighted_rrf_rejects_mismatched_weights(weights: list[float]) -> None:
    responses = [[models.ScoredPoint(id=1, score=10.0, version=0)], []]

    with pytest.raises(ValueError, match="Length of weights"):
        reciprocal_rank_fusion(responses, weights=weights)


def test_reciprocal_rank_fusion_empty_responses() -> None:
    responses: list[list[models.ScoredPoint]] = [[]]
    fused = reciprocal_rank_fusion(responses)
    assert fused == []

    responses = [
        [
            models.ScoredPoint(id="1", score=0.1, version=1),
            models.ScoredPoint(id="2", score=0.2, version=1),
            models.ScoredPoint(id="3", score=0.3, version=1),
        ],
        [],
    ]

    fused = reciprocal_rank_fusion(responses)

    assert fused[0].id == "1"
    assert np.isclose(fused[0].score, 1 / 2)
    assert fused[1].id == "2"
    assert np.isclose(fused[1].score, 1 / 3)
    assert fused[2].id == "3"
    assert np.isclose(fused[2].score, 1 / 4)


def test_distribution_based_score_fusion_empty_response() -> None:
    responses: list[list[models.ScoredPoint]] = [[]]
    fused = distribution_based_score_fusion(responses, limit=3)
    assert fused == []

    responses = [
        [
            models.ScoredPoint(id=1, version=0, score=85.0),
            models.ScoredPoint(id=0, version=0, score=76.0),
            models.ScoredPoint(id=5, version=0, score=68.0),
        ],
        [],
    ]

    fused = distribution_based_score_fusion(responses, limit=3)

    assert fused[0].id == 1
    assert fused[1].id == 0
    assert fused[2].id == 5


def test_distribution_based_score_fusion_zero_variance() -> None:
    score = 85.0
    responses = [
        [
            models.ScoredPoint(id=1, version=0, score=score),
            models.ScoredPoint(id=0, version=0, score=score),
            models.ScoredPoint(id=5, version=0, score=score),
        ],
        [],
    ]
    fused = distribution_based_score_fusion(
        [[models.ScoredPoint(id=1, version=0, score=score)]], limit=3
    )
    assert fused[0].id == 1
    assert fused[0].score == 0.5

    fused = distribution_based_score_fusion(responses, limit=3)
    assert len(fused) == 3
    assert all([p.score == 0.5 for p in fused])
