from copy import deepcopy

import pytest

from qdrant_client.http import models
from qdrant_client.hybrid.fusion import distribution_based_score_fusion


def point(identifier: int, score: float) -> models.ScoredPoint:
    return models.ScoredPoint(id=identifier, score=score, version=0, payload={"id": identifier})


@pytest.mark.parametrize("scores", [[3.0, 2.0, 1.0], [5.0, 5.0], [8.0]])
def test_dbsf_preserves_inputs_and_repeat_results(scores: list[float]) -> None:
    responses = [[point(i, score) for i, score in enumerate(scores)], [point(0, 9.0)]]
    original = deepcopy(responses)

    first = distribution_based_score_fusion(responses, limit=10)
    first_snapshot = deepcopy(first)
    second = distribution_based_score_fusion(responses, limit=10)

    assert responses == original
    assert first == first_snapshot
    assert second == first_snapshot
    assert all(result is not source for result in first for group in responses for source in group)


def test_dbsf_shared_point_matches_independent_responses() -> None:
    shared = point(1, 8.0)
    responses = [[shared, point(2, 2.0)], [shared, point(3, 4.0)]]
    independent = [[deepcopy(item) for item in group] for group in responses]

    expected = distribution_based_score_fusion(independent, limit=10)
    actual = distribution_based_score_fusion(responses, limit=10)

    assert [item.id for item in actual] == [item.id for item in expected]
    assert [item.score for item in actual] == pytest.approx([item.score for item in expected])


def test_dbsf_expected_scores_and_limit() -> None:
    responses = [[point(1, 3.0), point(2, 1.0)], [point(1, 8.0)], []]
    results = distribution_based_score_fusion(responses, limit=1)
    assert [item.id for item in results] == [1]
    assert results[0].score == pytest.approx(1.0 + 1.0 / (6.0 * 2.0**0.5))
    assert distribution_based_score_fusion([[], []], limit=5) == []
