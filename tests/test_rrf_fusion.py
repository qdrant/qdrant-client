from qdrant_client.http import models
from qdrant_client.hybrid.fusion import reciprocal_rank_fusion


def test_rrf_preserves_source_scores_and_content():
    point = models.ScoredPoint(
        id=1, version=0, score=10.0, payload={"text": "document"}, vector=[1.0, 0.0]
    )
    result = reciprocal_rank_fusion([[point], [point]])

    assert point.score == 10.0
    assert len(result) == 1
    assert result[0].score == 1.0
    assert result[0].id == point.id
    assert result[0].payload == point.payload
    assert result[0].vector == point.vector


def test_rrf_reusing_responses_preserves_previous_results():
    point = models.ScoredPoint(id=1, version=0, score=10.0)
    responses = [[point]]

    first = reciprocal_rank_fusion(responses)
    second = reciprocal_rank_fusion(responses, weights=[2.0])

    assert first[0].score == 0.5
    assert second[0].score == 2.0 / 3.0
    assert point.score == 10.0
