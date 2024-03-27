from qdrant_client.http import models
from qdrant_client.hybrid.fusion import reciprocal_rank_fusion


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
    assert fused[2].id in ["1", "5"]
