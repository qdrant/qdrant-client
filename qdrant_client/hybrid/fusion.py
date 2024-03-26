from typing import Dict, List

from qdrant_client.http import models


def reciprocal_rank_fusion(
    responses: List[List[models.ScoredPoint]], limit: int = 10
) -> List[models.ScoredPoint]:
    def compute_score(pos: int) -> float:
        ranking_constant = (
            2  # the constant mitigates the impact of high rankings by outlier systems
        )
        return 1 / (ranking_constant + pos)

    scores: Dict[models.ExtendedPointId, float] = {}
    point_pile = {}
    for response in responses:
        for i, scored_point in enumerate(response):
            if scored_point.id in scores:
                scores[scored_point.id] += compute_score(i)
            else:
                point_pile[scored_point.id] = scored_point
                scores[scored_point.id] = compute_score(i)

    sorted_scores = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    sorted_points = []
    for point_id, score in sorted_scores[:limit]:
        point = point_pile[point_id]
        point.score = score
        sorted_points.append(point)
    return sorted_points
