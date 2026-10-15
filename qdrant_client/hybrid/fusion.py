from qdrant_client.http import models


DEFAULT_RANKING_CONSTANT_K = 2


def reciprocal_rank_fusion(
    responses: list[list[models.ScoredPoint]],
    limit: int = 10,
    ranking_constant_k: int | None = None,
    weights: list[float] | None = None,
) -> list[models.ScoredPoint]:
    if weights is not None and len(weights) != len(responses):
        raise ValueError("Length of weights must match the number of responses in RRF")

    ranking_constant = (
        ranking_constant_k if ranking_constant_k is not None else DEFAULT_RANKING_CONSTANT_K
    )  # mitigates the impact of high rankings by outlier systems

    def compute_score(pos: int, score_weight: float = 1.0) -> float:
        if score_weight <= 0:
            return 0.0
        return 1 / ((pos + 1.0) / score_weight + ranking_constant - 1.0)

    scores: dict[models.ExtendedPointId, float] = {}
    point_pile = {}
    for response_idx, response in enumerate(responses):
        weight = weights[response_idx] if weights is not None else 1.0
        for i, scored_point in enumerate(response):
            if scored_point.id in scores:
                scores[scored_point.id] += compute_score(i, weight)
            else:
                point_pile[scored_point.id] = scored_point
                scores[scored_point.id] = compute_score(i, weight)

    sorted_scores = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    sorted_points = []
    for point_id, score in sorted_scores[:limit]:
        point = point_pile[point_id]
        point.score = score
        sorted_points.append(point)
    return sorted_points


def distribution_based_score_fusion(
    responses: list[list[models.ScoredPoint]],
    limit: int,
    smaller_is_better: list[bool] | None = None,
) -> list[models.ScoredPoint]:
    """Distribution-based score fusion.

    Args:
        responses: lists of scored points to fuse, each already sorted best-first.
        limit: how many points to return.
        smaller_is_better: for each response, whether a lower score means a better
            match (Euclid/Manhattan nearest-neighbour searches). Those scores are
            negated before normalization so that, like in core, every normalized
            score is oriented "bigger is better" before being summed up.
    """
    if smaller_is_better is not None and len(smaller_is_better) != len(responses):
        raise ValueError("Length of smaller_is_better must match the number of responses in DBSF")

    def normalize(response: list[models.ScoredPoint]) -> list[models.ScoredPoint]:
        if len(response) == 1:
            response[0].score = 0.5
            return response

        total = sum([point.score for point in response])
        mean = total / len(response)
        variance = sum([(point.score - mean) ** 2 for point in response]) / (len(response) - 1)

        if variance == 0:
            for point in response:
                point.score = 0.5
            return response

        std_dev = variance**0.5
        low = mean - 3 * std_dev
        high = mean + 3 * std_dev

        for point in response:
            point.score = (point.score - low) / (high - low)

        return response

    points_map: dict[models.ExtendedPointId, models.ScoredPoint] = {}
    for response_idx, response in enumerate(responses):
        if not response:
            continue
        if smaller_is_better is not None and smaller_is_better[response_idx]:
            for point in response:
                point.score = -point.score
        normalized = normalize(response)
        for point in normalized:
            entry = points_map.get(point.id)
            if entry is None:
                points_map[point.id] = point
            else:
                entry.score += point.score

    sorted_points = sorted(points_map.values(), key=lambda item: item.score, reverse=True)

    return sorted_points[:limit]
