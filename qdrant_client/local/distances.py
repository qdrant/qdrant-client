from enum import Enum
from typing import List, Optional, Tuple, Union

import numpy as np

from qdrant_client.conversions import common_types as types
from qdrant_client.http import models

EPSILON = 1.1920929e-7  # https://doc.rust-lang.org/std/f32/constant.EPSILON.html
# https://github.com/qdrant/qdrant/blob/7164ac4a5987d28f1c93f5712aef8e09e7d93555/lib/segment/src/spaces/simple_avx.rs#L99C10-L99C10


class RecoQuery:
    def __init__(
        self,
        positive: Optional[List[List[float]]] = None,
        negative: Optional[List[List[float]]] = None,
    ):
        positive = positive if positive is not None else []
        negative = negative if negative is not None else []
        self.positive: List[types.NumpyArray] = [np.array(vector) for vector in positive]
        self.negative: List[types.NumpyArray] = [np.array(vector) for vector in negative]


class DiscoveryQuery:
    def __init__(self, target: List[float], context_pairs: List[Tuple[List[float], List[float]]]):
        self.target: types.NumpyArray = np.array(target)
        self.context_pairs: List[Tuple[types.NumpyArray, types.NumpyArray]] = [
            (np.array(pair[0]), np.array(pair[1])) for pair in context_pairs
        ]


class ContextQuery:
    def __init__(self, context_pairs: List[Tuple[List[float], List[float]]]):
        self.context_pairs: List[Tuple[types.NumpyArray, types.NumpyArray]] = [
            (np.array(pair[0]), np.array(pair[1])) for pair in context_pairs
        ]


QueryVector = Union[DiscoveryQuery, ContextQuery, RecoQuery, types.NumpyArray]


class DistanceOrder(str, Enum):
    BIGGER_IS_BETTER = "bigger_is_better"
    SMALLER_IS_BETTER = "smaller_is_better"


def distance_to_order(distance: models.Distance) -> DistanceOrder:
    """
    Convert distance to order
    Args:
        distance: distance to convert
    Returns:
        order
    """
    if distance == models.Distance.EUCLID:
        return DistanceOrder.SMALLER_IS_BETTER

    return DistanceOrder.BIGGER_IS_BETTER


def cosine_similarity(query: types.NumpyArray, vectors: types.NumpyArray) -> types.NumpyArray:
    """
    Calculate cosine distance between query and vectors
    Args:
        query: query vector
        vectors: vectors to calculate distance with
    Returns:
        distances
    """
    query = query / np.linalg.norm(query)
    vectors = vectors / np.linalg.norm(vectors, axis=1)[:, np.newaxis]
    return np.dot(vectors, query)


def dot_product(query: types.NumpyArray, vectors: types.NumpyArray) -> types.NumpyArray:
    """
    Calculate dot product between query and vectors
    Args:
        query: query vector.
        vectors: vectors to calculate distance with
    Returns:
        distances
    """
    return np.dot(vectors, query)


def euclidean_distance(query: types.NumpyArray, vectors: types.NumpyArray) -> types.NumpyArray:
    """
    Calculate euclidean distance between query and vectors
    Args:
        query: query vector.
        vectors: vectors to calculate distance with
    Returns:
        distances
    """
    return np.linalg.norm(vectors - query, axis=1)


def calculate_distance(
    query: types.NumpyArray, vectors: types.NumpyArray, distance_type: models.Distance
) -> types.NumpyArray:
    if distance_type == models.Distance.COSINE:
        return cosine_similarity(query, vectors)
    elif distance_type == models.Distance.DOT:
        return dot_product(query, vectors)
    elif distance_type == models.Distance.EUCLID:
        return euclidean_distance(query, vectors)
    else:
        raise ValueError(f"Unknown distance type {distance_type}")


def scaled_fast_sigmoid(x: np.float32) -> np.float32:
    if np.isfinite(x):
        return 0.5 * (x / (1.0 + abs(x)) + 1.0)
    else:
        # To avoid NaNs, which gets: RuntimeWarning: invalid value encountered in scalar divide
        return x


def calculate_recommend_best_scores(
    query: RecoQuery, vectors: types.NumpyArray, distance_type: models.Distance
) -> types.NumpyArray:
    def get_best_scores(examples: List[types.NumpyArray]) -> types.NumpyArray:
        vector_count = vectors.shape[0]

        # Get scores to all examples
        scores: List[types.NumpyArray] = []
        for example in examples:
            score = calculate_distance(example, vectors, distance_type)
            scores.append(score)

        # Keep only max (or min) for each vector
        if distance_to_order(distance_type) == DistanceOrder.BIGGER_IS_BETTER:
            if len(scores) == 0:
                scores.append(np.full(vector_count, -np.inf))
            best_scores = np.array(scores, dtype=np.float32).max(axis=0)
        else:
            if len(scores) == 0:
                scores.append(np.full(vector_count, np.inf))
            best_scores = np.array(scores, dtype=np.float32).min(axis=0)

        return best_scores

    pos = get_best_scores(query.positive)
    neg = get_best_scores(query.negative)

    # Choose from best positive or best negative,
    # in in both cases we apply sigmoid and then negate depending on the order
    if distance_to_order(distance_type) == DistanceOrder.BIGGER_IS_BETTER:
        return np.where(
            pos > neg,
            np.fromiter((scaled_fast_sigmoid(xi) for xi in pos), pos.dtype),
            np.fromiter((-scaled_fast_sigmoid(xi) for xi in neg), neg.dtype),
        )
    else:
        # negative option is not negated here because of the DistanceOrder.SMALLER_IS_BETTER
        return np.where(
            pos < neg,
            np.fromiter((-scaled_fast_sigmoid(xi) for xi in pos), pos.dtype),
            np.fromiter((scaled_fast_sigmoid(xi) for xi in neg), neg.dtype),
        )


def calculate_discovery_ranks(
    context_pairs: List[Tuple[types.NumpyArray, types.NumpyArray]],
    vectors: types.NumpyArray,
    distance_type: models.Distance,
) -> types.NumpyArray:
    overall_ranks = np.zeros(vectors.shape[0], dtype=np.int32)
    for pair in context_pairs:
        pos = calculate_distance(pair[0], vectors, distance_type)  # distances to positive
        neg = calculate_distance(pair[1], vectors, distance_type)  # distances to negative

        if distance_to_order(distance_type) == DistanceOrder.BIGGER_IS_BETTER:
            pair_ranks = np.array(
                [
                    1 if is_bigger else 0 if is_equal else -1
                    for is_bigger, is_equal in zip(pos > neg, pos == neg)
                ]
            )
        else:
            pair_ranks = np.array(
                [
                    1 if is_smaller else 0 if is_equal else -1
                    for is_smaller, is_equal in zip(pos < neg, pos == neg)
                ]
            )
        overall_ranks += pair_ranks

    return overall_ranks


def calculate_discovery_scores(
    query: DiscoveryQuery, vectors: types.NumpyArray, distance_type: models.Distance
) -> types.NumpyArray:
    ranks = calculate_discovery_ranks(query.context_pairs, vectors, distance_type)

    distances_to_target = calculate_distance(query.target, vectors, distance_type)

    sigmoided_distances = np.fromiter(
        (scaled_fast_sigmoid(xi) for xi in distances_to_target), distances_to_target.dtype
    )

    return ranks + sigmoided_distances


def calculate_context_scores(
    query: ContextQuery, vectors: types.NumpyArray, distance_type: models.Distance
) -> types.NumpyArray:
    overall_scores = np.zeros(vectors.shape[0], dtype=np.float32)
    for pair in query.context_pairs:
        pos = calculate_distance(pair[0], vectors, distance_type)  # distances to positive
        neg = calculate_distance(pair[1], vectors, distance_type)  # distances to negative

        # Adjust
        if distance_type == models.Distance.EUCLID:
            pos = -(pos * pos)
            neg = -(neg * neg)

        difference = pos - neg - EPSILON
        pair_scores = np.minimum(difference, 0.0)
        overall_scores += pair_scores

    return overall_scores


def test_distances() -> None:
    query = np.array([1.0, 2.0, 3.0])
    vectors = np.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
    assert np.allclose(calculate_distance(query, vectors, models.Distance.COSINE), [1.0, 1.0])
    assert np.allclose(calculate_distance(query, vectors, models.Distance.DOT), [14.0, 14.0])
    assert np.allclose(calculate_distance(query, vectors, models.Distance.EUCLID), [0.0, 0.0])

    query = np.array([1.0, 0.0, 1.0])
    vectors = np.array([[1.0, 2.0, 3.0], [0.0, 1.0, 0.0]])

    assert np.allclose(
        calculate_distance(query, vectors, models.Distance.COSINE),
        [0.75592895, 0.0],
        atol=0.0001,
    )
    assert np.allclose(
        calculate_distance(query, vectors, models.Distance.DOT), [4.0, 0.0], atol=0.0001
    )
    assert np.allclose(
        calculate_distance(query, vectors, models.Distance.EUCLID),
        [2.82842712, 1.7320508],
        atol=0.0001,
    )
