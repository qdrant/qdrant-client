from enum import Enum
from typing import List, Optional, Union
from qdrant_client.conversions import common_types as types

import numpy as np

from qdrant_client.http import models


class RecoQuery:
    def __init__(
        self, 
        positive: Optional[List[List[float]]] = None, 
        negative: Optional[List[List[float]]] = None
    ):
        positive = positive if positive is not None else []
        negative = negative if negative is not None else []
        self.positive: List[types.NumpyArray] = [np.array(vector) for vector in positive]
        self.negative: List[types.NumpyArray] = [np.array(vector) for vector in negative]


QueryVector = Union[RecoQuery, types.NumpyArray]

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


def cosine_similarity(query: np.ndarray, vectors: np.ndarray) -> np.ndarray:
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


def dot_product(query: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    """
    Calculate dot product between query and vectors
    Args:
        query: query vector.
        vectors: vectors to calculate distance with
    Returns:
        distances
    """
    return np.dot(vectors, query)


def euclidean_distance(query: np.ndarray, vectors: np.ndarray) -> np.ndarray:
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
    query: np.ndarray, vectors: np.ndarray, distance_type: models.Distance
) -> types.NumpyArray:
    if distance_type == models.Distance.COSINE:
        return cosine_similarity(query, vectors)
    elif distance_type == models.Distance.DOT:
        return dot_product(query, vectors)
    elif distance_type == models.Distance.EUCLID:
        return euclidean_distance(query, vectors)
    else:
        raise ValueError(f"Unknown distance type {distance_type}")


def calculate_best_scores(
    query: RecoQuery, vectors: np.ndarray, distance_type: models.Distance
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
    # in case of choosing best negative, square and negate it to make it smaller than any positive
    if distance_to_order(distance_type) == DistanceOrder.BIGGER_IS_BETTER:
        return np.where(pos > neg, pos, -(neg*neg))
    else:
        # neg*neg is not negated here because of the DistanceOrder.SMALLER_IS_BETTER
        return np.where(pos < neg, pos, neg*neg)
        

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
