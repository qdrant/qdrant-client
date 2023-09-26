from enum import Enum
from typing import List
from qdrant_client.conversions import common_types as types

import numpy as np

from qdrant_client.http import models


class RecoQuery:
    def __init__(
        self, 
        positive: List[List[float]] = [], 
        negative: List[List[float]] = []
    ):
        self.positive: List[types.NumpyArray] = [np.array(vector) for vector in positive]
        self.negative: List[types.NumpyArray] = [np.array(vector) for vector in negative]


QueryVector = RecoQuery | types.NumpyArray

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
    vector_count = vectors.shape[0]
    
    # get all positive scores
    positive_scores: List[types.NumpyArray] = []
    for positive in query.positive:
        score = calculate_distance(positive, vectors, distance_type)
        positive_scores.append(score)
        
    # keep only max (or min) of each positive
    if distance_to_order(distance_type) == DistanceOrder.BIGGER_IS_BETTER:
        if len(positive_scores) == 0:
            positive_scores.append(np.full(vector_count, -np.inf))
        positive_scores = np.amax(np.array(positive_scores), axis=0)
    else:
        if len(positive_scores) == 0:
            positive_scores.append(np.full(vector_count, np.inf))
        positive_scores = np.amin(np.array(positive_scores), axis=0)

    # get all negative scores
    negative_scores: List[types.NumpyArray] = []
    for negative in query.negative:
        score = calculate_distance(negative, vectors, distance_type)
        negative_scores.append(score)
        
    # keep only max (or min) of each negative
    if distance_to_order(distance_type) == DistanceOrder.BIGGER_IS_BETTER:
        if len(negative_scores) == 0:
            negative_scores.append(np.full(vector_count, -np.inf))
        negative_scores = np.amax(np.array(negative_scores), axis=0)
    else:
        if len(negative_scores) == 0:
            negative_scores.append(np.full(vector_count, np.inf))
        negative_scores = np.amin(np.array(negative_scores), axis=0)
        
    # choose from best positive or best negative
    zip = np.stack((positive_scores, negative_scores), axis=1)
    
    if distance_to_order(distance_type) == DistanceOrder.BIGGER_IS_BETTER:
        cond_list = [zip[:,0] > zip[:,1], zip[:,0] <= zip[:,1]]
        choice_list = [zip[:,0], -(zip[:,1]**2)]
        return np.select(cond_list, choice_list, default=42)
    else:
        cond_list = [zip[:,0] < zip[:,1], zip[:,0] >= zip[:,1]]
        choice_list = [zip[:,0], zip[:,1]**2]
        return np.select(cond_list, choice_list, default=42)
        

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
