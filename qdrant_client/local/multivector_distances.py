from enum import Enum
from typing import List, Optional, Union

import numpy as np

from qdrant_client.conversions import common_types as types
from qdrant_client.http import models
from qdrant_client.local.distances import calculate_distance, distance_to_order

EPSILON = 1.1920929e-7  # https://doc.rust-lang.org/std/f32/constant.EPSILON.html
# https://github.com/qdrant/qdrant/blob/7164ac4a5987d28f1c93f5712aef8e09e7d93555/lib/segment/src/spaces/simple_avx.rs#L99C10-L99C10


class DistanceOrder(str, Enum):
    BIGGER_IS_BETTER = "bigger_is_better"
    SMALLER_IS_BETTER = "smaller_is_better"


class MultiRecoQuery:
    def __init__(
        self,
        positive: Optional[List[List[List[float]]]] = None,  # list of matrices
        negative: Optional[List[List[List[float]]]] = None,  # list of matrices
    ):
        positive = positive if positive is not None else []
        negative = negative if negative is not None else []

        self.positive: List[types.NumpyArray] = [np.array(vector) for vector in positive]
        self.negative: List[types.NumpyArray] = [np.array(vector) for vector in negative]

        assert not np.isnan(self.positive).any(), "Positive vectors must not contain NaN"
        assert not np.isnan(self.negative).any(), "Negative vectors must not contain NaN"


class MultiContextPair:
    def __init__(self, positive: List[List[float]], negative: List[List[float]]):
        self.positive: types.NumpyArray = np.array(positive)
        self.negative: types.NumpyArray = np.array(negative)

        assert not np.isnan(self.positive).any(), "Positive vector must not contain NaN"
        assert not np.isnan(self.negative).any(), "Negative vector must not contain NaN"


class MultiDiscoveryQuery:
    def __init__(self, target: List[List[float]], context: List[MultiContextPair]):
        self.target: types.NumpyArray = np.array(target)
        self.context = context

        assert not np.isnan(self.target).any(), "Target vector must not contain NaN"


class MultiContextQuery:
    def __init__(self, context_pairs: List[MultiContextPair]):
        self.context_pairs = context_pairs


MultiQueryVector = Union[
    MultiDiscoveryQuery,
    MultiContextQuery,
    MultiRecoQuery,
    # types.NumpyArray,
]


def max_sim(
    query: types.NumpyArray, matrices: List[types.NumpyArray], distance: models.Distance
) -> types.NumpyArray:
    """
    Calculate max similarity between query and matrices
    Args:
        query: query matrix: shape [query_token_nums, dim]
        matrices: matrices to calculate distance with, list of numpy matrices: shape [matrix_token_nums, dim]
    Returns:
        distances
    """
    similarities = []

    for matrix in matrices:
        sim_matrix = calculate_distance(query, matrix, distance)
        op = np.max if distance_to_order(distance) == DistanceOrder.BIGGER_IS_BETTER else np.min
        similarity = np.sum(op(sim_matrix, axis=-1))
        similarities.append(similarity)

    return np.array(similarities)
