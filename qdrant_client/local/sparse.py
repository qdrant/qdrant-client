from typing import List, Optional, Sequence

import numpy as np

from qdrant_client.conversions import common_types as types
from qdrant_client.local.distances import scaled_fast_sigmoid
from qdrant_client.models import Distance, SparseVector


class SparseRecoQuery:
    def __init__(
        self,
        positive: Optional[List[SparseVector]] = None,
        negative: Optional[List[SparseVector]] = None,
    ):
        positive = positive if positive is not None else []
        negative = negative if negative is not None else []

        for i, vector in enumerate(positive):
            validate_sparse_vector(vector)
            positive[i] = sort_sparse_vector(vector)

        for i, vector in enumerate(negative):
            validate_sparse_vector(vector)
            negative[i] = sort_sparse_vector(vector)

        self.positive = positive
        self.negative = negative


def empty_sparse_vector() -> SparseVector:
    return SparseVector(
        indices=[],
        values=[],
    )


def validate_sparse_vector(vector: SparseVector) -> None:
    assert len(vector.indices) == len(
        vector.values
    ), "Indices and values must have the same length"
    assert not np.isnan(vector.values).any(), "Values must not contain NaN"
    assert len(vector.indices) == len(set(vector.indices)), "Indices must be unique"


def is_sorted(vector: SparseVector) -> bool:
    for i in range(1, len(vector.indices)):
        if vector.indices[i] < vector.indices[i - 1]:
            return False
    return True


def sort_sparse_vector(vector: SparseVector) -> SparseVector:
    if is_sorted(vector):
        return vector

    sorted_indices = np.argsort(vector.indices)
    return SparseVector(
        indices=[vector.indices[i] for i in sorted_indices],
        values=[vector.values[i] for i in sorted_indices],
    )


def calculate_distance_sparse(
    query: SparseVector, vectors: List[SparseVector]
) -> types.NumpyArray:
    scores = []

    for vector in vectors:
        score = sparse_dot_product(query, vector)
        if score is not None:
            scores.append(score)
        else:
            # means no overlap
            scores.append(np.float32("-inf"))

    return np.array(scores, dtype=np.float32)


def calculate_sparse_recommend_best_scores(
    query: SparseRecoQuery, vectors: List[SparseVector]
) -> types.NumpyArray:
    def get_best_scores(examples: List[SparseVector]) -> types.NumpyArray:
        vector_count = len(vectors)

        # Get scores to all examples
        scores: List[types.NumpyArray] = []
        for example in examples:
            score = calculate_distance_sparse(example, vectors)
            scores.append(score)

        # Keep only max for each vector
        if len(scores) == 0:
            scores.append(np.full(vector_count, -np.inf))
        best_scores = np.array(scores, dtype=np.float32).max(axis=0)

        return best_scores

    pos = get_best_scores(query.positive)
    neg = get_best_scores(query.negative)

    # Choose from best positive or best negative,
    # in in both cases we apply sigmoid and then negate depending on the order
    return np.where(
        pos > neg,
        np.fromiter((scaled_fast_sigmoid(xi) for xi in pos), pos.dtype),
        np.fromiter((-scaled_fast_sigmoid(xi) for xi in neg), neg.dtype),
    )


# Expects sorted indices
# Returns None if no overlap
def sparse_dot_product(vector1: SparseVector, vector2: SparseVector) -> Optional[np.float32]:
    result = 0.0
    i, j = 0, 0
    overlap = False

    assert is_sorted(vector1), "Query sparse vector must be sorted"
    assert is_sorted(vector2), "Sparse vector to compare with must be sorted"

    while i < len(vector1.indices) and j < len(vector2.indices):
        if vector1.indices[i] == vector2.indices[j]:
            overlap = True
            result += vector1.values[i] * vector2.values[j]
            i += 1
            j += 1
        elif vector1.indices[i] < vector2.indices[j]:
            i += 1
        else:
            j += 1

    if overlap:
        return np.float32(result)
    else:
        return None


# Expects sorted indices
def combine_aggregate(vector1: SparseVector, vector2: SparseVector, op) -> SparseVector:
    result = empty_sparse_vector()
    i, j = 0, 0
    while i < len(vector1.indices) and j < len(vector2.indices):
        if vector1.indices[i] == vector2.indices[j]:
            result.indices.append(vector1.indices[i])
            result.values.append(op(vector1.values[i], vector2.values[j]))
            i += 1
            j += 1
        elif vector1.indices[i] < vector2.indices[j]:
            result.indices.append(vector1.indices[i])
            result.values.append(op(vector1.values[i], 0.0))
            i += 1
        else:
            result.indices.append(vector2.indices[j])
            result.values.append(op(0.0, vector2.values[j]))
            j += 1

    while i < len(vector1.indices):
        result.indices.append(vector1.indices[i])
        result.values.append(op(vector1.values[i], 0.0))
        i += 1

    while j < len(vector2.indices):
        result.indices.append(vector2.indices[j])
        result.values.append(op(0.0, vector2.values[j]))
        j += 1

    return result


# Expects sorted indices
def sparse_avg(vectors: Sequence[SparseVector]) -> SparseVector:
    result = empty_sparse_vector()
    if len(vectors) == 0:
        return result

    sparse_count = 0
    for vector in vectors:
        sparse_count += 1
        result = combine_aggregate(result, vector, lambda v1, v2: v1 + v2)

    result.values = np.divide(result.values, sparse_count).tolist()
    return result


# Expects sorted indices
def merge_positive_and_negative_avg(
    positive: SparseVector, negative: SparseVector
) -> SparseVector:
    return combine_aggregate(positive, negative, lambda pos, neg: pos + pos - neg)
