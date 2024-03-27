from typing import List, Optional

import numpy as np

from qdrant_client.conversions import common_types as types
from qdrant_client.http.models import SparseVector


def empty_sparse_vector() -> SparseVector:
    return SparseVector(
        indices=[],
        values=[],
    )


def validate_sparse_vector(vector: SparseVector) -> None:
    assert len(vector.indices) == len(
        vector.values
    ), "Indices and values must have the same length"
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
