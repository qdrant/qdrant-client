from qdrant_client.http.models import SparseVector
from typing import List
import numpy as np


def empty_sparse_vector() -> SparseVector:
    return SparseVector(
        indices=np.array([], dtype=np.int64),
        values=np.array([], dtype=np.float32),
    )


def validate_sparse_vector(vector: SparseVector) -> None:
    assert len(vector.indices) == len(vector.values)
    assert len(vector.indices) == len(set(vector.indices))


def is_sorted(vector: SparseVector) -> bool:
    for i in range(1, len(vector.indices)):
        if vector.indices[i] < vector.indices[i - 1]:
            return False
    return True


def sort(vector: SparseVector) -> SparseVector:
    if is_sorted(vector):
        return vector

    sorted_indices = np.argsort(vector.indices)
    return SparseVector(
        indices=vector.indices[sorted_indices],
        values=vector.values[sorted_indices],
    )


def calculate_distance_sparse(query: SparseVector, vectors: List[SparseVector]) -> List[float]:
    scores = []

    for vector in vectors:
        score = sparse_dot_product(query, vector)
        scores.append(score)

    return scores


# Expects sorted indices
def sparse_dot_product(vector1: SparseVector, vector2: SparseVector) -> float:
    result = 0.0
    i, j = 0, 0

    while i < len(vector1.indices) and j < len(vector2.indices):
        if vector1.indices[i] == vector2.indices[j]:
            result += vector1.values[i] * vector2.values[j]
            i += 1
            j += 1
        elif vector1.indices[i] < vector2.indices[j]:
            i += 1
        else:
            j += 1

    return result
