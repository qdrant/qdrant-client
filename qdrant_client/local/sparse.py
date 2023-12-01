from qdrant_client.conversions import common_types as types
from qdrant_client.http.models import SparseVector
from typing import List, Optional
import random
import numpy as np


def empty_sparse_vector() -> SparseVector:
    return SparseVector(
        indices=[],
        values=[],
    )


def validate_sparse_vector(vector: SparseVector) -> None:
    assert len(vector.indices) == len(vector.values), "Indices and values must have the same length"
    assert len(vector.indices) == len(set(vector.indices)), "Indices must be unique"


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


def calculate_distance_sparse(query: SparseVector, vectors: List[SparseVector]) -> types.NumpyArray:
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


# Generate random sparse vector with given size and density
# The density is the probability of non-zero value over the whole vector
def generate_random_sparse_vector(size: int, density: float) -> SparseVector:
    num_non_zero = int(size * density)
    indices: List[int] = random.sample(range(size), num_non_zero)
    values: List[float] = [round(random.random(), 6) for _ in range(num_non_zero)]
    indices.sort()
    sparse_vector = SparseVector(indices=indices, values=values)
    validate_sparse_vector(sparse_vector)
    return sparse_vector


def generate_random_sparse_vector_list(num_vectors: int, vector_size: int, vector_density: float) -> List[SparseVector]:
    sparse_vector_list = []
    for _ in range(num_vectors):
        sparse_vector = generate_random_sparse_vector(vector_size, vector_density)
        sparse_vector_list.append(sparse_vector)
    return sparse_vector_list
