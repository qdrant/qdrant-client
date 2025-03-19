import random
import uuid
from typing import Union

import numpy as np

from qdrant_client._pydantic_compat import construct
from qdrant_client.http import models
from qdrant_client.http.models import SparseVector
from qdrant_client.local.sparse import validate_sparse_vector
from tests.fixtures.payload import one_random_payload_please

# Constants
ROUND_PRECISION = 3


# =============================================================================
# Utility Functions for Dense Vectors
# =============================================================================

def find_mind_dist(vectors: np.ndarray) -> float:
    """
    Calculate the minimum cosine distance between vectors.
    """
    if len(vectors) > 1:
        vectors_norm = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
        cosine_sim_matrix = vectors_norm @ vectors_norm.T
        np.fill_diagonal(cosine_sim_matrix, -np.inf)
        max_cosine_similarity = np.max(cosine_sim_matrix)
        min_cosine_distance = 1 - max_cosine_similarity
        return min_cosine_distance
    else:
        return 1.0


def check_distance(vectors: np.ndarray, threshold: float = 10 ** (-ROUND_PRECISION + 1)) -> bool:
    """
    Check if the minimum cosine distance of vectors exceeds a threshold.
    """
    return find_mind_dist(vectors) > threshold


def generate_dense_vectors(num: int, size: int, tries=10) -> list[list[float]]:
    """
    Generate a list of dense vectors with a minimum distance check.
    """
    vectors = np.random.random(size=(num, size)).round(ROUND_PRECISION)

    while not check_distance(vectors):
        vectors = np.random.random(size=(num, size)).round(ROUND_PRECISION)
        tries-=1
        if tries < 0:
            raise RuntimeError(f"Can not find a dense vector in {tries} runs")
    return vectors.tolist()


def random_vectors(vector_sizes: Union[dict[str, int], int]) -> models.VectorStruct:
    """
    Generate random dense vectors.

    If an integer is provided, a single vector is returned.
    If a dict is provided, a dictionary of vectors is returned.
    """
    if isinstance(vector_sizes, int):
        return np.random.random(vector_sizes).round(ROUND_PRECISION).tolist()
    elif isinstance(vector_sizes, dict):
        return {name: np.random.random(size).round(ROUND_PRECISION).tolist() for name, size in vector_sizes.items()}
    else:
        raise ValueError("vector_sizes must be int or dict")


# =============================================================================
# Functions for Multivector Generation
# =============================================================================

def generate_random_multivector(vec_size: int, vec_count: int) -> list[list[float]]:
    """
    Generate a list of multivectors (each a list of floats).
    """
    return [np.random.random(vec_size).round(ROUND_PRECISION).tolist() for _ in range(vec_count)]


def random_multivectors(vector_sizes: Union[dict[str, int], int]) -> models.VectorStruct:
    """
    Generate random multivectors.

    For int input, returns a multivector with a random count (between 1 and 10).
    For dict input, returns a dictionary of multivectors.
    """
    if isinstance(vector_sizes, int):
        vec_count = random.randint(1, 10)
        return generate_random_multivector(vector_sizes, vec_count)
    elif isinstance(vector_sizes, dict):
        return {
            name: generate_random_multivector(size, random.randint(1, 10))
            for name, size in vector_sizes.items()
        }
    else:
        raise ValueError("vector_sizes must be int or dict")


# =============================================================================
# Functions for Sparse Vector Generation
# =============================================================================

def generate_random_sparse_vector(size: int, density: float) -> SparseVector:
    """
    Generate a random sparse vector with a given density.
    """
    num_non_zero = int(size * density)
    indices = random.sample(range(size), num_non_zero)
    values = [round(random.random(), 6) for _ in range(num_non_zero)]
    sparse_vector = SparseVector(indices=indices, values=values)
    validate_sparse_vector(sparse_vector)
    return sparse_vector


def generate_random_sparse_vector_uneven(size: int, density: float) -> SparseVector:
    """
    Generate a random sparse vector with uneven size modification.
    """
    if random.random() > 0.5:
        size = int(size * 0.3)
    return generate_random_sparse_vector(size, density)


def generate_random_sparse_vector_list(num_vectors: int, vector_size: int, vector_density: float) -> list[SparseVector]:
    """
    Generate a list of random sparse vectors.
    """
    return [generate_random_sparse_vector(vector_size, vector_density) for _ in range(num_vectors)]


def random_sparse_vectors(vector_sizes: dict[str, int], even: bool = True) -> models.VectorStruct:
    """
    Generate random sparse vectors for each key in vector_sizes.

    Uses even distribution if `even` is True; otherwise uses uneven generation.
    """
    vectors = {}
    for name, size in vector_sizes.items():
        if even:
            vectors[name] = generate_random_sparse_vector(size, density=0.2)
        else:
            vectors[name] = generate_random_sparse_vector_uneven(size, density=0.2)
    return vectors


# =============================================================================
# Point Creation Functions
# =============================================================================

def create_point(index: int, vector: any, with_payload: bool, random_ids: bool) -> models.PointStruct:
    """
    Create a point with the given vector and payload.
    """
    point_id = str(uuid.uuid4()) if random_ids else index
    payload = one_random_payload_please(index) if with_payload else None
    return construct(models.PointStruct, id=point_id, vector=vector, payload=payload)


def get_vector_for_point(
        vector_sizes: Union[int, dict[str, int]],
        sparse: bool,
        even_sparse: bool,
        multivector: bool,
        skip_vectors: bool
) -> any:
    """
    Retrieve a vector for a point based on provided options.

    This function supports only sparse or multivector options.
    """
    if sparse:
        vec = random_sparse_vectors(vector_sizes, even=even_sparse)
    elif multivector:
        vec = random_multivectors(vector_sizes)
    else:
        raise NotImplementedError("Only sparse or multivector options are supported in get_vector_for_point")

    if skip_vectors and vec and random.random() > 0.8:
        # When vec is a dict, remove one random key.
        key_to_skip = random.choice(list(vec.keys()))
        vec.pop(key_to_skip)
    return vec


def generate_dense_points_single(
        num_points: int,
        vector_size: int,
        with_payload: bool,
        random_ids: bool
) -> list[models.PointStruct]:
    """
    Generate points using dense single vectors.
    """
    dense_vectors = generate_dense_vectors(num_points, vector_size)
    return [create_point(i, vec, with_payload, random_ids) for i, vec in enumerate(dense_vectors)]


def generate_dense_points_multi(
        num_points: int,
        vector_sizes: dict[str, int],
        with_payload: bool,
        random_ids: bool
) -> list[models.PointStruct]:
    """
    Generate points using dense multivectors (a dictionary of vectors).
    """
    dense_vectors_dict = {
        name: generate_dense_vectors(num_points, size)
        for name, size in vector_sizes.items()
    }
    points = []
    for i in range(num_points):
        combined_vector = {name: dense_vectors_dict[name][i] for name in vector_sizes}
        points.append(create_point(i, combined_vector, with_payload, random_ids))
    return points


def generate_sparse_or_multivector_points(
        num_points: int,
        vector_sizes: Union[int, dict[str, int]],
        with_payload: bool,
        random_ids: bool,
        skip_vectors: bool,
        sparse: bool,
        even_sparse: bool,
        multivector: bool
) -> list[models.PointStruct]:
    """
    Generate points using either sparse or multivector formats.
    """
    points = []
    for i in range(num_points):
        vec = get_vector_for_point(vector_sizes, sparse, even_sparse, multivector, skip_vectors)
        points.append(create_point(i, vec, with_payload, random_ids))
    return points


def generate_points(
        num_points: int,
        vector_sizes: Union[dict[str, int], int],
        with_payload: bool = False,
        random_ids: bool = False,
        skip_vectors: bool = False,
        sparse: bool = False,
        even_sparse: bool = True,
        multivector: bool = False
) -> list[models.PointStruct]:
    """
    Generate a list of points with various vector options.

    For dense vectors (neither sparse nor multivector):
      - If vector_sizes is an int, generates single dense vectors.
      - If vector_sizes is a dict, generates dense multivectors.

    For sparse or multivector vectors:
      - Uses get_vector_for_point to determine vector type.
    """
    if skip_vectors and isinstance(vector_sizes, int):
        raise ValueError("skip_vectors is not supported for single vector")

    if not sparse and not multivector:
        if isinstance(vector_sizes, int):
            return generate_dense_points_single(num_points, vector_sizes, with_payload, random_ids)
        elif isinstance(vector_sizes, dict):
            return generate_dense_points_multi(num_points, vector_sizes, with_payload, random_ids)
    else:
        return generate_sparse_or_multivector_points(
            num_points,
            vector_sizes,
            with_payload,
            random_ids,
            skip_vectors,
            sparse,
            even_sparse,
            multivector
        )
