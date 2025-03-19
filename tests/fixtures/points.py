import random
import uuid
from tabnanny import check
from typing import Union

import numpy as np

from qdrant_client._pydantic_compat import construct
from qdrant_client.http import models
from qdrant_client.http.models import SparseVector
from qdrant_client.local.sparse import validate_sparse_vector
from tests.fixtures.payload import one_random_payload_please

ROUND_PRECISION = 3

def find_mind_dist(vectors: np.ndarray):
    vectors_norm = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    cosine_sim_matrix = vectors_norm @ vectors_norm.T
    np.fill_diagonal(cosine_sim_matrix, -np.inf)
    max_cosine_similarity = np.max(cosine_sim_matrix)
    min_cosine_distance = 1 - max_cosine_similarity
    return min_cosine_distance

def check_distance(vectors: np.ndarray, threshold: float = 10**(-ROUND_PRECISION + 1)) -> bool:
    return find_mind_dist(vectors) > threshold

def random_vectors(
    vector_sizes: Union[dict[str, int], int],
) -> models.VectorStruct:
    if isinstance(vector_sizes, int):
        return np.random.random(vector_sizes).round(ROUND_PRECISION).tolist()
    elif isinstance(vector_sizes, dict):
        vectors = {}
        for vector_name, vector_size in vector_sizes.items():
            vectors[vector_name] = np.random.random(vector_size).round(ROUND_PRECISION).tolist()
        return vectors
    else:
        raise ValueError("vector_sizes must be int or dict")

def random_multivectors(vector_sizes: Union[dict[str, int], int]) -> models.VectorStruct:
    if isinstance(vector_sizes, int):
        vec_count = random.randint(1, 10)
        return generate_random_multivector(vector_sizes, vec_count)
    elif isinstance(vector_sizes, dict):
        vectors = {}
        for vector_name, vector_size in vector_sizes.items():
            vec_count = random.randint(1, 10)
            vectors[vector_name] = generate_random_multivector(vector_size, vec_count)
        return vectors
    else:
        raise ValueError("vector_sizes must be int or dict")


def generate_random_multivector(vec_size: int, vec_count: int) -> list[list[float]]:
    multivec = []
    for _ in range(vec_count):
        multivec.append(np.random.random(vec_size).round(ROUND_PRECISION).tolist())
    return multivec

def generate_random_sparse_vector(size: int, density: float) -> SparseVector:
    num_non_zero = int(size * density)
    indices: list[int] = random.sample(range(size), num_non_zero)
    values: list[float] = [round(random.random(), 6) for _ in range(num_non_zero)]
    sparse_vector = SparseVector(indices=indices, values=values)
    validate_sparse_vector(sparse_vector)
    return sparse_vector


def generate_random_sparse_vector_uneven(size: int, density: float) -> SparseVector:
    if random.random() > 0.5:
        size = int(size * 0.3)
    return generate_random_sparse_vector(size, density)


def generate_random_sparse_vector_list(
    num_vectors: int, vector_size: int, vector_density: float
) -> list[SparseVector]:
    sparse_vector_list = []
    for _ in range(num_vectors):
        sparse_vector = generate_random_sparse_vector(vector_size, vector_density)
        sparse_vector_list.append(sparse_vector)
    return sparse_vector_list


def random_sparse_vectors(
    vector_sizes: dict[str, int],
    even: bool = True,
) -> models.VectorStruct:
    vectors = {}
    for vector_name, vector_size in vector_sizes.items():
        # use sparse vectors with 20% density
        if even:
            vectors[vector_name] = generate_random_sparse_vector(vector_size, density=0.2)
        else:
            vectors[vector_name] = generate_random_sparse_vector_uneven(vector_size, density=0.2)
    return vectors


def generate_dense_vectors(num: int, size: int) -> list[list[float]]:
    vectors = np.random.random(size=(num, size)).round(ROUND_PRECISION).tolist()
    while not check_distance(vectors):
        vectors = np.random.random(size=(num, size)).round(ROUND_PRECISION).tolist()
    return vectors


def create_point(index: int, vector: any, with_payload: bool, random_ids: bool) -> models.PointStruct:
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
    if sparse:
        vec = random_sparse_vectors(vector_sizes, even=even_sparse)
    elif multivector:
        vec = random_multivectors(vector_sizes)
    else:
        raise

    if skip_vectors and vec and random.random() > 0.8:
        key_to_skip = random.choice(list(vec.keys()))
        vec.pop(key_to_skip)
    return vec


def generate_dense_points_single(
        num_points: int,
        vector_size: int,
        with_payload: bool,
        random_ids: bool
) -> list[models.PointStruct]:
    dense_vectors = generate_dense_vectors(num_points, vector_size)
    points = []
    for i, vec in enumerate(dense_vectors):
        points.append(create_point(i, vec, with_payload, random_ids))
    return points


def generate_dense_points_multi(
        num_points: int,
        vector_sizes: dict[str, int],
        with_payload: bool,
        random_ids: bool
) -> list[models.PointStruct]:
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
        multivector: bool = False,
) -> list[models.PointStruct]:
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
