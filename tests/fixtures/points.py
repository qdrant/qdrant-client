import random
import uuid
from typing import Union

import numpy as np

from qdrant_client._pydantic_compat import construct
from qdrant_client.http import models
from qdrant_client.http.models import SparseVector
from qdrant_client.local.sparse import validate_sparse_vector
from tests.fixtures.payload import one_random_payload_please


import numpy as np
from numpy.linalg import norm
import warnings

def too_close(vectors: np.ndarray, distance: str, thold : float= 10e-5) -> bool:
    """
    Efficiently checks if any two distances between the last vector and the others
    differ by less than or equal to `thold`.

    Parameters:
        vectors (np.ndarray): Array of shape (N, D)
        distance (str): One of {'text', 'image', 'code'}
        thold (float): Threshold for minimal distance difference

    Returns:
        bool: True if any two distances are within threshold, else False
    """
    warnings.warn(
        "too_close() was called. Make sure this is intended, as it may affect performance.",
        category=UserWarning,
        stacklevel=2
    )

    if len(vectors) < 3:
        return False  # Need at least 2 distances to compare their difference

    last_vec = vectors[-1]
    other_vecs = vectors[:-1]

    if distance == 'image':
        # Negative dot product (pseudo-distance)
        distances = -np.dot(other_vecs, last_vec)

    elif distance == 'code':
        # Euclidean distance
        diff = other_vecs - last_vec
        distances = np.linalg.norm(diff, axis=1)
    else:
        # Cosine distance: 1 - cosine similarity
        normed_last = last_vec / norm(last_vec)
        normed_others = other_vecs / norm(other_vecs, axis=1, keepdims=True)
        distances = 1 - np.dot(normed_others, normed_last)

    diffs = np.abs(distances[:, None] - distances[None, :])
    np.fill_diagonal(diffs, np.inf)

    # if np.min(diffs) < thold:
    #     raise ValueError(np.min(diffs))
    return np.min(diffs) < thold


def random_vectors(
    vector_sizes: Union[dict[str, int], int],
    num_vectors = 1
) -> list[models.VectorStruct]:
    if isinstance(vector_sizes, int):
        return np.random.random((num_vectors, vector_sizes)).round(3).tolist()
    elif isinstance(vector_sizes, dict):
        vectors = {}
        for vector_name, vector_size in vector_sizes.items():
            generated_vecs = np.random.random((num_vectors, vector_size)).round(3)
            while too_close(generated_vecs, vector_name, 10e-8):
                generated_vecs = np.random.random((num_vectors, vector_size)).round(3)
            vectors[vector_name] = generated_vecs.tolist()

        output = []
        for i in range(num_vectors):
            rearranged_vect_list = {vector_name : vectors[vector_name][i] for vector_name in vector_sizes.keys()}
            output.append(rearranged_vect_list)
        return output
    else:
        raise ValueError("vector_sizes must be int or dict")


def random_multivectors(vector_sizes: Union[dict[str, int], int],  num_vectors = 1) -> list[models.VectorStruct]:
    output = []
    for i in range(num_vectors):
        if isinstance(vector_sizes, int):
            vec_count = random.randint(1, 10)
            output.append(generate_random_multivector(vector_sizes, vec_count))
        elif isinstance(vector_sizes, dict):
            vectors = {}
            for vector_name, vector_size in vector_sizes.items():
                vec_count = random.randint(1, 10)
                vectors[vector_name] = generate_random_multivector(vector_size, vec_count)
            output.append(vectors)
        else:
            raise ValueError("vector_sizes must be int or dict")
    return output

def generate_random_multivector(vec_size: int, vec_count: int) -> list[list[float]]:
    return np.round(np.random.random((vec_count, vec_size)), 3).tolist()

# Generate random sparse vector with given size and density
# The density is the probability of non-zero value over the whole vector
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
    num_vectors = 1
) -> list[models.VectorStruct]:
    output = []
    for i in range(num_vectors):
        vectors = {}
        for vector_name, vector_size in vector_sizes.items():
            # use sparse vectors with 20% density
            if even:
                vectors[vector_name] = generate_random_sparse_vector(vector_size, density=0.2)
            else:
                vectors[vector_name] = generate_random_sparse_vector_uneven(vector_size, density=0.2)
        output.append(vectors)
    return output


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

    if sparse:
        generated_vectors = random_sparse_vectors(vector_sizes, num_vectors=num_points, even=even_sparse)
    elif multivector:
        generated_vectors = random_multivectors(vector_sizes, num_vectors=num_points)
    else:
        generated_vectors = random_vectors(vector_sizes, num_vectors=num_points)

    points = []
    for i, vectors in enumerate(generated_vectors):
        payload = None
        if with_payload:
            payload = one_random_payload_please(i)
        idx = i
        if random_ids:
            idx = str(uuid.uuid4())

        points.append(
            construct(
                models.PointStruct,
                id=idx,
                vector=vectors,
                payload=payload,
            )
        )

    return points
