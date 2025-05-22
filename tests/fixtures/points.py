import random
import uuid
from typing import Union

import numpy as np

from qdrant_client._pydantic_compat import construct
from qdrant_client.http import models
from qdrant_client.http.models import SparseVector
from qdrant_client.local.sparse import validate_sparse_vector
from tests.fixtures.payload import one_random_payload_please

text_vector_size = 20

_text_vectors = np.load("data/text.npy", mmap_mode="r")[..., :text_vector_size]
_text_vectors_unique = np.unique(_text_vectors, axis=0)
_text_vectors_clean = _text_vectors_unique[~np.isnan(_text_vectors_unique).any(axis=1)].tolist()


def sample_queries(n: int) -> list[np.array]:
    _query_vectors = np.load("data/queries.npy", allow_pickle=True).astype(np.float32)[
        ..., :text_vector_size
    ]
    _query_vectors_unique = np.unique(_query_vectors, axis=0)
    _query_vectors = _query_vectors_unique.tolist()
    sampled_vectors = np.random.choice(len(_query_vectors), size=n, replace=False)
    return [_query_vectors[i].copy() for i in sampled_vectors]


def random_vectors(vector_sizes: Union[dict[str, int], int], idx=None) -> models.VectorStruct:
    if isinstance(vector_sizes, int):
        if idx:
            return _text_vectors_clean[idx].copy()
        else:
            return np.random.random(vector_sizes).tolist()  # .round(3)
    elif isinstance(vector_sizes, dict):
        vectors = {}
        for vector_name, vector_size in vector_sizes.items():
            if idx:
                vectors[vector_name] = _text_vectors_clean[idx].copy()
            else:
                vectors[vector_name] = np.random.random(vector_size).tolist()  # .round(3)
        return vectors
    else:
        raise ValueError("vector_sizes must be int or dict")


def random_multivectors(vector_sizes: Union[dict[str, int], int]) -> models.VectorStruct:
    if isinstance(vector_sizes, int):
        vec_count = random.randint(1, 10)
        return sample_random_multivector(vector_sizes, vec_count)
    elif isinstance(vector_sizes, dict):
        vectors = {}
        for vector_name, vector_size in vector_sizes.items():
            vec_count = random.randint(1, 10)
            vectors[vector_name] = sample_random_multivector(vector_size, vec_count)
        return vectors
    else:
        raise ValueError("vector_sizes must be int or dict")


def generate_random_multivector(vec_size: int, vec_count: int) -> list[list[float]]:
    multivec = []
    for _ in range(vec_count):
        multivec.append(np.random.random(vec_size).tolist())  # .round(3).
    return multivec


def sample_random_multivector(vec_size: int, vec_count: int) -> list[list[float]]:
    doc_vectors = _text_vectors_clean.copy()
    sampled_vectors = np.random.choice(len(doc_vectors), size=vec_count, replace=False)
    return [np.array(doc_vectors[i]).astype(np.float32).tolist() for i in sampled_vectors]


# Generate random sparse vector with given size and density
# The density is the probability of non-zero value over the whole vector
def generate_random_sparse_vector(size: int, density: float) -> SparseVector:
    num_non_zero = int(size * density)
    indices: list[int] = random.sample(range(size), num_non_zero)
    values: list[float] = [random.random() for _ in range(num_non_zero)]

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

    doc_vectors = _text_vectors_clean.copy()
    sampled_vectors = np.random.choice(len(doc_vectors), size=num_points, replace=False)
    points = []

    for i in range(num_points):
        payload = None
        if with_payload:
            payload = one_random_payload_please(i)

        idx = i
        if random_ids:
            idx = str(uuid.uuid4())

        if sparse:
            vectors = random_sparse_vectors(vector_sizes, even=even_sparse)
        elif multivector:
            vectors = random_multivectors(vector_sizes)
            if isinstance(vectors, dict):
                for name, vec in vectors.items():
                    assert np.array(vec).dtype.kind in ("f", "i")  # float or int
                    assert not np.isnan(np.array(vec)).any()
            else:
                for name, vec in enumerate(vectors):
                    assert np.array(vec).dtype.kind in ("f", "i")  # float or int
                    assert not np.isnan(np.array(vec)).any()
        else:
            vectors = random_vectors(vector_sizes, sampled_vectors[i])

        if skip_vectors:
            if random.random() > 0.8:
                vector_to_skip = random.choice(list(vectors.keys()))
                vectors.pop(vector_to_skip)

        points.append(
            construct(
                models.PointStruct,
                id=idx,
                vector=vectors,
                payload=payload,
            )
        )

    return points
