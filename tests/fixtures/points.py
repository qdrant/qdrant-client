import random
import uuid
from typing import Dict, List, Union

import numpy as np

from qdrant_client._pydantic_compat import construct
from qdrant_client.http import models
from qdrant_client.http.models import SparseVector
from qdrant_client.local.sparse import validate_sparse_vector
from tests.fixtures.payload import one_random_payload_please


def random_vectors(
    vector_sizes: Union[Dict[str, int], int],
) -> models.VectorStruct:
    if isinstance(vector_sizes, int):
        return np.random.random(vector_sizes).round(3).tolist()
    elif isinstance(vector_sizes, dict):
        vectors = {}
        for vector_name, vector_size in vector_sizes.items():
            vectors[vector_name] = np.random.random(vector_size).round(3).tolist()
        return vectors
    else:
        raise ValueError("vector_sizes must be int or dict")


# Generate random sparse vector with given size and density
# The density is the probability of non-zero value over the whole vector
def generate_random_sparse_vector(size: int, density: float) -> SparseVector:
    num_non_zero = int(size * density)
    indices: List[int] = random.sample(range(size), num_non_zero)
    values: List[float] = [round(random.random(), 6) for _ in range(num_non_zero)]
    sparse_vector = SparseVector(indices=indices, values=values)
    validate_sparse_vector(sparse_vector)
    return sparse_vector


def generate_random_sparse_vector_list(
    num_vectors: int, vector_size: int, vector_density: float
) -> List[SparseVector]:
    sparse_vector_list = []
    for _ in range(num_vectors):
        sparse_vector = generate_random_sparse_vector(vector_size, vector_density)
        sparse_vector_list.append(sparse_vector)
    return sparse_vector_list


def random_sparse_vectors(
    vector_sizes: Union[Dict[str, int], int],
) -> models.VectorStruct:
    vectors = {}
    for vector_name, vector_size in vector_sizes.items():
        # use sparse vectors with 20% density
        vectors[vector_name] = generate_random_sparse_vector(vector_size, density=0.2)
    return vectors


def generate_records(
    num_records: int,
    vector_sizes: Union[Dict[str, int], int],
    with_payload: bool = False,
    random_ids: bool = False,
    skip_vectors: bool = False,
    sparse: bool = False,
) -> List[models.Record]:
    if skip_vectors and isinstance(vector_sizes, int):
        raise ValueError("skip_vectors is not supported for single vector")

    records = []
    for i in range(num_records):
        payload = None
        if with_payload:
            payload = one_random_payload_please(i)

        idx = i
        if random_ids:
            idx = str(uuid.uuid4())

        if sparse:
            vectors = random_sparse_vectors(vector_sizes)
        else:
            vectors = random_vectors(vector_sizes)

        if skip_vectors:
            if random.random() > 0.8:
                vector_to_skip = random.choice(list(vectors.keys()))
                vectors.pop(vector_to_skip)

        records.append(
            construct(
                models.Record,
                id=idx,
                vector=vectors,
                payload=payload,
            )
        )

    return records
