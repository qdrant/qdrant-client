import uuid
from typing import Dict, List, Union

import numpy as np

from qdrant_client.http import models
from tests.fixtures.payload import one_random_payload_please, random_payload


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


def generate_records(
    num_records: int,
    vector_sizes: Union[Dict[str, int], int],
    with_payload: bool = False,
    random_ids: bool = False,
) -> List[models.Record]:
    records = []
    for i in range(num_records):
        payload = None
        if with_payload:
            payload = one_random_payload_please(i)

        idx = i
        if random_ids:
            idx = str(uuid.uuid4())

        records.append(
            models.Record.construct(
                id=idx,
                vector=random_vectors(vector_sizes),
                payload=payload,
            )
        )

    return records
