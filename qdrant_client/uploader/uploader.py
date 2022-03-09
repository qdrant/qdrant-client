import math
from abc import ABC
from itertools import islice, count
from typing import Optional, Iterable, Any, Callable

import numpy as np

from qdrant_client.http.models import ExtendedPointId
from qdrant_client.parallel_processor import Worker


def iter_batch(iterable, size) -> Iterable:
    """
    >>> list(iter_batch([1,2,3,4,5], 3))
    [[1, 2, 3], [4, 5]]
    """
    source_iter = iter(iterable)
    while source_iter:
        b = list(islice(source_iter, size))
        if len(b) == 0:
            break
        yield b


class BaseUploader(Worker, ABC):

    @classmethod
    def iterate_batches(cls,
                        vectors: np.ndarray,
                        payload: Optional[Iterable[dict]],
                        ids: Optional[Iterable[ExtendedPointId]],
                        batch_size: int,
                        ) -> Iterable:
        num_vectors, _dim = vectors.shape
        if ids is None:
            ids = range(num_vectors)

        ids_batches = iter_batch(ids, batch_size)
        if payload is None:
            payload_batches = (None for _ in count())
        else:
            payload_batches = iter_batch(payload, batch_size)

        num_batches = int(math.ceil(num_vectors / batch_size))
        vector_batches = (vectors[i * batch_size:(i + 1) * batch_size].tolist() for i in range(num_batches))

        yield from zip(ids_batches, vector_batches, payload_batches)
