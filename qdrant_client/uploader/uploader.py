import itertools
import math
from abc import ABC
from itertools import islice, count
from typing import Optional, Iterable, Any, Callable, Union, List

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
                        vectors: Union[np.ndarray, Iterable[List[float]]],
                        payload: Optional[Iterable[dict]],
                        ids: Optional[Iterable[ExtendedPointId]],
                        batch_size: int,
                        ) -> Iterable:
        if ids is None:
            ids = itertools.count()

        ids_batches = iter_batch(ids, batch_size)
        if payload is None:
            payload_batches = (None for _ in count())
        else:
            payload_batches = iter_batch(payload, batch_size)

        if isinstance(vectors, np.ndarray):
            num_vectors = vectors.shape[0]
            num_batches = int(math.ceil(num_vectors / batch_size))
            vector_batches = (vectors[i * batch_size:(i + 1) * batch_size].tolist() for i in range(num_batches))
        else:
            vector_batches = iter_batch(vectors, batch_size)

        yield from zip(ids_batches, vector_batches, payload_batches)
