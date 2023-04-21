import itertools
import math
from abc import ABC
from itertools import count, islice
from typing import Generator, Iterable, Optional, Union

import numpy as np

from qdrant_client.conversions.common_types import Record
from qdrant_client.http.models import ExtendedPointId, VectorStruct
from qdrant_client.parallel_processor import Worker


def iter_batch(iterable: Union[Iterable, Generator], size: int) -> Iterable:
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
    def iterate_records_batches(cls, records: Iterable[Record], batch_size: int) -> Iterable:
        record_batches = iter_batch(records, batch_size)
        for record_batch in record_batches:
            ids_batch = [record.id for record in record_batch]
            vectors_batch = [record.vector for record in record_batch]
            payload_batch = [record.payload for record in record_batch]
            yield ids_batch, vectors_batch, payload_batch

    @classmethod
    def iterate_batches(
        cls,
        vectors: Union[np.ndarray, Iterable[VectorStruct]],
        payload: Optional[Iterable[dict]],
        ids: Optional[Iterable[ExtendedPointId]],
        batch_size: int,
    ) -> Iterable:
        if ids is None:
            ids = itertools.count()

        ids_batches = iter_batch(ids, batch_size)
        if payload is None:
            payload_batches: Union[Generator, Iterable] = (None for _ in count())
        else:
            payload_batches = iter_batch(payload, batch_size)

        if isinstance(vectors, np.ndarray):
            vector_batches = _get_vector_batches_from_numpy(vectors, batch_size)
        else:
            vector_batches = iter_batch(vectors, batch_size)

        yield from zip(ids_batches, vector_batches, payload_batches)


def _get_vector_batches_from_numpy(
    vectors: np.ndarray, batch_size: int
) -> Union[Generator, Iterable]:
    num_vectors = vectors.shape[0]
    num_batches = int(math.ceil(num_vectors / batch_size))
    vector_batches: Union[Generator, Iterable] = (
        vectors[i * batch_size : (i + 1) * batch_size].tolist() for i in range(num_batches)
    )
    return vector_batches
