from abc import ABC
from itertools import islice
from typing import Any, Generator, Iterable

import numpy as np

from qdrant_client.conversions import common_types as types
from qdrant_client.conversions.common_types import Record
from qdrant_client.http.models import ExtendedPointId
from qdrant_client.parallel_processor import Worker


def iter_batch(iterable: Iterable | Generator, size: int) -> Iterable:
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


def take_exactly(items: Iterable, size: int, name: str) -> list:
    """Takes the next `size` items of `ids` or `payload`, one per vector, raises if fewer are left

    Items after them are not consumed, so `ids` and `payload` may be longer than the vectors, e.g.
    infinite generators.
    """
    taken = list(islice(items, size))
    if len(taken) < size:
        raise ValueError(f"{name} iterable is shorter than vectors iterable")
    return taken


class BaseUploader(Worker, ABC):
    @classmethod
    def iterate_records_batches(
        cls,
        records: Iterable[Record | types.PointStruct],
        batch_size: int,
    ) -> Iterable:
        record_batches = iter_batch(records, batch_size)
        for record_batch in record_batches:
            ids_batch, vectors_batch, payload_batch = [], [], []

            for record in record_batch:
                ids_batch.append(record.id)
                vectors_batch.append(record.vector)
                payload_batch.append(record.payload)

            yield ids_batch, vectors_batch, payload_batch

    @classmethod
    def iterate_batches(
        cls,
        vectors: dict[str, types.NumpyArray] | types.NumpyArray | Iterable[types.VectorStruct],
        payload: Iterable[dict] | None,
        ids: Iterable[ExtendedPointId] | None,
        batch_size: int,
    ) -> Iterable:
        if isinstance(vectors, np.ndarray):
            vector_batches: Iterable[Any] = cls._vector_batches_from_numpy(vectors, batch_size)
        elif isinstance(vectors, dict) and any(
            isinstance(value, np.ndarray) for value in vectors.values()
        ):
            vector_batches = cls._vector_batches_from_numpy_named_vectors(vectors, batch_size)
        else:
            vector_batches = iter_batch(vectors, batch_size)

        ids_iter = None if ids is None else iter(ids)
        payload_iter = None if payload is None else iter(payload)
        for vectors_batch in vector_batches:
            # RestBatchUploader batches named numpy vectors as a dict of parts of the arrays
            if isinstance(vectors_batch, dict):
                size = len(next(iter(vectors_batch.values())))
            else:
                size = len(vectors_batch)
            yield (
                None if ids_iter is None else take_exactly(ids_iter, size, "ids"),
                vectors_batch,
                None if payload_iter is None else take_exactly(payload_iter, size, "payload"),
            )

    @staticmethod
    def _from_numpy(vectors: types.NumpyArray) -> Any:
        """Converts a part of a numpy array before it is passed to `process`."""
        return vectors.tolist()

    @classmethod
    def _vector_batches_from_numpy(
        cls, vectors: types.NumpyArray, batch_size: int
    ) -> Iterable[Any]:
        for i in range(0, vectors.shape[0], batch_size):
            yield cls._from_numpy(vectors[i : i + batch_size])

    @classmethod
    def _vector_batches_from_numpy_named_vectors(
        cls, vectors: dict[str, types.NumpyArray], batch_size: int
    ) -> Iterable[dict[str, Any]]:
        if len(set([arr.shape[0] for arr in vectors.values()])) != 1:
            raise ValueError("Each named vector should have the same number of vectors")

        num_vectors = next(iter(vectors.values())).shape[0]
        # Convert dict[str, np.ndarray] to Generator(dict[str, vector]), a vector per name
        vector_batches = (
            {name: cls._from_numpy(vectors[name][i]) for name in vectors.keys()}
            for i in range(num_vectors)
        )
        yield from iter_batch(vector_batches, batch_size)
