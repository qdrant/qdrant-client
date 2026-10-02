from abc import ABC
from collections.abc import Sized
from itertools import chain, count, islice, zip_longest
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


_strict_missing: Any = object()


def _fail_fast_on_length_mismatch(
    ids: Iterable | None, payload: Iterable | None, vectors: Iterable
) -> None:
    """Raise immediately when sized `ids`/`payload` are shorter than sized
    `vectors`, before any batch is produced (avoids partial uploads).

    Only applies when all compared inputs are sized collections; generators
    and numpy inputs are validated lazily during iteration instead.
    """
    if isinstance(vectors, Sized):
        if isinstance(ids, Sized) and len(ids) < len(vectors):
            raise ValueError("ids iterable is shorter than vectors iterable")
        if isinstance(payload, Sized) and len(payload) < len(vectors):
            raise ValueError("payload iterable is shorter than vectors iterable")


def _zip_strict(
    ids: Iterable, vectors: Iterable, payload: Iterable
) -> Iterable[tuple[Any, Any, Any]]:
    """Item-level strict zip for upload streams.

    Raises `ValueError` when the `ids`/`payload` stream ends before
    `vectors`. `vectors` ending first keeps the previous behavior (extra
    ids/payload are ignored) so infinite id/payload generators stay
    supported. Lazy and O(1) memory.
    """
    for point_id, vector, payload_item in zip_longest(
        ids, vectors, payload, fillvalue=_strict_missing
    ):
        if vector is _strict_missing:
            break
        if point_id is _strict_missing:
            raise ValueError("ids iterable is shorter than vectors iterable")
        if payload_item is _strict_missing:
            raise ValueError("payload iterable is shorter than vectors iterable")
        yield point_id, vector, payload_item


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
        if ids is None:
            ids_stream: Iterable = (None for _ in count())
        else:
            ids_stream = iter(ids)

        if payload is None:
            payload_stream: Iterable = (None for _ in count())
        else:
            payload_stream = iter(payload)

        if isinstance(vectors, np.ndarray):
            vectors_stream: Iterable[Any] = chain.from_iterable(
                cls._vector_batches_from_numpy(vectors, batch_size)
            )
        elif isinstance(vectors, dict) and any(
            isinstance(value, np.ndarray) for value in vectors.values()
        ):
            vectors_stream = chain.from_iterable(
                cls._vector_batches_from_numpy_named_vectors(vectors, batch_size)
            )
        else:
            vectors_stream = iter(vectors)
            _fail_fast_on_length_mismatch(ids, payload, vectors)

        yield from iter_batch(_zip_strict(ids_stream, vectors_stream, payload_stream), batch_size)

    @staticmethod
    def _vector_batches_from_numpy(vectors: types.NumpyArray, batch_size: int) -> Iterable[float]:
        for i in range(0, vectors.shape[0], batch_size):
            yield vectors[i : i + batch_size].tolist()

    @staticmethod
    def _vector_batches_from_numpy_named_vectors(
        vectors: dict[str, types.NumpyArray], batch_size: int
    ) -> Iterable[dict[str, list[float]]]:
        if len(set([arr.shape[0] for arr in vectors.values()])) != 1:
            raise ValueError("Each named vector should have the same number of vectors")

        num_vectors = next(iter(vectors.values())).shape[0]
        # Convert dict[str, np.ndarray] to Generator(dict[str, list[float]])
        vector_batches = (
            {name: vectors[name][i].tolist() for name in vectors.keys()}
            for i in range(num_vectors)
        )
        yield from iter_batch(vector_batches, batch_size)
