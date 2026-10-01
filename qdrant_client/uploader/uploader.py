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


def iterate_upload_items(
    vectors: Iterable[Any],
    ids: Iterable[Any] | None,
    payload: Iterable[Any] | None,
) -> Iterable[tuple[Any, Any, Any]]:
    """Yield upload fields together and reject auxiliary iterables that end before vectors.

    Vectors determine the number of uploaded points. IDs and payloads may be infinite iterables,
    which is useful for generated values, so values remaining after vectors are intentionally ignored.
    """
    ids_iterator = iter(ids) if ids is not None else None
    payload_iterator = iter(payload) if payload is not None else None

    for vector in vectors:
        if ids_iterator is None:
            point_id = None
        else:
            try:
                point_id = next(ids_iterator)
            except StopIteration as exc:
                raise ValueError("ids iterable is shorter than vectors iterable") from exc

        if payload_iterator is None:
            point_payload = None
        else:
            try:
                point_payload = next(payload_iterator)
            except StopIteration as exc:
                raise ValueError("payload iterable is shorter than vectors iterable") from exc

        yield point_id, vector, point_payload


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

        vector_items = (vector for vector_batch in vector_batches for vector in vector_batch)
        upload_items = iterate_upload_items(vector_items, ids, payload)

        for upload_batch in iter_batch(upload_items, batch_size):
            yield (
                None if ids is None else [item[0] for item in upload_batch],
                [item[1] for item in upload_batch],
                None if payload is None else [item[2] for item in upload_batch],
            )

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
