import itertools
import math
from abc import ABC
from itertools import count, islice
from typing import Any, Dict, Generator, Iterable, List, Optional, Tuple, Union

import numpy as np

from qdrant_client.conversions import common_types as types
from qdrant_client.conversions.common_types import Record
from qdrant_client.http.models import ExtendedPointId, ShardKey
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
    def split_batch_by_shard_key(
        cls,
        ids_batch: Iterable,
        vectors_batch: Iterable,
        payload_batch: Iterable,
        shard_keys_batch: List[Optional[ShardKey]],
    ) -> Iterable:
        bucket_map = {shard_key: index for index, shard_key in enumerate(set(shard_keys_batch))}
        if len(bucket_map) == 1:
            yield (ids_batch, vectors_batch, payload_batch), list(bucket_map.keys())[0]
        else:
            buckets: list[list[tuple]] = [[] for _ in range(len(bucket_map))]

            for id_, vector, payload, shard_key in zip(
                ids_batch, vectors_batch, payload_batch, shard_keys_batch
            ):
                buckets[bucket_map[shard_key]].append((id_, vector, payload))

            for shard_key, index in bucket_map.items():
                ids_batch_for_shard, vectors_batch_for_shard, payload_batch_for_shard = zip(
                    *buckets[index]
                )
                yield (
                    ids_batch_for_shard,
                    vectors_batch_for_shard,
                    payload_batch_for_shard,
                ), shard_key

    @classmethod
    def iterate_records_batches(cls, records: Iterable[Record], batch_size: int) -> Iterable:
        record_batches = iter_batch(records, batch_size)
        for record_batch in record_batches:
            ids_batch = [record.id for record in record_batch]
            vectors_batch = [record.vector for record in record_batch]
            payload_batch = [record.payload for record in record_batch]
            shard_keys_batch = [record.shard_key for record in record_batch]
            yield from cls.split_batch_by_shard_key(
                ids_batch, vectors_batch, payload_batch, shard_keys_batch
            )

    @classmethod
    def iterate_batches(
        cls,
        vectors: Union[
            Dict[str, types.NumpyArray], types.NumpyArray, Iterable[types.VectorStruct]
        ],
        payload: Optional[Iterable[dict]],
        ids: Optional[Iterable[ExtendedPointId]],
        shard_keys: Optional[Iterable[ShardKey]],
        batch_size: int,
    ) -> Iterable:
        if ids is None:
            ids = itertools.count()

        ids_batches = iter_batch(ids, batch_size)
        if payload is None:
            payload_batches: Union[Generator, Iterable] = (
                (None for _ in range(batch_size)) for _ in count()
            )
        else:
            payload_batches = iter_batch(payload, batch_size)

        if shard_keys is None:
            shard_keys_batches: Union[Generator, Iterable] = (
                (None for _ in range(batch_size)) for _ in count()
            )
        else:
            shard_keys_batches = iter_batch(shard_keys, batch_size)

        if isinstance(vectors, np.ndarray):
            vector_batches: Iterable[Any] = cls._vector_batches_from_numpy(vectors, batch_size)
        elif isinstance(vectors, dict) and any(
            isinstance(value, np.ndarray) for value in vectors.values()
        ):
            vector_batches = cls._vector_batches_from_numpy_named_vectors(vectors, batch_size)
        else:
            vector_batches = iter_batch(vectors, batch_size)

        for id_batch, vector_batch, payload_batch, shard_key_batch in zip(
            ids_batches, vector_batches, payload_batches, shard_keys_batches
        ):
            yield from cls.split_batch_by_shard_key(
                id_batch, vector_batch, payload_batch, list(shard_key_batch)
            )

    @staticmethod
    def _vector_batches_from_numpy(vectors: types.NumpyArray, batch_size: int) -> Iterable[float]:
        for i in range(0, vectors.shape[0], batch_size):
            yield vectors[i : i + batch_size].tolist()

    @staticmethod
    def _vector_batches_from_numpy_named_vectors(
        vectors: Dict[str, types.NumpyArray], batch_size: int
    ) -> Iterable[Dict[str, List[float]]]:
        assert (
            len(set([arr.shape[0] for arr in vectors.values()])) == 1
        ), "Each named vector should have the same number of vectors"

        num_vectors = next(iter(vectors.values())).shape[0]
        # Convert Dict[str, np.ndarray] to Generator(Dict[str, List[float]])
        vector_batches = (
            {name: vectors[name][i].tolist() for name in vectors.keys()}
            for i in range(num_vectors)
        )
        yield from iter_batch(vector_batches, batch_size)
