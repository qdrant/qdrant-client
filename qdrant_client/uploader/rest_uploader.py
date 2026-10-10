from itertools import count
from time import sleep
from typing import Any, Iterable
from uuid import uuid4

import numpy as np

from qdrant_client import grpc as grpc
from qdrant_client.common.client_exceptions import ResourceExhaustedResponse
from qdrant_client.http import SyncApis
from qdrant_client import models as rest
from qdrant_client.uploader.uploader import BaseUploader
from qdrant_client.common.client_warnings import show_warning
from qdrant_client.conversions import common_types as types
from qdrant_client.conversions.conversion import GrpcToRest


def upload_batch(
    openapi_client: SyncApis,
    collection_name: str,
    batch: tuple | rest.Batch,  # type: ignore[name-defined]
    max_retries: int,
    shard_key_selector: rest.ShardKeySelector | None,  # type: ignore[name-defined]
    update_filter: rest.Filter | None,  # type: ignore[name-defined]
    update_mode: rest.UpdateMode | None = None,  # type: ignore[name-defined]
    wait: bool = False,
) -> bool:
    points = batch_to_points(batch)

    attempt = 0
    while attempt <= max_retries:
        try:
            openapi_client.points_api.upsert_points(
                collection_name=collection_name,
                point_insert_operations=rest.PointsList(  # type: ignore[attr-defined]
                    points=points,
                    shard_key=shard_key_selector,
                    update_filter=update_filter,
                    update_mode=update_mode,
                ),
                wait=wait,
            )
            break
        except ResourceExhaustedResponse as ex:
            show_warning(
                message=f"Batch upload failed due to rate limit. Waiting for {ex.retry_after_s} seconds before retrying...",
                category=UserWarning,
                stacklevel=7,
            )
            sleep(ex.retry_after_s)

        except Exception as e:
            if attempt == max_retries:
                raise e

            show_warning(
                message=f"Batch upload failed {attempt + 1} times. Retrying...",
                category=UserWarning,
                stacklevel=7,
            )

            attempt += 1
    return True


def _to_list(vector: Any) -> Any:
    return vector.tolist() if isinstance(vector, np.ndarray) else vector


def batch_to_points(
    batch: tuple | rest.Batch,  # type: ignore[name-defined]
) -> list[rest.PointStruct]:  # type: ignore[name-defined]
    ids_batch, vectors_batch, payload_batch = batch

    # RestBatchUploader passes parts of numpy arrays on as they are, the workers make lists here
    if isinstance(vectors_batch, np.ndarray):
        vectors_batch = vectors_batch.tolist()
    elif isinstance(vectors_batch, dict):
        # a part of an array per name, split into a dict per point. A part of an array of arrays,
        # e.g. of multivectors of different lengths, makes a list of arrays
        columns = {name: part.tolist() for name, part in vectors_batch.items()}
        vectors_batch = [
            {name: _to_list(vector) for name, vector in zip(columns, row)}
            for row in zip(*columns.values())
        ]

    ids_batch = (str(uuid4()) for _ in count()) if ids_batch is None else ids_batch
    payload_batch = (None for _ in count()) if payload_batch is None else payload_batch

    return [
        rest.PointStruct(  # type: ignore[attr-defined]
            id=idx,
            vector=_to_list(vector) or {},
            payload=payload,
        )
        for idx, vector, payload in zip(ids_batch, vectors_batch, payload_batch)
    ]


class RestBatchUploader(BaseUploader):
    def __init__(
        self,
        uri: str,
        collection_name: str,
        max_retries: int,
        wait: bool = False,
        shard_key_selector: types.ShardKeySelector | None = None,
        update_filter: types.Filter | None = None,
        update_mode: types.UpdateMode | None = None,
        **kwargs: Any,
    ):
        self.collection_name = collection_name
        self.openapi_client: SyncApis = SyncApis(host=uri, **kwargs)
        self.max_retries = max_retries
        self._wait = wait
        self._shard_key_selector = shard_key_selector
        self._update_filter = (
            GrpcToRest.convert_filter(model=update_filter)
            if isinstance(update_filter, grpc.Filter)
            else update_filter
        )
        self._update_mode = update_mode

    @staticmethod
    def _from_numpy(vectors: types.NumpyArray) -> Any:
        """Keeps a part of a numpy array as it is, `batch_to_points` makes lists of it

        With `parallel > 1`, batches wait in a queue of the main process until a worker takes
        them, and are pickled on the way. A part of an array waits as a view of the caller's
        array, which takes almost no memory, and pickles as raw bytes. Lists would be built in
        the main process and take 32 bytes per python float, 8 times a float32: about 3 MB for 64
        vectors of 1536 dimensions. They also pickle slower. With `parallel=1`, the conversion
        only moves into `batch_to_points`, with the same result.
        """
        return vectors

    @classmethod
    def _vector_batches_from_numpy_named_vectors(
        cls, vectors: dict[str, types.NumpyArray], batch_size: int
    ) -> Iterable[dict[str, types.NumpyArray]]:
        """Batches of named vectors as a part of each array, for the reasons of `_from_numpy`"""
        if len(set([arr.shape[0] for arr in vectors.values()])) != 1:
            raise ValueError("Each named vector should have the same number of vectors")

        num_vectors = next(iter(vectors.values())).shape[0]
        for start in range(0, num_vectors, batch_size):
            yield {name: value[start : start + batch_size] for name, value in vectors.items()}

    @classmethod
    def start(
        cls,
        collection_name: str | None = None,
        uri: str = "http://localhost:6333",
        max_retries: int = 3,
        **kwargs: Any,
    ) -> "RestBatchUploader":
        if not collection_name:
            raise RuntimeError("Collection name could not be empty")
        return cls(uri=uri, collection_name=collection_name, max_retries=max_retries, **kwargs)

    def process(self, items: Iterable[Any]) -> Iterable[bool]:
        for batch in items:
            yield upload_batch(
                self.openapi_client,
                self.collection_name,
                batch,
                shard_key_selector=self._shard_key_selector,
                max_retries=self.max_retries,
                update_filter=self._update_filter,
                update_mode=self._update_mode,
                wait=self._wait,
            )
