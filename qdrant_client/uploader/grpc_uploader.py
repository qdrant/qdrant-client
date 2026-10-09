from functools import lru_cache
from itertools import count
from time import sleep
from typing import Any, Generator, Iterable
from uuid import uuid4

import numpy as np
from google.protobuf.internal import api_implementation

from qdrant_client import grpc as grpc
from qdrant_client import models as rest
from qdrant_client.common.client_exceptions import ResourceExhaustedResponse
from qdrant_client.connection import get_channel
from qdrant_client.conversions.conversion import RestToGrpc, payload_to_grpc
from qdrant_client.uploader.uploader import BaseUploader
from qdrant_client.common.client_warnings import show_warning
from qdrant_client.conversions import common_types as types


# only upb parses the bytes fast and to the floats it makes of python floats: pure python protobuf
# is slower at it and loses NaN payloads, the cpp one of protobuf 3 rounds differently near FLT_MAX
_UPB = api_implementation.Type() == "upb"


@lru_cache(maxsize=64)
def _dense_vector_prefix(dim: int, multi: bool) -> bytes:
    """What protobuf writes before the floats of a `DenseVector` of `dim` floats

    With `multi`, of a `DenseVector` in a `MultiDenseVector`. Cut from a serialized vector of
    zeros, which ends with its floats.
    """
    message: Any = grpc.DenseVector(data=[0.0] * dim)
    if multi:
        message = grpc.MultiDenseVector(vectors=[message])
    serialized = message.SerializeToString()
    return serialized[: len(serialized) - 4 * dim]


def _is_numpy_vector(vector: Any) -> bool:
    # other arrays: empty ones, of long doubles or not of numbers, and subclasses like matrices
    # (their rows are 2-d) or masked arrays, take the path of lists as before
    return (
        _UPB
        and type(vector) in (np.ndarray, np.memmap)
        and vector.ndim in (1, 2)
        and vector.size > 0
        and vector.dtype.kind in "fiub"
        and vector.dtype.itemsize <= 8
    )


def _set_numpy_vector(target: grpc.Vector, vector: np.ndarray) -> None:
    # through doubles to floats, as tolist() and protobuf convert numbers, e.g. signaling NaNs
    # become quiet and values out of the float range inf, without warnings
    with np.errstate(all="ignore"):
        vector = vector.astype(np.float64, copy=False).astype("<f4")
    if vector.ndim == 1:
        target.dense.MergeFromString(_dense_vector_prefix(len(vector), False) + vector.tobytes())
    else:
        prefix = _dense_vector_prefix(vector.shape[1], True)
        target.multi_dense.MergeFromString(b"".join([prefix + row.tobytes() for row in vector]))


def convert_vector_struct(vector: Any) -> grpc.Vectors:
    """`RestToGrpc.convert_vector_struct`, which also takes numpy arrays and dicts of them

    Reading the bytes of an array is many times faster than turning it into python floats which
    protobuf then converts to C floats one by one.
    """
    if _is_numpy_vector(vector):
        vectors = grpc.Vectors()
        _set_numpy_vector(vectors.vector, vector)
        return vectors
    if isinstance(vector, dict) and vector and all(map(_is_numpy_vector, vector.values())):
        vectors = grpc.Vectors()
        for name, value in vector.items():
            _set_numpy_vector(vectors.vectors.vectors[name], value)
        return vectors

    if isinstance(vector, np.ndarray):
        vector = vector.tolist()
    elif isinstance(vector, dict):
        vector = {
            name: value.tolist() if isinstance(value, np.ndarray) else value
            for name, value in vector.items()
        }
    return RestToGrpc.convert_vector_struct(vector)


def upload_batch_grpc(
    points_client: grpc.PointsStub,
    collection_name: str,
    batch: rest.Batch | tuple,  # type: ignore[name-defined]
    max_retries: int,
    shard_key_selector: grpc.ShardKeySelector | None,  # type: ignore[name-defined]
    update_filter: grpc.Filter | None,
    update_mode: grpc.UpdateMode = None,  # type: ignore  # protobuf < 5.29 does not allow Union[enum, None]
    wait: bool = False,
    timeout: int | None = None,
) -> bool:
    ids_batch, vectors_batch, payload_batch = batch

    ids_batch = (
        (grpc.PointId(uuid=str(uuid4())) for _ in count()) if ids_batch is None else ids_batch
    )
    payload_batch = (None for _ in count()) if payload_batch is None else payload_batch

    points = [
        grpc.PointStruct(
            id=RestToGrpc.convert_extended_point_id(idx)
            if not isinstance(idx, grpc.PointId)
            else idx,
            vectors=convert_vector_struct(vector),
            payload=payload_to_grpc(payload or {}),
        )
        for idx, vector, payload in zip(ids_batch, vectors_batch, payload_batch)
    ]

    attempt = 0
    while attempt < max_retries:
        try:
            points_client.Upsert(
                grpc.UpsertPoints(
                    collection_name=collection_name,
                    points=points,
                    wait=wait,
                    shard_key_selector=shard_key_selector,
                    update_filter=update_filter,
                    update_mode=update_mode,
                ),
                timeout=timeout,
            )
            break
        except ResourceExhaustedResponse as ex:
            show_warning(
                message=f"Batch upload failed due to rate limit. Waiting for {ex.retry_after_s} seconds before retrying...",
                category=UserWarning,
                stacklevel=8,
            )
            sleep(ex.retry_after_s)

        except Exception as e:
            show_warning(
                message=f"Batch upload failed {attempt + 1} times. Retrying...",
                category=UserWarning,
                stacklevel=8,
            )

            if attempt == max_retries - 1:
                raise e

            attempt += 1
    return True


class GrpcBatchUploader(BaseUploader):
    def __init__(
        self,
        host: str,
        port: int,
        collection_name: str,
        max_retries: int,
        wait: bool = False,
        shard_key_selector: types.ShardKeySelector | None = None,
        update_filter: types.Filter | None = None,
        update_mode: types.UpdateMode | None = None,
        **kwargs: Any,
    ):
        self.collection_name = collection_name
        self._host = host
        self._port = port
        self.max_retries = max_retries
        self._kwargs = kwargs
        self._wait = wait
        self._shard_key_selector = (
            RestToGrpc.convert_shard_key_selector(shard_key_selector)
            if shard_key_selector is not None
            else None
        )
        self._timeout = kwargs.pop("timeout", None)
        self._update_filter = (
            RestToGrpc.convert_filter(update_filter)
            if isinstance(update_filter, rest.Filter)  # type: ignore[attr-defined]
            else update_filter
        )
        self._update_mode = (
            RestToGrpc.convert_update_mode(update_mode)
            if isinstance(update_mode, rest.UpdateMode)  # type: ignore[attr-defined]
            else update_mode
        )

    @staticmethod
    def _from_numpy(vectors: types.NumpyArray) -> Any:
        # convert_vector_struct reads plain arrays, others become lists as before: the rows of a
        # matrix are 2-d, a masked array hides values, an array of objects may hold anything
        return vectors if type(vectors) in (np.ndarray, np.memmap) else vectors.tolist()

    @classmethod
    def start(
        cls,
        collection_name: str | None = None,
        host: str = "localhost",
        port: int = 6334,
        max_retries: int = 3,
        **kwargs: Any,
    ) -> "GrpcBatchUploader":
        if not collection_name:
            raise RuntimeError("Collection name could not be empty")

        return cls(
            host=host,
            port=port,
            collection_name=collection_name,
            max_retries=max_retries,
            **kwargs,
        )

    def process_upload(self, items: Iterable[Any]) -> Generator[bool, None, None]:
        channel = get_channel(host=self._host, port=self._port, **self._kwargs)
        points_client = grpc.PointsStub(channel)
        for batch in items:
            yield upload_batch_grpc(
                points_client,
                self.collection_name,
                batch,
                shard_key_selector=self._shard_key_selector,
                update_filter=self._update_filter,
                update_mode=self._update_mode,
                max_retries=self.max_retries,
                wait=self._wait,
                timeout=self._timeout,
            )

    def process(self, items: Iterable[Any]) -> Iterable[bool]:
        yield from self.process_upload(items)
