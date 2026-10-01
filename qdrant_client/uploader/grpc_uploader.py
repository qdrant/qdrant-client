from itertools import count
from time import sleep
from typing import Any, Generator, Iterable
from uuid import uuid4

import numpy as np

from qdrant_client import grpc as grpc
from qdrant_client import models as rest
from qdrant_client.common.client_exceptions import ResourceExhaustedResponse
from qdrant_client.connection import get_channel
from qdrant_client.conversions.conversion import RestToGrpc, payload_to_grpc
from qdrant_client.uploader.uploader import BaseUploader
from qdrant_client.common.client_warnings import show_warning
from qdrant_client.conversions import common_types as types


def _varint(value: int) -> bytes:
    result = bytearray()
    while value > 0x7F:
        result.append(value & 0x7F | 0x80)
        value >>= 7
    result.append(value)
    return bytes(result)


def _length_delimited_tag(message: Any, field: str) -> bytes:
    return _varint(message.DESCRIPTOR.fields_by_name[field].number << 3 | 2)


_DENSE_VECTOR_DATA = _length_delimited_tag(grpc.DenseVector, "data")
_MULTI_DENSE_VECTOR_VECTORS = _length_delimited_tag(grpc.MultiDenseVector, "vectors")


def _dense_vector_bytes(vector: np.ndarray) -> bytes:
    """Serialized `DenseVector` with the values of a 1-d array"""
    # values out of the float range become inf, as protobuf does with python floats (pure python
    # protobuf < 6.30 also turns values slightly above the float maximum into inf, a cast rounds
    # them down to the maximum, as upb does)
    with np.errstate(over="ignore"):
        if vector.dtype.kind != "f" or vector.dtype.itemsize > 8:
            # through a double, the way `tolist()` and protobuf convert other numbers
            vector = vector.astype(np.float64)
        data = vector.astype("<f4", copy=False).tobytes()
    return _DENSE_VECTOR_DATA + _varint(len(data)) + data


def _set_numpy_vector(target: grpc.Vector, vector: np.ndarray) -> None:
    if vector.ndim == 2 and len(vector) > 0:
        target.multi_dense.MergeFromString(
            b"".join(
                _MULTI_DENSE_VECTOR_VECTORS + _varint(len(dense)) + dense
                for dense in map(_dense_vector_bytes, vector)
            )
        )
    else:  # an empty 2-d array is an empty dense vector, like an empty list
        target.dense.MergeFromString(_dense_vector_bytes(vector.reshape(-1)))


def _is_numpy_vector(vector: Any) -> bool:
    # other arrays take the path of lists, to fail or be converted the same way
    return isinstance(vector, np.ndarray) and vector.ndim in (1, 2) and vector.dtype.kind in "fiub"


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
        # convert_vector_struct reads the arrays directly
        return vectors

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
