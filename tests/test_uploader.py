import os
import pickle
from typing import Any, Callable
from unittest.mock import MagicMock

import numpy as np
import pytest

from qdrant_client import QdrantClient, models
from qdrant_client.conversions.conversion import RestToGrpc
from qdrant_client.parallel_processor import ParallelWorkerPool, Worker
from qdrant_client.uploader.grpc_uploader import (
    GrpcBatchUploader,
    convert_vector_struct,
    upload_batch_grpc,
)
from qdrant_client.uploader.rest_uploader import RestBatchUploader, batch_to_points, upload_batch
from qdrant_client.uploader.uploader import BaseUploader


@pytest.mark.parametrize("prefer_grpc", [False, True])
def test_upload_batch_retries_then_raises(prefer_grpc: bool) -> None:
    client = MagicMock()
    upsert = client.Upsert if prefer_grpc else client.points_api.upsert_points
    upsert.side_effect = ConnectionError("upload failed")
    upload = upload_batch_grpc if prefer_grpc else upload_batch
    batch = ([1, 2, 3], [[0.1], [0.2], [0.3]], [None, None, None])

    with pytest.warns(UserWarning, match="Retrying") as record:
        with pytest.raises(ConnectionError, match="upload failed"):
            upload(
                client,
                "test_collection",
                batch,
                max_retries=2,
                shard_key_selector=None,
                update_filter=None,
            )

    assert upsert.call_count == 3
    # the last failure raises instead of announcing another retry
    assert [str(warning.message) for warning in record] == [
        "Batch upload failed 1 times. Retrying...",
        "Batch upload failed 2 times. Retrying...",
    ]


def test_upload_rejects_negative_max_retries() -> None:
    # nothing listens on this port, the check has to fire before any request
    client = QdrantClient(url="http://localhost:1", check_compatibility=False)

    # parallel workers would hide the error behind a generic RuntimeError
    with pytest.raises(ValueError, match="max_retries"):
        client.upload_points(
            "test_collection",
            [models.PointStruct(id=1, vector=[0.1])],
            max_retries=-1,
            parallel=2,
        )


def test_negative_parallel_raises() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        ParallelWorkerPool(-1, Worker)


@pytest.mark.parametrize("cpu_count, expected", [(3, 3), (None, 1)])
def test_zero_parallel_uses_all_cores(
    monkeypatch: pytest.MonkeyPatch, cpu_count: int | None, expected: int
) -> None:
    monkeypatch.setattr(os, "cpu_count", lambda: cpu_count)

    assert ParallelWorkerPool(0, Worker).num_workers == expected


def _serialized_or_error(convert: Callable[[Any], Any], vector: Any) -> Any:
    try:
        # bytes, because NaN != NaN in message comparison
        return convert(vector).SerializeToString(deterministic=True)
    except Exception as e:
        return type(e)


@pytest.mark.parametrize(
    "vector",
    [
        # rounds differently to a float directly and through a double
        np.array([2**60 + 2**36 + 1, -5], dtype=np.int64),
        # a signaling NaN, which becomes quiet through a double
        np.array([0x7FA00000], dtype=np.uint32).view(np.float32),
        # lengths of more than one byte
        np.arange(40, dtype=np.float32),
        np.arange(120, dtype=np.float32).reshape(3, 40),
        # converted to lists, to be converted or rejected as before
        np.zeros((0, 3)),
        np.zeros((2, 2, 2)),
        np.array([1j], dtype=np.complex64),
        np.ma.masked_array([1.0, 2.0, 3.0], mask=[False, True, False]),
        {"dense": np.arange(3, dtype=np.float32), "multi": np.ones((2, 2))},
        {"dense": np.arange(3, dtype=np.float32), "list": [1.0, 2.0]},
    ],
    ids=lambda vector: type(vector).__name__,
)
def test_grpc_uploader_numpy_vectors(vector: Any) -> None:
    def old_path(vector: Any) -> Any:
        # numpy vectors used to be converted to lists before the conversion
        if isinstance(vector, np.ndarray):
            vector = vector.tolist()
        elif isinstance(vector, dict):
            vector = {
                name: value.tolist() if isinstance(value, np.ndarray) else value
                for name, value in vector.items()
            }
        return RestToGrpc.convert_vector_struct(vector)

    assert _serialized_or_error(convert_vector_struct, vector) == _serialized_or_error(
        old_path, vector
    )


@pytest.mark.parametrize(
    "vectors",
    [
        np.arange(15, dtype=np.float32).reshape(5, 3),
        # the rows of a matrix are 2-d
        np.matrix(np.arange(15, dtype=np.float32).reshape(5, 3)),
        {"dense": np.ones((5, 3)), "multi": np.ones((5, 2, 3))},
    ],
    ids=lambda vectors: type(vectors).__name__,
)
def test_grpc_uploader_numpy_batches(vectors: Any) -> None:
    def converted(uploader: Any, convert: Callable[[Any], Any]) -> list[list[Any]]:
        batches = uploader.iterate_batches(vectors=vectors, payload=None, ids=None, batch_size=2)
        return [
            [_serialized_or_error(convert, vector) for vector in vectors_batch]
            for _, vectors_batch, _ in batches
        ]

    # the batches used to be converted to lists for both protocols
    assert converted(GrpcBatchUploader, convert_vector_struct) == converted(
        BaseUploader, RestToGrpc.convert_vector_struct
    )


def _multivectors(*lengths: int) -> np.ndarray:
    """Multivectors with these numbers of 3-d vectors, in an array of arrays as numpy keeps them"""
    multivectors: np.ndarray = np.empty(len(lengths), dtype=object)
    for i, length in enumerate(lengths):
        multivectors[i] = np.arange(length * 3, dtype=np.float32).reshape(length, 3)
    return multivectors


@pytest.mark.parametrize(
    "vectors",
    [
        np.arange(15, dtype=np.float32).reshape(5, 3),
        _multivectors(2, 3, 1, 4, 2),
        {"dense": np.ones((5, 3)), "multi": _multivectors(2, 3, 1, 4, 2)},
        {"dense": np.ones((5, 3)), "multi": np.ones((5, 2, 3), dtype=np.float16)},
    ],
    ids=["dense", "multi", "named", "named-same-lengths"],
)
def test_rest_uploader_numpy_batches(vectors: Any) -> None:
    def points(uploader: type[BaseUploader]) -> list[Any]:
        batches = uploader.iterate_batches(
            vectors=vectors, payload=None, ids=list(range(5)), batch_size=2
        )
        # pickled, as for parallel workers
        return [batch_to_points(pickle.loads(pickle.dumps(batch))) for batch in batches]

    # the batches used to be converted to lists before they were sent to the workers
    assert points(RestBatchUploader) == points(BaseUploader)


@pytest.mark.parametrize("value", [(0.1, 0.2), "abc", None], ids=["tuple", "str", "None"])
def test_grpc_uploader_rejects_unknown_named_vectors(value: Any) -> None:
    sparse = models.SparseVector(indices=[1], values=[0.5])

    # used to be skipped, uploading the point without the vector
    with pytest.raises(ValueError, match="invalid VectorStruct model: vector 'dense'"):
        convert_vector_struct({"dense": value, "sparse": sparse})
