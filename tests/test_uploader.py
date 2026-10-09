from typing import Any, Callable

import numpy as np
import pytest

from qdrant_client.conversions.conversion import RestToGrpc
from qdrant_client.uploader.grpc_uploader import GrpcBatchUploader, convert_vector_struct
from qdrant_client.uploader.uploader import BaseUploader


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
