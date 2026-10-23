import asyncio
import inspect
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

from qdrant_client import AsyncQdrantClient, QdrantClient, models
from qdrant_client.uploader.rest_uploader import batch_to_points


@pytest.fixture(params=[QdrantClient, AsyncQdrantClient])
def client_class(request):
    return request.param


def finish(result: Any) -> Any:
    return asyncio.run(result) if inspect.isawaitable(result) else result


@pytest.mark.parametrize("method", ["upload_points", "upload_collection"])
@pytest.mark.parametrize("batch_size", [0, -1])
@pytest.mark.parametrize("empty", [False, True])
def test_invalid_batch_size_does_not_consume_input(
    client_class, method: str, batch_size: int, empty: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = client_class(":memory:")
    consumed = []

    def items(value):
        consumed.append(value)
        if not empty:
            yield value

    values = items(models.PointStruct(id=1, vector=[1.0]) if method == "upload_points" else [1.0])
    kwargs = {} if method == "upload_points" else {"ids": items(1), "payload": items({"n": 0})}

    upload = MagicMock()
    monkeypatch.setattr(client._client, method, upload)
    try:
        with pytest.raises(ValueError, match="batch_size"):
            getattr(client, method)("test", values, batch_size=batch_size, **kwargs)
        assert consumed == []
        upload.assert_not_called()
    finally:
        finish(client.close())


@pytest.mark.parametrize("prefer_grpc", [False, True])
@pytest.mark.parametrize("input_kind", ["points", "list", "generator", "numpy", "named_numpy"])
@pytest.mark.parametrize("batch_size", [0, -1, 1, np.int64(2)])
def test_remote_upload_batch_sizes(
    client_class,
    prefer_grpc: bool,
    input_kind: str,
    batch_size: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = client_class(
        url="http://localhost:1",
        prefer_grpc=prefer_grpc,
        check_compatibility=False,
        trust_env=False,
    )
    vectors = [[1.0], [2.0], [3.0]]
    payload = [{"n": n} for n in range(3)]
    batches = []

    def collect_batches(*, batches_iterator, **kwargs):
        batches.extend(batches_iterator)

    dispatch = MagicMock(side_effect=collect_batches)
    # Keep the real REST/gRPC batching path, but never start workers or send requests.
    monkeypatch.setattr(client._client, "_upload_collection", dispatch)
    try:
        if input_kind == "points":
            method = client.upload_points
            kwargs = {
                "points": [
                    models.PointStruct(id=n, vector=vector, payload=payload[n])
                    for n, vector in enumerate(vectors)
                ]
            }
        else:
            inputs = {
                "list": vectors,
                "generator": iter(vectors),
                "numpy": np.array(vectors),
                "named_numpy": {"dense": np.array(vectors)},
            }
            method = client.upload_collection
            kwargs = {"vectors": inputs[input_kind], "ids": [0, 1, 2], "payload": payload}

        if batch_size < 1:
            with pytest.raises(ValueError, match="batch_size"):
                method("test", batch_size=batch_size, parallel=2, **kwargs)
            dispatch.assert_not_called()
        else:
            method("test", batch_size=batch_size, parallel=2, **kwargs)
            assert [len(batch[0]) for batch in batches] == (
                [1, 1, 1] if batch_size == 1 else [2, 1]
            )
            assert [point_id for batch in batches for point_id in batch[0]] == [0, 1, 2]
            assert [item for batch in batches for item in batch[2]] == payload
            expected_vectors = (
                [{"dense": vector} for vector in vectors]
                if input_kind == "named_numpy"
                else vectors
            )
            assert [point.vector for batch in batches for point in batch_to_points(batch)] == (
                expected_vectors
            )
    finally:
        finish(client.close())


@pytest.mark.parametrize("method", ["upload_points", "upload_collection"])
@pytest.mark.parametrize("batch_size", [0, -1, 1, np.int64(2)])
def test_local_upload_batch_sizes(client_class, method: str, batch_size: int) -> None:
    client = client_class(":memory:")
    try:
        finish(
            client.create_collection(
                "test", vectors_config=models.VectorParams(size=1, distance=models.Distance.DOT)
            )
        )
        kwargs = (
            {"points": [models.PointStruct(id=1, vector=[1.0])]}
            if method == "upload_points"
            else {"vectors": [[1.0]], "ids": [1]}
        )
        if batch_size < 1:
            with pytest.raises(ValueError, match="batch_size"):
                getattr(client, method)("test", batch_size=batch_size, **kwargs)
            assert finish(client.count("test")).count == 0
        else:
            # Local mode still uploads all points at once, regardless of valid batch size.
            getattr(client, method)("test", batch_size=batch_size, **kwargs)
            assert finish(client.count("test")).count == 1
    finally:
        finish(client.close())
