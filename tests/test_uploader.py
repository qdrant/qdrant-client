import os
from unittest.mock import MagicMock

import pytest

from qdrant_client import QdrantClient, models
from qdrant_client.common.client_exceptions import ResourceExhaustedResponse
from qdrant_client.parallel_processor import ParallelWorkerPool, Worker
from qdrant_client.uploader.grpc_uploader import upload_batch_grpc
from qdrant_client.uploader.rest_uploader import upload_batch


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


@pytest.mark.parametrize("prefer_grpc", [False, True])
@pytest.mark.parametrize("max_retries", [0, 2])
def test_rate_limit_exhausts_upload_retry_budget(prefer_grpc: bool, max_retries: int) -> None:
    client = MagicMock()
    upsert = client.Upsert if prefer_grpc else client.points_api.upsert_points
    error = ResourceExhaustedResponse("limited", retry_after_s=0)
    # A finite fallback ensures a broken retry loop cannot hang the test.
    upsert.side_effect = [error] * (max_retries + 2) + [RuntimeError("budget exceeded")]
    upload = upload_batch_grpc if prefer_grpc else upload_batch

    with pytest.raises(ResourceExhaustedResponse) as raised:
        upload(
            client,
            "test_collection",
            ([1], [[0.1]], [None]),
            max_retries=max_retries,
            shard_key_selector=None,
            update_filter=None,
        )

    assert raised.value is error
    assert upsert.call_count == max_retries + 1


@pytest.mark.parametrize("prefer_grpc", [False, True])
def test_rate_limits_and_other_failures_share_retry_budget(prefer_grpc: bool) -> None:
    client = MagicMock()
    upsert = client.Upsert if prefer_grpc else client.points_api.upsert_points
    error = ResourceExhaustedResponse("limited", retry_after_s=0)
    upsert.side_effect = [error, ConnectionError("temporary"), error, None]
    upload = upload_batch_grpc if prefer_grpc else upload_batch

    with pytest.raises(ResourceExhaustedResponse) as raised:
        upload(
            client,
            "test_collection",
            ([1], [[0.1]], [None]),
            max_retries=2,
            shard_key_selector=None,
            update_filter=None,
        )

    assert raised.value is error
    assert upsert.call_count == 3


@pytest.mark.parametrize("prefer_grpc", [False, True])
def test_upload_recovers_from_rate_limit_within_retry_budget(prefer_grpc: bool) -> None:
    client = MagicMock()
    upsert = client.Upsert if prefer_grpc else client.points_api.upsert_points
    upsert.side_effect = [ResourceExhaustedResponse("limited", retry_after_s=0), None]
    upload = upload_batch_grpc if prefer_grpc else upload_batch

    assert upload(
        client,
        "test_collection",
        ([1], [[0.1]], [None]),
        max_retries=1,
        shard_key_selector=None,
        update_filter=None,
    )
    assert upsert.call_count == 2


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
