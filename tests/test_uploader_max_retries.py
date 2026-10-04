from unittest.mock import MagicMock

import pytest

from qdrant_client.http import models as rest
from qdrant_client.uploader import grpc_uploader, rest_uploader
from qdrant_client.uploader.grpc_uploader import upload_batch_grpc

_BATCH = ([1, 2, 3], [[0.1], [0.2], [0.3]], [None, None, None])


def _rest_client() -> MagicMock:
    client = MagicMock()
    client.points_api.upsert_points.side_effect = lambda **_kwargs: None
    return client


def _grpc_client() -> MagicMock:
    client = MagicMock()
    client.Upsert.side_effect = lambda *_args, **_kwargs: None
    return client


@pytest.mark.parametrize("max_retries", [0, -1])
def test_rest_upload_batch_rejects_non_positive_max_retries(max_retries: int) -> None:
    """A non-positive retry budget must not look like a successful upload.

    ``max_retries`` counts attempts, not extra retries, so anything below 1 used to
    skip the loop and return ``True`` without sending a single point.
    """
    client = _rest_client()

    with pytest.raises(ValueError, match="max_retries"):
        rest_uploader.upload_batch(
            openapi_client=client,
            collection_name="c",
            batch=_BATCH,
            max_retries=max_retries,
            shard_key_selector=None,
            update_filter=None,
            wait=False,
        )

    assert client.points_api.upsert_points.call_count == 0


@pytest.mark.parametrize("max_retries", [0, -1])
def test_grpc_upload_batch_rejects_non_positive_max_retries(max_retries: int) -> None:
    client = _grpc_client()

    with pytest.raises(ValueError, match="max_retries"):
        upload_batch_grpc(
            points_client=client,
            collection_name="c",
            batch=_BATCH,
            max_retries=max_retries,
            shard_key_selector=None,
            update_filter=None,
            wait=False,
        )

    assert client.Upsert.call_count == 0


def test_rest_upload_batch_still_uploads_with_a_single_attempt() -> None:
    """One attempt is the smallest valid budget and must go through unchanged."""
    client = _rest_client()

    assert rest_uploader.upload_batch(
        openapi_client=client,
        collection_name="c",
        batch=_BATCH,
        max_retries=1,
        shard_key_selector=None,
        update_filter=None,
        wait=False,
    )
    assert client.points_api.upsert_points.call_count == 1


def test_grpc_upload_batch_still_uploads_with_a_single_attempt() -> None:
    client = _grpc_client()

    assert upload_batch_grpc(
        points_client=client,
        collection_name="c",
        batch=_BATCH,
        max_retries=1,
        shard_key_selector=None,
        update_filter=None,
        wait=False,
    )
    assert client.Upsert.call_count == 1


def test_rest_upload_batch_default_budget_is_unaffected() -> None:
    """The public default is 3; this pins that the guard does not touch it."""
    import inspect

    from qdrant_client.qdrant_remote import QdrantRemote

    default = inspect.signature(QdrantRemote.upload_points).parameters["max_retries"].default
    assert default == 3


def test_rest_upload_batch_accepts_an_explicit_vector_model() -> None:
    """Guard placement must not disturb the normal call path for real point models."""
    client = _rest_client()
    points = [rest.PointStruct(id=1, vector=[0.1])]

    assert rest_uploader.upload_batch(
        openapi_client=client,
        collection_name="c",
        batch=([p.id for p in points], [p.vector for p in points], [None]),
        max_retries=1,
        shard_key_selector=None,
        update_filter=None,
        wait=True,
    )
    assert client.points_api.upsert_points.call_count == 1
