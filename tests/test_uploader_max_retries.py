from unittest.mock import MagicMock

import pytest

from qdrant_client import QdrantClient, models
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
