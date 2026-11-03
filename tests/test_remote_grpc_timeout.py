from unittest.mock import MagicMock, PropertyMock, patch

import pytest

from qdrant_client import grpc, models
from qdrant_client.qdrant_remote import QdrantRemote

DEFAULT_TIMEOUT = 5
CALL_TIMEOUT = 77

# (QdrantRemote method, kwargs besides collection_name, PointsStub method it calls)
CASES = [
    (
        "upsert",
        {"points": [models.PointStruct(id=1, vector=[0.1, 0.2])]},
        "Upsert",
    ),
    (
        "update_vectors",
        {"points": [models.PointVectors(id=1, vector=[0.1, 0.2])]},
        "UpdateVectors",
    ),
    (
        "delete_vectors",
        {"points": [1], "vectors": ["x"]},
        "DeleteVectors",
    ),
    ("delete", {"points_selector": models.PointIdsList(points=[1])}, "Delete"),
    ("set_payload", {"payload": {"a": 1}, "points": [1]}, "SetPayload"),
    ("overwrite_payload", {"payload": {"a": 1}, "points": [1]}, "OverwritePayload"),
    ("delete_payload", {"keys": ["a"], "points": [1]}, "DeletePayload"),
    ("clear_payload", {"points_selector": [1]}, "ClearPayload"),
    (
        "batch_update_points",
        {
            "update_operations": [
                models.DeleteOperation(delete=models.PointIdsList(points=[1])),
            ]
        },
        "UpdateBatch",
    ),
]


def _remote() -> QdrantRemote:
    return QdrantRemote(
        host="localhost", prefer_grpc=True, timeout=DEFAULT_TIMEOUT, check_compatibility=False
    )


@pytest.mark.parametrize("method, kwargs, stub_method", CASES)
@pytest.mark.parametrize("call_timeout, expected", [(CALL_TIMEOUT, CALL_TIMEOUT), (None, DEFAULT_TIMEOUT)])
def test_grpc_write_calls_use_per_call_timeout(method, kwargs, stub_method, call_timeout, expected):
    stub = MagicMock()
    stub_call = getattr(stub, stub_method)
    # The conversion of the response is irrelevant here; only the call deadline is.
    stub_call.side_effect = RuntimeError("stop after the gRPC call")

    remote = _remote()
    with patch.object(QdrantRemote, "grpc_points", new_callable=PropertyMock, return_value=stub):
        with pytest.raises(RuntimeError, match="stop after the gRPC call"):
            getattr(remote, method)(
                collection_name="c", timeout=call_timeout, **kwargs
            )

    assert stub_call.call_count == 1
    assert stub_call.call_args.kwargs["timeout"] == expected
