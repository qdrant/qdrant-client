from unittest.mock import MagicMock

import pytest

from qdrant_client import QdrantClient, grpc, models
from qdrant_client.qdrant_remote import QdrantRemote

POINTS_RESPONSE = grpc.PointsOperationResponse(
    result=grpc.UpdateResult(operation_id=1, status=grpc.UpdateStatus.Completed),
    time=0.0,
)

FALSY_SHARD_KEYS = [0, ""]


def make_grpc_client() -> tuple[QdrantClient, MagicMock]:
    client = QdrantClient(prefer_grpc=True, check_compatibility=False)
    stub = MagicMock()
    for method in (
        "Delete",
        "DeleteVectors",
        "SetPayload",
        "OverwritePayload",
        "DeletePayload",
        "ClearPayload",
    ):
        getattr(stub, method).return_value = POINTS_RESPONSE
    client._client._grpc_points_client_pool = [stub]
    return client, stub


@pytest.mark.parametrize("shard_key", FALSY_SHARD_KEYS)
@pytest.mark.parametrize(
    "call,stub_method",
    [
        (lambda client, kw: client.delete("c", models.PointIdsList(points=[1]), **kw), "Delete"),
        (
            lambda client, kw: client.delete_vectors(
                "c", ["v"], models.PointIdsList(points=[1]), **kw
            ),
            "DeleteVectors",
        ),
        (
            lambda client, kw: client.set_payload(
                "c", {"a": 1}, models.PointIdsList(points=[1]), **kw
            ),
            "SetPayload",
        ),
        (
            lambda client, kw: client.overwrite_payload(
                "c", {"a": 1}, models.PointIdsList(points=[1]), **kw
            ),
            "OverwritePayload",
        ),
        (
            lambda client, kw: client.delete_payload(
                "c", ["a"], models.PointIdsList(points=[1]), **kw
            ),
            "DeletePayload",
        ),
        (
            lambda client, kw: client.clear_payload("c", models.PointIdsList(points=[1]), **kw),
            "ClearPayload",
        ),
    ],
)
def test_grpc_falsy_shard_key_selector_is_not_dropped(call, stub_method, shard_key):
    # 0 and "" are valid shard keys (ShardKey = StrictInt | StrictStr), but used to be
    # dropped by `shard_key_selector or opt_shard_key_selector`, silently applying
    # the operation to all shards
    client, stub = make_grpc_client()
    call(client, {"shard_key_selector": shard_key})
    request = getattr(stub, stub_method).call_args[0][0]
    assert request.HasField(
        "shard_key_selector"
    ), f"{stub_method} dropped falsy shard key {shard_key!r}"
    assert request.shard_key_selector == grpc.ShardKeySelector(
        shard_keys=[models_shard_key_to_grpc(shard_key)]
    )


def models_shard_key_to_grpc(shard_key: models.ShardKey) -> grpc.ShardKey:
    if isinstance(shard_key, str):
        return grpc.ShardKey(keyword=shard_key)
    return grpc.ShardKey(number=shard_key)


def test_grpc_embedded_shard_key_still_used():
    client, stub = make_grpc_client()
    client.delete("c", models.PointIdsList(points=[1], shard_key="us"))
    request = stub.Delete.call_args[0][0]
    assert request.HasField("shard_key_selector")
    assert request.shard_key_selector == grpc.ShardKeySelector(
        shard_keys=[grpc.ShardKey(keyword="us")]
    )


def test_grpc_explicit_shard_key_overrides_embedded():
    client, stub = make_grpc_client()
    client.delete("c", models.PointIdsList(points=[1], shard_key="us"), shard_key_selector="eu")
    request = stub.Delete.call_args[0][0]
    assert request.shard_key_selector == grpc.ShardKeySelector(
        shard_keys=[grpc.ShardKey(keyword="eu")]
    )


def test_rest_selector_preserves_embedded_shard_key():
    # used to be overwritten with None when no explicit shard_key_selector was passed,
    # silently applying the operation to all shards (while grpc mode kept the shard key)
    selector = QdrantRemote._try_argument_to_rest_selector(
        models.PointIdsList(points=[1], shard_key="us"), None
    )
    assert selector.shard_key == "us"

    selector = QdrantRemote._try_argument_to_rest_selector(
        models.PointIdsList(points=[1], shard_key="us"), "eu"
    )
    assert selector.shard_key == "eu"

    selector = QdrantRemote._try_argument_to_rest_selector(
        models.PointIdsList(points=[1], shard_key="us"), 0
    )
    assert selector.shard_key == 0


def test_rest_points_and_filter_returns_embedded_shard_key():
    _points, _filter, _shard_key = QdrantRemote._try_argument_to_rest_points_and_filter(
        models.PointIdsList(points=[1], shard_key="us")
    )
    assert _points == [1]
    assert _filter is None
    assert _shard_key == "us"

    _points, _filter, _shard_key = QdrantRemote._try_argument_to_rest_points_and_filter(
        models.FilterSelector(filter=models.Filter(), shard_key=0)
    )
    assert _points is None
    assert _filter is not None
    assert _shard_key == 0

    _points, _filter, _shard_key = QdrantRemote._try_argument_to_rest_points_and_filter(
        models.PointIdsList(points=[1])
    )
    assert _shard_key is None


def test_rest_delete_vectors_uses_embedded_shard_key():
    client = QdrantClient(check_compatibility=False)
    api = MagicMock()
    api.delete_vectors.return_value = MagicMock(
        result=models.UpdateResult(operation_id=1, status=models.UpdateStatus.COMPLETED)
    )
    client._client.openapi_client.points_api = api
    client.delete_vectors("c", ["v"], models.PointIdsList(points=[1], shard_key="us"))
    request = api.delete_vectors.call_args.kwargs["delete_vectors"]
    assert request.shard_key == "us"
