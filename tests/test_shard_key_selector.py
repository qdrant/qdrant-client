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


def make_rest_client() -> tuple[QdrantClient, MagicMock]:
    client = QdrantClient(check_compatibility=False)
    api = MagicMock()
    rest_response = MagicMock(
        result=models.UpdateResult(operation_id=1, status=models.UpdateStatus.COMPLETED)
    )
    for method in (
        "delete_points",
        "delete_vectors",
        "set_payload",
        "overwrite_payload",
        "delete_payload",
        "clear_payload",
    ):
        getattr(api, method).return_value = rest_response
    client._client.openapi_client.points_api = api
    return client, api


REST_CASES = [
    (lambda c, sel, kw: c.delete("c", sel, **kw), "delete_points", "points_selector"),
    (
        lambda c, sel, kw: c.delete_vectors("c", ["v"], sel, **kw),
        "delete_vectors",
        "delete_vectors",
    ),
    (lambda c, sel, kw: c.set_payload("c", {"a": 1}, sel, **kw), "set_payload", "set_payload"),
    (
        lambda c, sel, kw: c.overwrite_payload("c", {"a": 1}, sel, **kw),
        "overwrite_payload",
        "set_payload",
    ),
    (
        lambda c, sel, kw: c.delete_payload("c", ["a"], sel, **kw),
        "delete_payload",
        "delete_payload",
    ),
    (lambda c, sel, kw: c.clear_payload("c", sel, **kw), "clear_payload", "points_selector"),
]


@pytest.mark.parametrize("api_call,api_method,request_kwarg", REST_CASES)
def test_rest_embedded_shard_key_propagates(api_call, api_method, request_kwarg):
    client, api = make_rest_client()
    api_call(client, models.PointIdsList(points=[1], shard_key="us"), {})
    request = getattr(api, api_method).call_args.kwargs[request_kwarg]
    assert request.shard_key == "us", f"{api_method} dropped embedded shard key"


@pytest.mark.parametrize("shard_key", FALSY_SHARD_KEYS)
@pytest.mark.parametrize("api_call,api_method,request_kwarg", REST_CASES)
def test_rest_explicit_falsy_shard_key_propagates(api_call, api_method, request_kwarg, shard_key):
    # embedded shard key present to also assert explicit falsy keys take precedence
    client, api = make_rest_client()
    api_call(
        client,
        models.PointIdsList(points=[1], shard_key="us"),
        {"shard_key_selector": shard_key},
    )
    request = getattr(api, api_method).call_args.kwargs[request_kwarg]
    assert request.shard_key == shard_key, f"{api_method} dropped falsy shard key {shard_key!r}"
