from contextlib import closing

import pytest

from qdrant_client import QdrantClient, grpc, models


SELECTORS = [
    (grpc.WithPayloadSelector(include=grpc.PayloadIncludeSelector(fields=["a"])), ["a"]),
    (
        grpc.WithPayloadSelector(exclude=grpc.PayloadExcludeSelector(fields=["a"])),
        models.PayloadSelectorExclude(exclude=["a"]),
    ),
    (grpc.WithPayloadSelector(enable=True), True),
    (grpc.WithPayloadSelector(enable=False), False),
]


@pytest.fixture
def client():
    with closing(QdrantClient(":memory:")) as client:
        client.create_collection(
            "c", vectors_config=models.VectorParams(size=2, distance=models.Distance.DOT)
        )
        client.upsert("c", [models.PointStruct(id=1, vector=[1.0, 2.0], payload={"a": 1, "b": 2})])
        yield client


@pytest.mark.parametrize("grpc_selector, rest_selector", SELECTORS)
def test_grpc_with_payload_selector(client, grpc_selector, rest_selector):
    calls = [
        lambda with_payload: client.retrieve("c", [1], with_payload=with_payload)[0],
        lambda with_payload: client.scroll("c", with_payload=with_payload)[0][0],
        lambda with_payload: client.query_points(
            "c", query=[1.0, 0.0], with_payload=with_payload
        ).points[0],
    ]
    for call in calls:
        assert call(grpc_selector).payload == call(rest_selector).payload
