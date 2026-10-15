import inspect
from typing import Any, Callable

import pytest

from qdrant_client import AsyncQdrantClient, QdrantClient, grpc, models


@pytest.mark.asyncio
@pytest.mark.parametrize("client_type", [QdrantClient, AsyncQdrantClient])
@pytest.mark.parametrize(
    "selector,expected",
    [
        pytest.param(grpc.WithPayloadSelector(enable=True), {"a": 1, "b": 2}, id="enabled"),
        pytest.param(grpc.WithPayloadSelector(enable=False), None, id="disabled"),
        pytest.param(
            grpc.WithPayloadSelector(include=grpc.PayloadIncludeSelector(fields=["a"])),
            {"a": 1},
            id="include",
        ),
        pytest.param(
            grpc.WithPayloadSelector(exclude=grpc.PayloadExcludeSelector(fields=["a"])),
            {"b": 2},
            id="exclude",
        ),
        pytest.param(
            grpc.WithPayloadSelector(include=grpc.PayloadIncludeSelector(fields=[])),
            {},
            id="empty-include",
        ),
        pytest.param(
            grpc.WithPayloadSelector(exclude=grpc.PayloadExcludeSelector(fields=[])),
            {"a": 1, "b": 2},
            id="empty-exclude",
        ),
    ],
)
async def test_local_reads_accept_grpc_payload_selectors(
    client_type: type[QdrantClient] | type[AsyncQdrantClient],
    selector: grpc.WithPayloadSelector,
    expected: models.Payload | None,
) -> None:
    """Local clients apply every gRPC selector without changing stored payloads."""

    async def call(method: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Run the same read contract against synchronous and asynchronous clients."""
        result = method(*args, **kwargs)
        return await result if inspect.isawaitable(result) else result

    client = client_type(":memory:")
    try:
        await call(client.create_collection, "test", vectors_config={})
        await call(
            client.upsert,
            "test",
            [models.PointStruct(id=1, vector={}, payload={"a": 1, "b": 2})],
        )

        retrieved = await call(client.retrieve, "test", [1], with_payload=selector)
        scrolled, _ = await call(client.scroll, "test", with_payload=selector)
        queried = await call(client.query_points, "test", with_payload=selector)
        for records in (retrieved, scrolled, queried.points):
            assert len(records) == 1
            assert records[0].id == 1
            assert records[0].payload == expected

        stored = await call(client.retrieve, "test", [1], with_payload=True)
        assert stored[0].payload == {"a": 1, "b": 2}
    finally:
        await call(client.close)
