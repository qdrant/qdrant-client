from concurrent import futures
from typing import Iterator

import grpc
import pytest

from qdrant_client import AsyncQdrantClient, QdrantClient
from qdrant_client import grpc as q_grpc
from qdrant_client.context_headers import async_headers, headers


@pytest.fixture
def grpc_endpoint() -> Iterator[tuple[int, list[dict[str, str]]]]:
    received: list[dict[str, str]] = []

    class RecordingCollections(q_grpc.CollectionsServicer):
        def List(
            self, request: q_grpc.ListCollectionsRequest, context: grpc.ServicerContext
        ) -> q_grpc.ListCollectionsResponse:
            received.append(dict(context.invocation_metadata()))
            return q_grpc.ListCollectionsResponse(collections=[], time=0.0)

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
    q_grpc.add_CollectionsServicer_to_server(RecordingCollections(), server)
    port = server.add_insecure_port("127.0.0.1:0")
    server.start()
    try:
        yield port, received
    finally:
        server.stop(grace=None)


def test_grpc_sends_mixed_case_headers(
    grpc_endpoint: tuple[int, list[dict[str, str]]],
) -> None:
    port, received = grpc_endpoint
    client = QdrantClient(
        host="127.0.0.1",
        grpc_port=port,
        prefer_grpc=True,
        check_compatibility=False,
        headers={"X-Request-Id": "from-client"},
    )

    with headers({"X-Trace-Id": "from-context"}):
        client.get_collections()

    assert received[0]["x-request-id"] == "from-client"
    assert received[0]["x-trace-id"] == "from-context"
    client.close()


@pytest.mark.asyncio
async def test_async_grpc_sends_mixed_case_headers(
    grpc_endpoint: tuple[int, list[dict[str, str]]],
) -> None:
    port, received = grpc_endpoint
    client = AsyncQdrantClient(
        host="127.0.0.1",
        grpc_port=port,
        prefer_grpc=True,
        check_compatibility=False,
        headers={"X-Request-Id": "from-client"},
    )

    async with async_headers({"X-Trace-Id": "from-context"}):
        await client.get_collections()

    assert received[0]["x-request-id"] == "from-client"
    assert received[0]["x-trace-id"] == "from-context"
    await client.close()
