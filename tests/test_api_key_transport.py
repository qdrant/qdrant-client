from concurrent.futures import ThreadPoolExecutor

import grpc
import httpx
import pytest

from qdrant_client import AsyncQdrantClient, QdrantClient
from qdrant_client import grpc as q_grpc

pytestmark = [
    pytest.mark.filterwarnings("ignore:Api key is used with an insecure connection"),
    pytest.mark.filterwarnings("ignore:`api-key` has been passed in `headers`"),
]


@pytest.fixture
def grpc_endpoint():
    received = []

    class RecordingCollections(q_grpc.CollectionsServicer):
        def List(self, request, context):
            received.append(list(context.invocation_metadata()))
            return q_grpc.ListCollectionsResponse(collections=[], time=0.0)

    with ThreadPoolExecutor(max_workers=1) as executor:
        server = grpc.server(executor)
        q_grpc.add_CollectionsServicer_to_server(RecordingCollections(), server)
        port = server.add_insecure_port("127.0.0.1:0")
        assert port != 0
        server.start()
        try:
            yield port, received
        finally:
            server.stop(grace=0).wait(timeout=5)


@pytest.mark.asyncio
@pytest.mark.parametrize("is_async", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize("api_key", [None, "from-param"], ids=["header-only", "override"])
async def test_api_key_reaches_rest_transport(is_async, api_key):
    received = []

    def handle(request):
        received.append(request)
        return httpx.Response(
            200, json={"result": {"collections": []}, "status": "ok", "time": 0.0}
        )

    kwargs = {
        "url": "https://localhost:6333",
        "api_key": api_key,
        "headers": {"api-key": "from-headers", "x-custom": "value"},
        "check_compatibility": False,
        "transport": httpx.MockTransport(handle),
    }
    if is_async:
        async_client = AsyncQdrantClient(**kwargs)
        try:
            result = await async_client.get_collections()
        finally:
            await async_client.close()
    else:
        client = QdrantClient(**kwargs)
        try:
            result = client.get_collections()
        finally:
            client.close()

    assert result.collections == []
    assert len(received) == 1
    assert received[0].method == "GET"
    assert received[0].url.path == "/collections"
    assert received[0].headers.get_list("api-key") == [api_key or "from-headers"]
    assert received[0].headers["x-custom"] == "value"


@pytest.mark.asyncio
@pytest.mark.parametrize("is_async", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize("api_key", [None, "from-param"], ids=["header-only", "override"])
async def test_api_key_reaches_grpc_server(grpc_endpoint, is_async, api_key):
    port, received = grpc_endpoint
    kwargs = {
        "host": "127.0.0.1",
        "grpc_port": port,
        "prefer_grpc": True,
        "https": False,
        "api_key": api_key,
        "headers": {"api-key": "from-headers", "x-custom": "value"},
        "check_compatibility": False,
        "timeout": 5,
    }
    if is_async:
        async_client = AsyncQdrantClient(**kwargs)
        try:
            result = await async_client.get_collections()
        finally:
            await async_client.close()
    else:
        client = QdrantClient(**kwargs)
        try:
            result = client.get_collections()
        finally:
            client.close()

    assert result.collections == []
    assert len(received) == 1
    assert [value for key, value in received[0] if key == "api-key"] == [api_key or "from-headers"]
    assert ("x-custom", "value") in received[0]
