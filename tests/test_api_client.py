from unittest.mock import AsyncMock, MagicMock

import pytest

from qdrant_client.http.api_client import ApiClient, AsyncApiClient

COLLECTION_EXISTS_URL = "/collections/{collection_name}/exists"


def build_sync_url(collection_name: str) -> str:
    client = ApiClient("http://localhost:6333")
    client.send = MagicMock(return_value=None)
    client.request(
        type_=None,
        method="GET",
        url=COLLECTION_EXISTS_URL,
        path_params={"collection_name": collection_name},
    )
    return str(client.send.call_args[0][0].url)


async def build_async_url(collection_name: str) -> str:
    client = AsyncApiClient("http://localhost:6333")
    client.send = AsyncMock(return_value=None)
    await client.request(
        type_=None,
        method="GET",
        url=COLLECTION_EXISTS_URL,
        path_params={"collection_name": collection_name},
    )
    return str(client.send.call_args[0][0].url)


class TestPathParamsEncoding:
    def test_plain_collection_name_is_not_escaped(self):
        assert build_sync_url("my_collection-1") == (
            "http://localhost:6333/collections/my_collection-1/exists"
        )

    def test_slash_in_collection_name_stays_a_single_path_segment(self):
        # without escaping, the name would add a path segment and the request would be
        # sent to a route which does not exist
        assert build_sync_url("example/collection1") == (
            "http://localhost:6333/collections/example%2Fcollection1/exists"
        )

    def test_space_in_collection_name_is_escaped(self):
        assert build_sync_url("my collection") == (
            "http://localhost:6333/collections/my%20collection/exists"
        )

    def test_non_string_path_params_are_supported(self):
        client = ApiClient("http://localhost:6333")
        client.send = MagicMock(return_value=None)
        client.request(
            type_=None,
            method="GET",
            url="/collections/{collection_name}/shards/{shard_id}",
            path_params={"collection_name": "test_collection", "shard_id": 1},
        )
        assert str(client.send.call_args[0][0].url) == (
            "http://localhost:6333/collections/test_collection/shards/1"
        )

    @pytest.mark.asyncio
    async def test_async_client_escapes_path_params(self):
        assert await build_async_url("example/collection1") == (
            "http://localhost:6333/collections/example%2Fcollection1/exists"
        )

    @pytest.mark.asyncio
    async def test_async_plain_collection_name_is_not_escaped(self):
        assert await build_async_url("my_collection-1") == (
            "http://localhost:6333/collections/my_collection-1/exists"
        )
