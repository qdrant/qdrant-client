"""Tests for httpx.Client / httpx.AsyncClient injection (issue #1068)."""

import httpx
import pytest

from qdrant_client import AsyncQdrantClient, QdrantClient
from qdrant_client.http import ApiClient, AsyncApiClient
from qdrant_client.qdrant_remote import QdrantRemote
from qdrant_client.async_qdrant_remote import AsyncQdrantRemote


class TestSyncHttpClientInjection:
    """Caller-supplied httpx.Client is used verbatim by QdrantClient / QdrantRemote."""

    def test_injected_client_is_used_by_remote(self):
        injected = httpx.Client()
        try:
            client = QdrantClient(check_compatibility=False, http_client=injected)
            assert isinstance(client._client, QdrantRemote)
            # Path: QdrantClient -> QdrantRemote -> SyncApis -> ApiClient -> httpx.Client
            api_client = client._client.openapi_client.client
            assert isinstance(api_client, ApiClient)
            assert api_client._client is injected
        finally:
            injected.close()

    def test_injected_client_is_used_by_remote_directly(self):
        # Bypass the QdrantClient facade to confirm QdrantRemote.__init__ alone
        # honors the parameter.
        injected = httpx.Client()
        try:
            remote = QdrantRemote(check_compatibility=False, http_client=injected)
            api_client = remote.openapi_client.client
            assert isinstance(api_client, ApiClient)
            assert api_client._client is injected
        finally:
            injected.close()

    def test_default_path_still_builds_its_own_client(self):
        client = QdrantClient(check_compatibility=False)
        api_client = client._client.openapi_client.client
        # Sanity: the default branch ran and wired up a real httpx.Client.
        assert isinstance(api_client._client, httpx.Client)
        api_client._client.close()

    def test_injected_client_keeps_rest_headers_middleware(self):
        # rest_headers middleware must still be attached even when the user
        # supplies their own httpx.Client, otherwise observability of the
        # qdrant-client would regress as a side effect of opting in.
        from unittest.mock import MagicMock

        from qdrant_client.context_headers import headers

        injected = httpx.Client()
        try:
            client = QdrantClient(check_compatibility=False, http_client=injected)
            api_client = client._client.openapi_client.client
            # Drive the middleware chain with a fake request. If rest_headers
            # is in the chain, the x-tracing-id from the context will be
            # stamped onto the request before it reaches the inner send.
            request = httpx.Request("GET", "http://localhost:6333/collections")
            with headers({"x-tracing-id": "trace-1068"}):
                api_client.middleware(request, MagicMock(return_value="response"))
            assert request.headers.get("x-tracing-id") == "trace-1068"
        finally:
            injected.close()

    def test_init_options_round_trip_preserves_http_client(self):
        # The QdrantClient -> AsyncQdrantClient path uses _init_options; verify
        # the http_client is propagated so callers can build an async client
        # from a sync one without losing the injected transport.
        injected = httpx.Client()
        try:
            sync = QdrantClient(check_compatibility=False, http_client=injected)
            assert sync._init_options["http_client"] is injected
        finally:
            injected.close()


class TestAsyncHttpClientInjection:
    """Caller-supplied httpx.AsyncClient is used verbatim."""

    @pytest.mark.asyncio
    async def test_injected_async_client_is_used_by_remote(self):
        injected = httpx.AsyncClient()
        try:
            client = AsyncQdrantClient(check_compatibility=False, http_client=injected)
            assert isinstance(client._client, AsyncQdrantRemote)
            api_client = client._client.openapi_client.client
            assert isinstance(api_client, AsyncApiClient)
            assert api_client._async_client is injected
        finally:
            await injected.aclose()

    @pytest.mark.asyncio
    async def test_default_path_still_builds_its_own_async_client(self):
        client = AsyncQdrantClient(check_compatibility=False)
        api_client = client._client.openapi_client.client
        # Sanity: the default branch ran and wired up a real httpx.AsyncClient.
        assert isinstance(api_client._async_client, httpx.AsyncClient)
        await api_client._async_client.aclose()

    @pytest.mark.asyncio
    async def test_injected_async_client_keeps_rest_headers_middleware(self):
        from unittest.mock import AsyncMock

        from qdrant_client.context_headers import async_headers

        injected = httpx.AsyncClient()
        try:
            client = AsyncQdrantClient(check_compatibility=False, http_client=injected)
            api_client = client._client.openapi_client.client
            request = httpx.Request("GET", "http://localhost:6333/collections")
            async with async_headers({"x-tracing-id": "async-trace-1068"}):
                await api_client.middleware(request, AsyncMock(return_value="response"))
            assert request.headers.get("x-tracing-id") == "async-trace-1068"
        finally:
            await injected.aclose()
