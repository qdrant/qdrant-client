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

    def test_init_options_round_trip_excludes_http_client(self):
        # init_options is used to round-trip a QdrantClient into an
        # AsyncQdrantClient and back. http_client is deliberately excluded:
        # an httpx.Client passed into an async client (or vice versa) would
        # mismatch at runtime, so callers must pass the http_client explicitly
        # to each side. The non-http_client args must still survive the
        # round-trip, which is what the rest of this test covers.
        injected = httpx.Client()
        try:
            sync = QdrantClient(
                check_compatibility=False,
                http_client=injected,
                timeout=42,
            )
            assert "http_client" not in sync._init_options
            assert sync._init_options["timeout"] == 42
        finally:
            injected.close()

    def test_close_does_not_close_injected_client(self):
        # ApiClient.close() must not close a caller-supplied httpx.Client;
        # the caller owns its lifecycle and may keep using it after
        # QdrantClient is closed. Direct ownership check, not behavior-by-
        # exception-swallow.
        injected = httpx.Client()
        client = QdrantClient(check_compatibility=False, http_client=injected)
        # Constructor must have flagged the injected client as not-owned.
        assert client._client.openapi_client.client._owns_client is False
        client.close()
        assert not injected.is_closed
        injected.close()

    def test_close_does_close_built_client(self):
        # Regression: the default path (no http_client) builds and owns
        # the httpx.Client, so closing the qdrant client closes the
        # underlying httpx.Client too. No monkey-patching the flag.
        client = QdrantClient(check_compatibility=False)
        api_client = client._client.openapi_client.client
        # Constructor must have flagged the built client as owned.
        assert api_client._owns_client is True
        assert not api_client._client.is_closed
        api_client.close()
        assert api_client._client.is_closed


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
        # Sanity: the default branch ran, wired up a real httpx.AsyncClient,
        # and flagged it as owned by us.
        assert isinstance(api_client._async_client, httpx.AsyncClient)
        assert api_client._owns_client is True
        # Closing the qdrant client must close the built httpx.AsyncClient.
        assert not api_client._async_client.is_closed
        await client.close()
        assert api_client._async_client.is_closed

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

    @pytest.mark.asyncio
    async def test_aclose_does_not_close_injected_client(self):
        # AsyncApiClient.aclose() must not close a caller-supplied
        # httpx.AsyncClient; the caller owns its lifecycle. Direct
        # ownership check, not behavior-by-exception-swallow.
        injected = httpx.AsyncClient()
        client = AsyncQdrantClient(check_compatibility=False, http_client=injected)
        # Constructor must have flagged the injected client as not-owned.
        assert client._client.openapi_client.client._owns_client is False
        await client.close()
        assert not injected.is_closed
        await injected.aclose()

    @pytest.mark.asyncio
    async def test_init_options_round_trip_excludes_http_client(self):
        # Mirror of the sync test: http_client is excluded from init_options
        # so an async client cannot leak into a sync client (and vice versa).
        injected = httpx.AsyncClient()
        try:
            async_client = AsyncQdrantClient(
                check_compatibility=False,
                http_client=injected,
                timeout=42,
            )
            assert "http_client" not in async_client._init_options
            assert async_client._init_options["timeout"] == 42
        finally:
            await injected.aclose()
