"""Regression tests for health_check() on the client classes.

Issue #1289. Verifies that QdrantClient, AsyncQdrantClient, QdrantRemote,
AsyncQdrantRemote, QdrantLocal, and AsyncQdrantLocal all expose a
health_check() method that returns a bool: True on success, False on any
failure. Exceptions are folded into a False return rather than raised.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from qdrant_client import QdrantClient
from qdrant_client.async_qdrant_client import AsyncQdrantClient


class TestSyncFacadeHealthCheck:
    """QdrantClient.health_check() delegates to the inner client."""

    def test_local_in_memory(self):
        client = QdrantClient(":memory:")
        assert client.health_check() is True

    def test_local_after_close(self):
        client = QdrantClient(":memory:")
        client.close()
        assert client.health_check() is False

    def test_local_path(self):
        client = QdrantClient(path="/tmp/qdrant_health_test")
        assert client.health_check() is True

    def test_remote_success(self):
        client = QdrantClient("http://localhost:6333", prefer_grpc=False)
        client._client.openapi_client.service_api.healthz = MagicMock(
            return_value="all is good"
        )
        assert client.health_check() is True
        client._client.openapi_client.service_api.healthz.assert_called_once()

    def test_remote_failure_returns_false(self):
        client = QdrantClient("http://localhost:6333", prefer_grpc=False)
        client._client.openapi_client.service_api.healthz = MagicMock(
            side_effect=ConnectionError("refused")
        )
        assert client.health_check() is False

    def test_remote_timeout_returns_false(self):
        client = QdrantClient("http://localhost:6333", prefer_grpc=False)
        client._client.openapi_client.service_api.healthz = MagicMock(
            side_effect=TimeoutError("slow")
        )
        assert client.health_check() is False


class TestAsyncFacadeHealthCheck:
    """AsyncQdrantClient.health_check() is async and must be awaited."""

    @pytest.mark.asyncio
    async def test_local_in_memory(self):
        client = AsyncQdrantClient(":memory:")
        assert await client.health_check() is True

    @pytest.mark.asyncio
    async def test_local_after_close(self):
        client = AsyncQdrantClient(":memory:")
        await client.close()
        assert await client.health_check() is False

    @pytest.mark.asyncio
    async def test_local_path(self):
        client = AsyncQdrantClient(path="/tmp/qdrant_health_test_async")
        assert await client.health_check() is True

    @pytest.mark.asyncio
    async def test_remote_success(self):
        client = AsyncQdrantClient("http://localhost:6333", prefer_grpc=False)
        client._client.http.service_api.healthz = AsyncMock(return_value="all is good")
        assert await client.health_check() is True
        client._client.http.service_api.healthz.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_remote_failure_returns_false(self):
        client = AsyncQdrantClient("http://localhost:6333", prefer_grpc=False)
        client._client.http.service_api.healthz = AsyncMock(
            side_effect=ConnectionError("refused")
        )
        assert await client.health_check() is False

    @pytest.mark.asyncio
    async def test_remote_timeout_returns_false(self):
        client = AsyncQdrantClient("http://localhost:6333", prefer_grpc=False)
        client._client.http.service_api.healthz = AsyncMock(
            side_effect=TimeoutError("slow")
        )
        assert await client.health_check() is False


class TestSyncRemoteHealthCheck:
    """QdrantRemote.health_check() calls /healthz via the openapi client."""

    def test_returns_true_on_success(self):
        from qdrant_client.qdrant_remote import QdrantRemote

        client = QdrantRemote("http://localhost:6333", prefer_grpc=False)
        client.openapi_client.service_api.healthz = MagicMock(
            return_value="all is good"
        )
        assert client.health_check() is True

    def test_returns_false_on_connection_error(self):
        from qdrant_client.qdrant_remote import QdrantRemote

        client = QdrantRemote("http://localhost:6333", prefer_grpc=False)
        client.openapi_client.service_api.healthz = MagicMock(
            side_effect=ConnectionError("refused")
        )
        assert client.health_check() is False

    def test_returns_false_on_http_error(self):
        from qdrant_client.qdrant_remote import QdrantRemote

        client = QdrantRemote("http://localhost:6333", prefer_grpc=False)
        client.openapi_client.service_api.healthz = MagicMock(
            side_effect=Exception("500 Internal Server Error")
        )
        assert client.health_check() is False


class TestAsyncRemoteHealthCheck:
    """AsyncQdrantRemote.health_check() awaits /healthz via the http client."""

    @pytest.mark.asyncio
    async def test_returns_true_on_success(self):
        from qdrant_client.async_qdrant_remote import AsyncQdrantRemote

        client = AsyncQdrantRemote("http://localhost:6333", prefer_grpc=False)
        client.http.service_api.healthz = AsyncMock(return_value="all is good")
        assert await client.health_check() is True

    @pytest.mark.asyncio
    async def test_returns_false_on_connection_error(self):
        from qdrant_client.async_qdrant_remote import AsyncQdrantRemote

        client = AsyncQdrantRemote("http://localhost:6333", prefer_grpc=False)
        client.http.service_api.healthz = AsyncMock(
            side_effect=ConnectionError("refused")
        )
        assert await client.health_check() is False


class TestLocalHealthCheck:
    """QdrantLocal.health_check() returns the not-_closed flag."""

    def test_in_memory_is_alive(self):
        from qdrant_client.local.qdrant_local import QdrantLocal

        local = QdrantLocal(":memory:")
        assert local.health_check() is True

    def test_after_close_is_dead(self):
        from qdrant_client.local.qdrant_local import QdrantLocal

        local = QdrantLocal(":memory:")
        local.close()
        assert local.health_check() is False

    def test_returns_bool_not_int(self):
        from qdrant_client.local.qdrant_local import QdrantLocal

        local = QdrantLocal(":memory:")
        result = local.health_check()
        assert isinstance(result, bool)
        assert result is True  # explicitly True, not 1


class TestAsyncLocalHealthCheck:
    """AsyncQdrantLocal.health_check() returns the not-_closed flag (sync)."""

    def test_in_memory_is_alive(self):
        from qdrant_client.local.async_qdrant_local import AsyncQdrantLocal

        local = AsyncQdrantLocal(":memory:")
        assert local.health_check() is True

    def test_after_close_is_dead(self):
        from qdrant_client.local.async_qdrant_local import AsyncQdrantLocal

        local = AsyncQdrantLocal(":memory:")
        # AsyncQdrantLocal.close is async
        import asyncio

        asyncio.run(local.close())
        assert local.health_check() is False

    def test_returns_bool_not_int(self):
        from qdrant_client.local.async_qdrant_local import AsyncQdrantLocal

        local = AsyncQdrantLocal(":memory:")
        result = local.health_check()
        assert isinstance(result, bool)
        assert result is True  # explicitly True, not 1


class TestHealthCheckNeverRaises:
    """The contract: any failure is folded into a False return, never raised."""

    def test_sync_never_raises_on_attribute_error(self):
        remote = QdrantClient("http://localhost:6333", prefer_grpc=False)
        remote._client.openapi_client.service_api.healthz = MagicMock(
            side_effect=AttributeError("broken")
        )
        # Should not raise
        assert remote.health_check() is False

    @pytest.mark.asyncio
    async def test_async_never_raises_on_runtime_error(self):
        client = AsyncQdrantClient("http://localhost:6333", prefer_grpc=False)
        client._client.http.service_api.healthz = AsyncMock(
            side_effect=RuntimeError("oops")
        )
        # Should not raise
        assert await client.health_check() is False
