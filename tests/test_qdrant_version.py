"""Regression tests for qdrant_version() on the client classes.

Issue #1294. Verifies that QdrantClient, AsyncQdrantClient, QdrantRemote,
AsyncQdrantRemote, QdrantLocal, and AsyncQdrantLocal all expose a
qdrant_version() method that returns a str: the server version for
remote mode, the client library version for local mode. Returns
"<unknown>" on any failure rather than raising.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from qdrant_client import QdrantClient
from qdrant_client.async_qdrant_client import AsyncQdrantClient


class TestSyncFacadeQdrantVersion:
    """QdrantClient.qdrant_version() delegates to the inner client."""

    def test_local_in_memory(self):
        client = QdrantClient(":memory:")
        # Local mode returns the client library version, not a server version.
        result = client.qdrant_version()
        assert isinstance(result, str)
        assert result != "<unknown>"
        # The result should look like a version (e.g. "1.18.0" or "1.18.1-dev")
        assert any(c.isdigit() for c in result)

    def test_local_path(self):
        client = QdrantClient(path="/tmp/qdrant_version_test")
        result = client.qdrant_version()
        assert isinstance(result, str)
        assert result != "<unknown>"

    def test_remote_success(self):
        client = QdrantClient("http://localhost:6333", prefer_grpc=False)
        mock_root = MagicMock()
        mock_root.version = "1.10.0"
        client._client.openapi_client.service_api.root = MagicMock(
            return_value=mock_root
        )
        assert client.qdrant_version() == "1.10.0"
        client._client.openapi_client.service_api.root.assert_called_once()

    def test_remote_failure_returns_unknown(self):
        client = QdrantClient("http://localhost:6333", prefer_grpc=False)
        client._client.openapi_client.service_api.root = MagicMock(
            side_effect=ConnectionError("refused")
        )
        assert client.qdrant_version() == "<unknown>"

    def test_remote_timeout_returns_unknown(self):
        client = QdrantClient("http://localhost:6333", prefer_grpc=False)
        client._client.openapi_client.service_api.root = MagicMock(
            side_effect=TimeoutError("slow")
        )
        assert client.qdrant_version() == "<unknown>"

    def test_remote_api_error_returns_unknown(self):
        client = QdrantClient("http://localhost:6333", prefer_grpc=False)
        client._client.openapi_client.service_api.root = MagicMock(
            side_effect=Exception("500 Internal Server Error")
        )
        assert client.qdrant_version() == "<unknown>"


class TestAsyncFacadeQdrantVersion:
    """AsyncQdrantClient.qdrant_version() is async and must be awaited."""

    @pytest.mark.asyncio
    async def test_local_in_memory(self):
        client = AsyncQdrantClient(":memory:")
        result = await client.qdrant_version()
        assert isinstance(result, str)
        assert result != "<unknown>"

    @pytest.mark.asyncio
    async def test_local_path(self):
        client = AsyncQdrantClient(path="/tmp/qdrant_version_test_async")
        result = await client.qdrant_version()
        assert isinstance(result, str)
        assert result != "<unknown>"

    @pytest.mark.asyncio
    async def test_remote_success(self):
        client = AsyncQdrantClient("http://localhost:6333", prefer_grpc=False)
        mock_root = MagicMock()
        mock_root.version = "1.10.0"
        client._client.http.service_api.root = AsyncMock(return_value=mock_root)
        assert await client.qdrant_version() == "1.10.0"
        client._client.http.service_api.root.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_remote_failure_returns_unknown(self):
        client = AsyncQdrantClient("http://localhost:6333", prefer_grpc=False)
        client._client.http.service_api.root = AsyncMock(
            side_effect=ConnectionError("refused")
        )
        assert await client.qdrant_version() == "<unknown>"

    @pytest.mark.asyncio
    async def test_remote_timeout_returns_unknown(self):
        client = AsyncQdrantClient("http://localhost:6333", prefer_grpc=False)
        client._client.http.service_api.root = AsyncMock(
            side_effect=TimeoutError("slow")
        )
        assert await client.qdrant_version() == "<unknown>"


class TestSyncRemoteQdrantVersion:
    """QdrantRemote.qdrant_version() calls service_api.root() and returns the version field."""

    def test_returns_version_field(self):
        from qdrant_client.qdrant_remote import QdrantRemote

        client = QdrantRemote("http://localhost:6333", prefer_grpc=False)
        mock_root = MagicMock()
        mock_root.version = "1.10.0"
        client.openapi_client.service_api.root = MagicMock(return_value=mock_root)
        assert client.qdrant_version() == "1.10.0"

    def test_returns_unknown_on_connection_error(self):
        from qdrant_client.qdrant_remote import QdrantRemote

        client = QdrantRemote("http://localhost:6333", prefer_grpc=False)
        client.openapi_client.service_api.root = MagicMock(
            side_effect=ConnectionError("refused")
        )
        assert client.qdrant_version() == "<unknown>"

    def test_returns_unknown_on_attribute_error(self):
        from qdrant_client.qdrant_remote import QdrantRemote

        client = QdrantRemote("http://localhost:6333", prefer_grpc=False)
        # The root() call might raise AttributeError if the API client
        # is misconfigured.
        client.openapi_client.service_api.root = MagicMock(
            side_effect=AttributeError("missing attr")
        )
        assert client.qdrant_version() == "<unknown>"


class TestAsyncRemoteQdrantVersion:
    """AsyncQdrantRemote.qdrant_version() awaits service_api.root() and returns the version field."""

    @pytest.mark.asyncio
    async def test_returns_version_field(self):
        from qdrant_client.async_qdrant_remote import AsyncQdrantRemote

        client = AsyncQdrantRemote("http://localhost:6333", prefer_grpc=False)
        mock_root = MagicMock()
        mock_root.version = "1.10.0"
        client.http.service_api.root = AsyncMock(return_value=mock_root)
        assert await client.qdrant_version() == "1.10.0"

    @pytest.mark.asyncio
    async def test_returns_unknown_on_connection_error(self):
        from qdrant_client.async_qdrant_remote import AsyncQdrantRemote

        client = AsyncQdrantRemote("http://localhost:6333", prefer_grpc=False)
        client.http.service_api.root = AsyncMock(side_effect=ConnectionError("refused"))
        assert await client.qdrant_version() == "<unknown>"


class TestLocalQdrantVersion:
    """QdrantLocal.qdrant_version() returns the client library version."""

    def test_returns_string(self):
        from qdrant_client.local.qdrant_local import QdrantLocal

        local = QdrantLocal(":memory:")
        result = local.qdrant_version()
        assert isinstance(result, str)
        assert result != "<unknown>"

    def test_does_not_raise(self):
        from qdrant_client.local.qdrant_local import QdrantLocal

        local = QdrantLocal(":memory:")
        # Just calling it should not raise
        result = local.qdrant_version()
        assert isinstance(result, str)


class TestAsyncLocalQdrantVersion:
    """AsyncQdrantLocal.qdrant_version() returns the client library version (sync, no I/O)."""

    def test_returns_string(self):
        from qdrant_client.local.async_qdrant_local import AsyncQdrantLocal

        local = AsyncQdrantLocal(":memory:")
        result = local.qdrant_version()
        assert isinstance(result, str)
        assert result != "<unknown>"

    def test_does_not_raise(self):
        from qdrant_client.local.async_qdrant_local import AsyncQdrantLocal

        local = AsyncQdrantLocal(":memory:")
        result = local.qdrant_version()
        assert isinstance(result, str)


class TestQdrantVersionNeverRaises:
    """The contract: any failure is folded into "<unknown>", never raised."""

    def test_sync_never_raises(self):
        remote = QdrantClient("http://localhost:6333", prefer_grpc=False)
        remote._client.openapi_client.service_api.root = MagicMock(
            side_effect=RuntimeError("oops")
        )
        # Should not raise
        assert remote.qdrant_version() == "<unknown>"

    @pytest.mark.asyncio
    async def test_async_never_raises(self):
        client = AsyncQdrantClient("http://localhost:6333", prefer_grpc=False)
        client._client.http.service_api.root = AsyncMock(
            side_effect=RuntimeError("oops")
        )
        # Should not raise
        assert await client.qdrant_version() == "<unknown>"


class TestQdrantVersionStringType:
    """qdrant_version() must always return a str."""

    def test_local_returns_str(self):
        client = QdrantClient(":memory:")
        result = client.qdrant_version()
        assert isinstance(result, str)
        assert not isinstance(result, bytes)

    def test_remote_unknown_returns_str(self):
        client = QdrantClient("http://localhost:6333", prefer_grpc=False)
        client._client.openapi_client.service_api.root = MagicMock(
            side_effect=Exception("broken")
        )
        result = client.qdrant_version()
        assert isinstance(result, str)
        assert result == "<unknown>"
