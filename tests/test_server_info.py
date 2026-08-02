"""Regression tests for server_info() on the client classes.

Issue #1296. Verifies that QdrantClient, AsyncQdrantClient, QdrantRemote,
AsyncQdrantRemote, QdrantLocal, and AsyncQdrantLocal all expose a
server_info() method that returns either a VersionInfo model (on
success) or None (on any failure). Local mode returns a synthetic
VersionInfo with title="qdrant-client (local mode)".
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from qdrant_client import QdrantClient
from qdrant_client.async_qdrant_client import AsyncQdrantClient


class TestSyncFacadeServerInfo:
    """QdrantClient.server_info() delegates to the inner client."""

    def test_local_in_memory_returns_synthetic(self):
        client = QdrantClient(":memory:")
        result = client.server_info()
        assert result is not None
        assert result.title == "qdrant-client (local mode)"
        assert isinstance(result.version, str)
        assert result.commit is None

    def test_local_path_returns_synthetic(self):
        client = QdrantClient(path="/tmp/qdrant_server_info_test")
        result = client.server_info()
        assert result is not None
        assert result.title == "qdrant-client (local mode)"

    def test_remote_success(self):
        client = QdrantClient("http://localhost:6333", prefer_grpc=False)
        mock_root = MagicMock()
        mock_root.title = "qdrant"
        mock_root.version = "1.10.0"
        mock_root.commit = "a1b2c3d4"
        client._client.openapi_client.service_api.root = MagicMock(
            return_value=mock_root
        )
        result = client.server_info()
        assert result is mock_root

    def test_remote_failure_returns_none(self):
        client = QdrantClient("http://localhost:6333", prefer_grpc=False)
        client._client.openapi_client.service_api.root = MagicMock(
            side_effect=ConnectionError("refused")
        )
        assert client.server_info() is None

    def test_remote_timeout_returns_none(self):
        client = QdrantClient("http://localhost:6333", prefer_grpc=False)
        client._client.openapi_client.service_api.root = MagicMock(
            side_effect=TimeoutError("slow")
        )
        assert client.server_info() is None


class TestAsyncFacadeServerInfo:
    """AsyncQdrantClient.server_info() is async and must be awaited."""

    @pytest.mark.asyncio
    async def test_local_in_memory_returns_synthetic(self):
        client = AsyncQdrantClient(":memory:")
        result = await client.server_info()
        assert result is not None
        assert result.title == "qdrant-client (local mode)"

    @pytest.mark.asyncio
    async def test_local_path_returns_synthetic(self):
        client = AsyncQdrantClient(path="/tmp/qdrant_server_info_test_async")
        result = await client.server_info()
        assert result is not None
        assert result.title == "qdrant-client (local mode)"

    @pytest.mark.asyncio
    async def test_remote_success(self):
        client = AsyncQdrantClient("http://localhost:6333", prefer_grpc=False)
        mock_root = MagicMock()
        mock_root.title = "qdrant"
        mock_root.version = "1.10.0"
        mock_root.commit = "a1b2c3d4"
        client._client.http.service_api.root = AsyncMock(return_value=mock_root)
        result = await client.server_info()
        assert result is mock_root

    @pytest.mark.asyncio
    async def test_remote_failure_returns_none(self):
        client = AsyncQdrantClient("http://localhost:6333", prefer_grpc=False)
        client._client.http.service_api.root = AsyncMock(
            side_effect=ConnectionError("refused")
        )
        assert await client.server_info() is None


class TestSyncRemoteServerInfo:
    """QdrantRemote.server_info() calls service_api.root() and returns the VersionInfo."""

    def test_returns_version_info(self):
        from qdrant_client.qdrant_remote import QdrantRemote

        client = QdrantRemote("http://localhost:6333", prefer_grpc=False)
        mock_root = MagicMock()
        mock_root.title = "qdrant"
        mock_root.version = "1.10.0"
        mock_root.commit = "a1b2c3d4"
        client.openapi_client.service_api.root = MagicMock(return_value=mock_root)
        assert client.server_info() is mock_root

    def test_returns_none_on_connection_error(self):
        from qdrant_client.qdrant_remote import QdrantRemote

        client = QdrantRemote("http://localhost:6333", prefer_grpc=False)
        client.openapi_client.service_api.root = MagicMock(
            side_effect=ConnectionError("refused")
        )
        assert client.server_info() is None

    def test_returns_none_on_attribute_error(self):
        from qdrant_client.qdrant_remote import QdrantRemote

        client = QdrantRemote("http://localhost:6333", prefer_grpc=False)
        client.openapi_client.service_api.root = MagicMock(
            side_effect=AttributeError("missing attr")
        )
        assert client.server_info() is None


class TestAsyncRemoteServerInfo:
    """AsyncQdrantRemote.server_info() awaits service_api.root() and returns the VersionInfo."""

    @pytest.mark.asyncio
    async def test_returns_version_info(self):
        from qdrant_client.async_qdrant_remote import AsyncQdrantRemote

        client = AsyncQdrantRemote("http://localhost:6333", prefer_grpc=False)
        mock_root = MagicMock()
        mock_root.title = "qdrant"
        mock_root.version = "1.10.0"
        mock_root.commit = "a1b2c3d4"
        client.http.service_api.root = AsyncMock(return_value=mock_root)
        assert await client.server_info() is mock_root

    @pytest.mark.asyncio
    async def test_returns_none_on_connection_error(self):
        from qdrant_client.async_qdrant_remote import AsyncQdrantRemote

        client = AsyncQdrantRemote("http://localhost:6333", prefer_grpc=False)
        client.http.service_api.root = AsyncMock(side_effect=ConnectionError("refused"))
        assert await client.server_info() is None


class TestLocalServerInfo:
    """QdrantLocal.server_info() returns a synthetic VersionInfo."""

    def test_returns_synthetic_version_info(self):
        from qdrant_client.local.qdrant_local import QdrantLocal

        local = QdrantLocal(":memory:")
        result = local.server_info()
        assert result is not None
        assert result.title == "qdrant-client (local mode)"
        assert isinstance(result.version, str)
        assert result.commit is None

    def test_never_returns_none(self):
        from qdrant_client.local.qdrant_local import QdrantLocal

        local = QdrantLocal(":memory:")
        # Local mode never fails
        result = local.server_info()
        assert result is not None


class TestAsyncLocalServerInfo:
    """AsyncQdrantLocal.server_info() returns a synthetic VersionInfo (sync, no I/O)."""

    def test_returns_synthetic_version_info(self):
        from qdrant_client.local.async_qdrant_local import AsyncQdrantLocal

        local = AsyncQdrantLocal(":memory:")
        result = local.server_info()
        assert result is not None
        assert result.title == "qdrant-client (local mode)"

    def test_never_returns_none(self):
        from qdrant_client.local.async_qdrant_local import AsyncQdrantLocal

        local = AsyncQdrantLocal(":memory:")
        result = local.server_info()
        assert result is not None


class TestServerInfoNeverRaises:
    """The contract: any remote failure is folded into None, never raised."""

    def test_sync_never_raises(self):
        remote = QdrantClient("http://localhost:6333", prefer_grpc=False)
        remote._client.openapi_client.service_api.root = MagicMock(
            side_effect=RuntimeError("oops")
        )
        # Should not raise
        assert remote.server_info() is None

    @pytest.mark.asyncio
    async def test_async_never_raises(self):
        client = AsyncQdrantClient("http://localhost:6333", prefer_grpc=False)
        client._client.http.service_api.root = AsyncMock(
            side_effect=RuntimeError("oops")
        )
        # Should not raise
        assert await client.server_info() is None


class TestServerInfoTypeContract:
    """server_info() returns a VersionInfo model on success, None on failure."""

    def test_local_returns_version_info(self):
        client = QdrantClient(":memory:")
        result = client.server_info()
        # Should be a VersionInfo (or subclass), not None, not a dict
        assert result is not None
        assert hasattr(result, "title")
        assert hasattr(result, "version")
        assert hasattr(result, "commit")

    def test_remote_success_returns_version_info(self):
        client = QdrantClient("http://localhost:6333", prefer_grpc=False)
        mock_root = MagicMock()
        mock_root.title = "qdrant"
        mock_root.version = "1.10.0"
        mock_root.commit = "a1b2c3d4"
        client._client.openapi_client.service_api.root = MagicMock(
            return_value=mock_root
        )
        result = client.server_info()
        assert hasattr(result, "title")
        assert hasattr(result, "version")
        assert hasattr(result, "commit")
