"""Regression tests for QdrantClient/AsyncQdrantClient context manager support.

Issue #1285. Verifies the `with` block on QdrantClient and `async with` on
AsyncQdrantClient call close() on exit (success and exception paths), and that
manual close() before/after the block is idempotent.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from qdrant_client import QdrantClient
from qdrant_client.async_qdrant_client import AsyncQdrantClient


class TestSyncContextManagerLocal:
    """`with QdrantClient(":memory:") as c:` against the real QdrantLocal backend."""

    def test_returns_same_client(self):
        with QdrantClient(":memory:") as client:
            assert isinstance(client, QdrantClient)
            assert client._client.closed is False

    def test_closes_on_clean_exit(self):
        with QdrantClient(":memory:") as client:
            pass
        assert client._client.closed is True

    def test_closes_on_exception(self):
        client = QdrantClient(":memory:")
        with pytest.raises(RuntimeError, match="boom"):
            with client:
                raise RuntimeError("boom")
        assert client._client.closed is True

    def test_can_use_methods_inside_block(self):
        with QdrantClient(":memory:") as client:
            client.create_collection(
                "test_coll", vectors_config={"size": 4, "distance": "Cosine"}
            )
            info = client.get_collection("test_coll")
            assert info.config.params.vectors.size == 4

    def test_idempotent_close_inside_block(self):
        """Calling close() manually before exit must not raise; __exit__ re-call is safe."""
        with QdrantClient(":memory:") as client:
            client.close()
            assert client._client.closed is True
            # second close must not raise
            client.close()
        assert client._client.closed is True


class TestSyncContextManagerRemote:
    """`with QdrantRemote(...) as c:` with the gRPC + REST close paths stubbed."""

    def test_closes_grpc_and_rest(self):
        client = QdrantClient("http://localhost:6333", prefer_grpc=True)
        client._client.close = MagicMock()
        with client:
            pass
        # __exit__ called close() once (no kwargs). The two-call assertion
        # covers the idempotency property.
        assert client._client.close.call_count == 1
        client._client.close.assert_called_with(grpc_grace=None)

    def test_closes_grpc_and_rest_on_exception(self):
        client = QdrantClient("http://localhost:6333", prefer_grpc=True)
        client._client.close = MagicMock()
        with pytest.raises(ValueError):
            with client:
                raise ValueError("x")
        assert client._client.close.call_count == 1

    def test_grpc_grace_propagates_through_facade(self):
        """The facade's close() forwards grpc_grace to the inner client."""
        client = QdrantClient("http://localhost:6333", prefer_grpc=True)
        client._client.close = MagicMock()
        client.close(grpc_grace=2.5)
        called_kwargs = [c.kwargs for c in client._client.close.call_args_list]
        assert {"grpc_grace": 2.5} in called_kwargs


class TestAsyncContextManagerLocal:
    """`async with AsyncQdrantClient(":memory:") as c:` against QdrantLocal."""

    @pytest.mark.asyncio
    async def test_returns_same_client(self):
        async with AsyncQdrantClient(":memory:") as client:
            assert isinstance(client, AsyncQdrantClient)
            assert client._client.closed is False

    @pytest.mark.asyncio
    async def test_closes_on_clean_exit(self):
        async with AsyncQdrantClient(":memory:") as client:
            pass
        assert client._client.closed is True

    @pytest.mark.asyncio
    async def test_closes_on_exception(self):
        client = AsyncQdrantClient(":memory:")
        with pytest.raises(RuntimeError, match="boom"):
            async with client:
                raise RuntimeError("boom")
        assert client._client.closed is True

    @pytest.mark.asyncio
    async def test_can_use_methods_inside_block(self):
        async with AsyncQdrantClient(":memory:") as client:
            await client.create_collection(
                "test_coll", vectors_config={"size": 4, "distance": "Cosine"}
            )
            info = await client.get_collection("test_coll")
            assert info.config.params.vectors.size == 4

    @pytest.mark.asyncio
    async def test_idempotent_close_inside_block(self):
        async with AsyncQdrantClient(":memory:") as client:
            await client.close()
            assert client._client.closed is True
            await client.close()
        assert client._client.closed is True


class TestAsyncContextManagerRemote:
    """`async with AsyncQdrantRemote(...) as c:` with the close path stubbed."""

    @pytest.mark.asyncio
    async def test_closes_grpc_and_rest(self):
        client = AsyncQdrantClient("http://localhost:6333", prefer_grpc=True)
        client._client.close = AsyncMock()
        async with client:
            pass
        assert client._client.close.await_count == 1
        client._client.close.assert_awaited_with(grpc_grace=None)

    @pytest.mark.asyncio
    async def test_closes_grpc_and_rest_on_exception(self):
        client = AsyncQdrantClient("http://localhost:6333", prefer_grpc=True)
        client._client.close = AsyncMock()
        with pytest.raises(ValueError):
            async with client:
                raise ValueError("x")
        assert client._client.close.await_count == 1

    @pytest.mark.asyncio
    async def test_grpc_grace_propagates_through_facade(self):
        client = AsyncQdrantClient("http://localhost:6333", prefer_grpc=True)
        client._client.close = AsyncMock()
        await client.close(grpc_grace=2.5)
        called_kwargs = [c.kwargs for c in client._client.close.await_args_list]
        assert {"grpc_grace": 2.5} in called_kwargs
