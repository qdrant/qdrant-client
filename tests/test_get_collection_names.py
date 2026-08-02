"""Regression tests for `get_collection_names()` on the client classes.

Issue #1302. Verifies that QdrantClient, AsyncQdrantClient, QdrantRemote,
AsyncQdrantRemote, QdrantLocal, and AsyncQdrantLocal all expose a
`get_collection_names()` method that returns a `list[str]` of all
collection names. For local mode, returns directly from the in-memory
dict (no `get_collections()` round-trip). For remote mode, delegates
to `get_collections()` and extracts `.name`.
"""

import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from qdrant_client import QdrantClient
from qdrant_client.async_qdrant_client import AsyncQdrantClient
from qdrant_client.http import models as rest_models


def _make_collection_desc(name: str) -> rest_models.CollectionDescription:
    return rest_models.CollectionDescription(name=name)


class TestSyncFacadeGetCollectionNames:
    """QdrantClient.get_collection_names() delegates to the inner client."""

    def test_local_in_memory_empty(self):
        client = QdrantClient(":memory:")
        try:
            assert client.get_collection_names() == []
        finally:
            client.close()

    def test_local_in_memory_after_create(self):
        client = QdrantClient(":memory:")
        try:
            from qdrant_client.http import models

            client.create_collection(
                "a", vectors_config=models.VectorParams(size=4, distance=models.Distance.COSINE)
            )
            client.create_collection(
                "b", vectors_config=models.VectorParams(size=4, distance=models.Distance.COSINE)
            )
            names = client.get_collection_names()
            assert isinstance(names, list)
            assert sorted(names) == ["a", "b"]
        finally:
            client.close()

    def test_local_in_memory_returns_list_not_model(self):
        # The whole point of this method: list[str], not CollectionsResponse.
        client = QdrantClient(":memory:")
        try:
            names = client.get_collection_names()
            assert not isinstance(names, rest_models.CollectionsResponse)
            assert all(isinstance(n, str) for n in names)
        finally:
            client.close()

    def test_local_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            client = QdrantClient(path=str(Path(tmp) / "qdrant_gcn_test"))
            try:
                assert client.get_collection_names() == []
            finally:
                client.close()

    def test_remote_rest_success(self):
        client = QdrantClient("http://localhost:6333", prefer_grpc=False)
        try:
            mock_response = MagicMock()
            mock_response.collections = [
                _make_collection_desc("a"),
                _make_collection_desc("b"),
            ]
            client._client.get_collections = MagicMock(return_value=mock_response)
            assert client.get_collection_names() == ["a", "b"]
            client._client.get_collections.assert_called_once()
        finally:
            client.close()

    def test_remote_rest_empty(self):
        client = QdrantClient("http://localhost:6333", prefer_grpc=False)
        try:
            mock_response = MagicMock()
            mock_response.collections = []
            client._client.get_collections = MagicMock(return_value=mock_response)
            assert client.get_collection_names() == []
        finally:
            client.close()


class TestAsyncFacadeGetCollectionNames:
    """AsyncQdrantClient.get_collection_names() is async; awaits the inner."""

    @pytest.mark.asyncio
    async def test_local_in_memory_empty(self):
        client = AsyncQdrantClient(":memory:")
        try:
            result = await client.get_collection_names()
            assert result == []
        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_local_in_memory_after_create(self):
        client = AsyncQdrantClient(":memory:")
        try:
            from qdrant_client.http import models

            await client.create_collection(
                "a", vectors_config=models.VectorParams(size=4, distance=models.Distance.COSINE)
            )
            await client.create_collection(
                "b", vectors_config=models.VectorParams(size=4, distance=models.Distance.COSINE)
            )
            names = await client.get_collection_names()
            assert isinstance(names, list)
            assert sorted(names) == ["a", "b"]
        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_local_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            client = AsyncQdrantClient(path=str(Path(tmp) / "qdrant_gcn_test_async"))
            try:
                result = await client.get_collection_names()
                assert result == []
            finally:
                await client.close()

    @pytest.mark.asyncio
    async def test_remote_rest_success(self):
        client = AsyncQdrantClient("http://localhost:6333", prefer_grpc=False)
        try:
            mock_response = MagicMock()
            mock_response.collections = [
                _make_collection_desc("a"),
                _make_collection_desc("b"),
            ]
            # Inner is async; get_collection_names must await the coroutine.
            client._client.get_collections = AsyncMock(return_value=mock_response)
            result = await client.get_collection_names()
            assert result == ["a", "b"]
        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_property_is_async(self):
        # The async facade's get_collection_names is a coroutine function.
        # If a maintainer accidentally makes it sync, this test catches it.
        client = AsyncQdrantClient(":memory:")
        try:
            assert hasattr(client.get_collection_names, "__await__") or hasattr(
                client.get_collection_names, "__call__"
            )
            # Sanity: the actual call returns a coroutine
            coro = client.get_collection_names()
            import inspect as _inspect

            assert _inspect.iscoroutine(coro)
            await coro
        finally:
            await client.close()


class TestLocalDirectAccess:
    """The local classes read self.collections.keys() directly (no model)."""

    def test_local_no_get_collections_call(self):
        # The local path should not round-trip through get_collections();
        # it should read self.collections directly.
        client = QdrantClient(":memory:")
        try:
            called = []
            original = client._client.get_collections

            def spy(**kwargs):
                called.append(kwargs)
                return original(**kwargs)

            client._client.get_collections = spy
            client.get_collection_names()
            assert called == []
        finally:
            client.close()

    def test_local_after_close_raises(self):
        client = QdrantClient(":memory:")
        client.close()
        import pytest

        with pytest.raises(RuntimeError, match="closed"):
            client.get_collection_names()

    @pytest.mark.asyncio
    async def test_async_local_after_close_raises(self):
        client = AsyncQdrantClient(":memory:")
        await client.close()
        with pytest.raises(RuntimeError, match="closed"):
            await client.get_collection_names()
