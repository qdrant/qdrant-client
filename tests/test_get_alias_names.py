"""Regression tests for `get_alias_names()` on the client classes.

Issue #1306. Verifies that QdrantClient, AsyncQdrantClient, QdrantRemote,
AsyncQdrantRemote, QdrantLocal, and AsyncQdrantLocal all expose a
`get_alias_names()` method that returns a `list[str]` of all alias
names. For local mode, returns directly from the in-memory dict (no
`get_aliases()` round-trip). For remote mode, delegates to
`get_aliases()` and extracts `.alias_name`.
"""

import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from qdrant_client import QdrantClient
from qdrant_client.async_qdrant_client import AsyncQdrantClient
from qdrant_client.http import models as rest_models


def _make_alias_desc(name: str) -> rest_models.AliasDescription:
    return rest_models.AliasDescription(alias_name=name, collection_name="dummy")


class TestSyncFacadeGetAliasNames:
    """QdrantClient.get_alias_names() delegates to the inner client."""

    def test_local_in_memory_empty(self):
        client = QdrantClient(":memory:")
        try:
            assert client.get_alias_names() == []
        finally:
            client.close()

    def test_local_in_memory_after_create_alias(self):
        client = QdrantClient(":memory:")
        try:
            from qdrant_client.http import models

            client.create_collection(
                "a", vectors_config=models.VectorParams(size=4, distance=models.Distance.COSINE)
            )
            client.create_collection(
                "b", vectors_config=models.VectorParams(size=4, distance=models.Distance.COSINE)
            )
            client.update_collection_aliases(
                change_aliases_operations=[
                    models.CreateAliasOperation(
                        create_alias=models.CreateAlias(alias_name="alias_a", collection_name="a")
                    ),
                    models.CreateAliasOperation(
                        create_alias=models.CreateAlias(alias_name="alias_b", collection_name="b")
                    ),
                ]
            )
            names = client.get_alias_names()
            assert isinstance(names, list)
            assert sorted(names) == ["alias_a", "alias_b"]
        finally:
            client.close()

    def test_local_in_memory_returns_list_not_model(self):
        client = QdrantClient(":memory:")
        try:
            names = client.get_alias_names()
            assert not isinstance(names, rest_models.CollectionsAliasesResponse)
            assert all(isinstance(n, str) for n in names)
        finally:
            client.close()

    def test_local_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            client = QdrantClient(path=str(Path(tmp) / "qdrant_gan_test"))
            try:
                assert client.get_alias_names() == []
            finally:
                client.close()

    def test_remote_rest_success(self):
        client = QdrantClient("http://localhost:6333", prefer_grpc=False)
        try:
            mock_response = MagicMock()
            mock_response.aliases = [
                _make_alias_desc("alias_a"),
                _make_alias_desc("alias_b"),
            ]
            client._client.get_aliases = MagicMock(return_value=mock_response)
            assert client.get_alias_names() == ["alias_a", "alias_b"]
            client._client.get_aliases.assert_called_once()
        finally:
            client.close()

    def test_remote_rest_empty(self):
        client = QdrantClient("http://localhost:6333", prefer_grpc=False)
        try:
            mock_response = MagicMock()
            mock_response.aliases = []
            client._client.get_aliases = MagicMock(return_value=mock_response)
            assert client.get_alias_names() == []
        finally:
            client.close()


class TestAsyncFacadeGetAliasNames:
    """AsyncQdrantClient.get_alias_names() is async; awaits the inner."""

    @pytest.mark.asyncio
    async def test_local_in_memory_empty(self):
        client = AsyncQdrantClient(":memory:")
        try:
            result = await client.get_alias_names()
            assert result == []
        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_local_in_memory_after_create_alias(self):
        client = AsyncQdrantClient(":memory:")
        try:
            from qdrant_client.http import models

            await client.create_collection(
                "a", vectors_config=models.VectorParams(size=4, distance=models.Distance.COSINE)
            )
            await client.create_collection(
                "b", vectors_config=models.VectorParams(size=4, distance=models.Distance.COSINE)
            )
            await client.update_collection_aliases(
                change_aliases_operations=[
                    models.CreateAliasOperation(
                        create_alias=models.CreateAlias(alias_name="alias_a", collection_name="a")
                    ),
                    models.CreateAliasOperation(
                        create_alias=models.CreateAlias(alias_name="alias_b", collection_name="b")
                    ),
                ]
            )
            names = await client.get_alias_names()
            assert isinstance(names, list)
            assert sorted(names) == ["alias_a", "alias_b"]
        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_local_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            client = AsyncQdrantClient(path=str(Path(tmp) / "qdrant_gan_test_async"))
            try:
                result = await client.get_alias_names()
                assert result == []
            finally:
                await client.close()

    @pytest.mark.asyncio
    async def test_remote_rest_success(self):
        client = AsyncQdrantClient("http://localhost:6333", prefer_grpc=False)
        try:
            mock_response = MagicMock()
            mock_response.aliases = [
                _make_alias_desc("alias_a"),
                _make_alias_desc("alias_b"),
            ]
            client._client.get_aliases = AsyncMock(return_value=mock_response)
            result = await client.get_alias_names()
            assert result == ["alias_a", "alias_b"]
        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_property_is_async(self):
        client = AsyncQdrantClient(":memory:")
        try:
            coro = client.get_alias_names()
            import inspect as _inspect

            assert _inspect.iscoroutine(coro)
            await coro
        finally:
            await client.close()


class TestLocalDirectAccess:
    """The local classes read self.aliases.keys() directly (no model)."""

    def test_local_no_get_aliases_call(self):
        client = QdrantClient(":memory:")
        try:
            called = []
            original = client._client.get_aliases

            def spy(**kwargs):
                called.append(kwargs)
                return original(**kwargs)

            client._client.get_aliases = spy
            client.get_alias_names()
            assert called == []
        finally:
            client.close()

    def test_local_after_close_raises(self):
        client = QdrantClient(":memory:")
        client.close()
        import pytest

        with pytest.raises(RuntimeError, match="closed"):
            client.get_alias_names()

    @pytest.mark.asyncio
    async def test_async_local_after_close_raises(self):
        client = AsyncQdrantClient(":memory:")
        await client.close()
        with pytest.raises(RuntimeError, match="closed"):
            await client.get_alias_names()
