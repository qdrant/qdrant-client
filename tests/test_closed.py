"""Regression tests for the `closed` property on the client facades.

Issue #1299. Verifies that QdrantClient and AsyncQdrantClient expose a
`closed` property that returns False before close() is called and True
afterwards, delegating to the inner client. Works for :memory: local
mode, path= local mode, and remote mode.
"""

import tempfile
from pathlib import Path

import pytest

from qdrant_client import QdrantClient
from qdrant_client.async_qdrant_client import AsyncQdrantClient


class TestSyncFacadeClosed:
    """QdrantClient.closed returns False before close(), True after."""

    def test_local_in_memory_starts_false(self):
        client = QdrantClient(":memory:")
        try:
            assert client.closed is False
        finally:
            client.close()

    def test_local_in_memory_true_after_close(self):
        client = QdrantClient(":memory:")
        client.close()
        assert client.closed is True

    def test_local_path_starts_false(self):
        with tempfile.TemporaryDirectory() as tmp:
            client = QdrantClient(path=str(Path(tmp) / "qdrant_test"))
            try:
                assert client.closed is False
            finally:
                client.close()

    def test_local_path_true_after_close(self):
        with tempfile.TemporaryDirectory() as tmp:
            client = QdrantClient(path=str(Path(tmp) / "qdrant_test"))
            client.close()
            assert client.closed is True

    def test_remote_starts_false(self):
        client = QdrantClient("http://localhost:6333", prefer_grpc=False)
        try:
            assert client.closed is False
        finally:
            client.close()

    def test_remote_true_after_close(self):
        client = QdrantClient("http://localhost:6333", prefer_grpc=False)
        client.close()
        assert client.closed is True

    def test_delegates_to_inner_client(self):
        # closed should be the same value the inner client exposes,
        # not a copy or a stale snapshot.
        client = QdrantClient(":memory:")
        try:
            assert client.closed == client._client.closed
            client.close()
            assert client.closed == client._client.closed
        finally:
            client.close()

    def test_double_close_keeps_closed_true(self):
        client = QdrantClient(":memory:")
        client.close()
        assert client.closed is True
        client.close()
        assert client.closed is True


class TestAsyncFacadeClosed:
    """AsyncQdrantClient.closed is a sync property, returnable without await."""

    @pytest.mark.asyncio
    async def test_local_in_memory_starts_false(self):
        client = AsyncQdrantClient(":memory:")
        try:
            assert client.closed is False
        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_local_in_memory_true_after_close(self):
        client = AsyncQdrantClient(":memory:")
        await client.close()
        assert client.closed is True

    @pytest.mark.asyncio
    async def test_local_path_starts_false(self):
        with tempfile.TemporaryDirectory() as tmp:
            client = AsyncQdrantClient(path=str(Path(tmp) / "qdrant_test_async"))
            try:
                assert client.closed is False
            finally:
                await client.close()

    @pytest.mark.asyncio
    async def test_local_path_true_after_close(self):
        with tempfile.TemporaryDirectory() as tmp:
            client = AsyncQdrantClient(path=str(Path(tmp) / "qdrant_test_async"))
            await client.close()
            assert client.closed is True

    @pytest.mark.asyncio
    async def test_remote_starts_false(self):
        client = AsyncQdrantClient("http://localhost:6333", prefer_grpc=False)
        try:
            assert client.closed is False
        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_remote_true_after_close(self):
        client = AsyncQdrantClient("http://localhost:6333", prefer_grpc=False)
        await client.close()
        assert client.closed is True

    @pytest.mark.asyncio
    async def test_delegates_to_inner_client(self):
        client = AsyncQdrantClient(":memory:")
        try:
            assert client.closed == client._client.closed
            await client.close()
            assert client.closed == client._client.closed
        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_double_close_keeps_closed_true(self):
        client = AsyncQdrantClient(":memory:")
        await client.close()
        assert client.closed is True
        await client.close()
        assert client.closed is True

    @pytest.mark.asyncio
    async def test_property_is_not_coroutine(self):
        # closed is a synchronous @property — it must not return a coroutine.
        # If a maintainer accidentally makes it async, this test catches it.
        client = AsyncQdrantClient(":memory:")
        try:
            result = client.closed
            assert not hasattr(result, "__await__")
            assert isinstance(result, bool)
        finally:
            await client.close()
