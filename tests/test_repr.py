"""Regression tests for __repr__ on the client classes.

Issue #1287. Verifies that QdrantClient, AsyncQdrantClient, QdrantRemote,
AsyncQdrantRemote, QdrantLocal, and AsyncQdrantLocal all produce meaningful
repr() output that includes the connection info and never leaks the api_key.
"""

import pytest

from qdrant_client import QdrantClient
from qdrant_client.async_qdrant_client import AsyncQdrantClient


class TestQdrantClientRepr:
    """QdrantClient (the sync facade)."""

    def test_local_in_memory(self):
        client = QdrantClient(":memory:")
        r = repr(client)
        assert r.startswith("<QdrantClient mode=local location=':memory:'>")

    def test_local_path(self):
        client = QdrantClient(path="/tmp/qdrant")
        r = repr(client)
        assert r.startswith("<QdrantClient mode=local location='/tmp/qdrant'>")

    def test_remote_default(self):
        client = QdrantClient("localhost", port=6333, prefer_grpc=True)
        r = repr(client)
        assert r.startswith("<QdrantClient mode=remote ")
        assert "scheme=http" in r
        assert "host='localhost'" in r
        assert "port=6333" in r
        assert "prefer_grpc=True" in r

    def test_remote_https(self):
        client = QdrantClient("https://api.qdrant.io", api_key="sk-secret-12345")
        r = repr(client)
        assert "scheme=https" in r
        assert "host='api.qdrant.io'" in r
        # api_key must NEVER appear in the repr
        assert "sk-secret-12345" not in r
        assert "api_key" not in r

    def test_remote_url_with_port(self):
        client = QdrantClient(url="https://example.com:1234")
        r = repr(client)
        assert "scheme=https" in r
        assert "host='example.com'" in r
        assert "port=1234" in r


class TestQdrantRemoteRepr:
    """QdrantRemote (the inner sync class, directly instantiated)."""

    def test_default(self):
        from qdrant_client.qdrant_remote import QdrantRemote

        client = QdrantRemote(url="https://api.qdrant.io", api_key="sk-secret-12345")
        r = repr(client)
        assert r.startswith("<QdrantRemote ")
        assert "scheme=https" in r
        assert "host='api.qdrant.io'" in r
        assert "sk-secret-12345" not in r

    def test_prefer_grpc(self):
        from qdrant_client.qdrant_remote import QdrantRemote

        client = QdrantRemote(host="localhost", port=6333, prefer_grpc=True)
        r = repr(client)
        assert "prefer_grpc=True" in r
        assert "scheme=http" in r


class TestQdrantLocalRepr:
    """QdrantLocal (the inner sync class, directly instantiated)."""

    def test_in_memory(self):
        from qdrant_client.local.qdrant_local import QdrantLocal

        local = QdrantLocal(":memory:")
        r = repr(local)
        assert r == "<QdrantLocal location=':memory:'>"

    def test_path(self):
        from qdrant_client.local.qdrant_local import QdrantLocal

        local = QdrantLocal("/tmp/qdrant_storage")
        r = repr(local)
        assert r == "<QdrantLocal location='/tmp/qdrant_storage'>"


class TestAsyncQdrantClientRepr:
    """AsyncQdrantClient (the async facade)."""

    def test_local_in_memory(self):
        client = AsyncQdrantClient(":memory:")
        r = repr(client)
        assert r.startswith("<AsyncQdrantClient mode=local location=':memory:'>")

    def test_local_path(self):
        client = AsyncQdrantClient(path="/tmp/qdrant")
        r = repr(client)
        assert r.startswith("<AsyncQdrantClient mode=local location='/tmp/qdrant'>")

    def test_remote(self):
        client = AsyncQdrantClient("localhost", port=6333, prefer_grpc=True)
        r = repr(client)
        assert r.startswith("<AsyncQdrantClient mode=remote ")
        assert "scheme=http" in r
        assert "host='localhost'" in r
        assert "port=6333" in r
        assert "prefer_grpc=True" in r

    def test_remote_https_no_api_key_leak(self):
        client = AsyncQdrantClient("https://api.qdrant.io", api_key="sk-secret-12345")
        r = repr(client)
        assert "scheme=https" in r
        assert "host='api.qdrant.io'" in r
        assert "sk-secret-12345" not in r
        assert "api_key" not in r


class TestAsyncQdrantRemoteRepr:
    """AsyncQdrantRemote (the inner async class, directly instantiated)."""

    def test_default(self):
        from qdrant_client.async_qdrant_remote import AsyncQdrantRemote

        client = AsyncQdrantRemote(
            url="https://api.qdrant.io", api_key="sk-secret-12345"
        )
        r = repr(client)
        assert r.startswith("<AsyncQdrantRemote ")
        assert "scheme=https" in r
        assert "host='api.qdrant.io'" in r
        assert "sk-secret-12345" not in r


class TestAsyncQdrantLocalRepr:
    """AsyncQdrantLocal (the inner async class, directly instantiated)."""

    def test_in_memory(self):
        from qdrant_client.local.async_qdrant_local import AsyncQdrantLocal

        local = AsyncQdrantLocal(":memory:")
        r = repr(local)
        assert r == "<AsyncQdrantLocal location=':memory:'>"

    def test_path(self):
        from qdrant_client.local.async_qdrant_local import AsyncQdrantLocal

        local = AsyncQdrantLocal("/tmp/qdrant_storage")
        r = repr(local)
        assert r == "<AsyncQdrantLocal location='/tmp/qdrant_storage'>"


class TestReprIsStr:
    """repr() must return a str (never a coroutine). It is called synchronously
    by Python's built-in formatter, so the async client's __repr__ must be a
    regular def, not an async def (which would return a coroutine)."""

    def test_repr_returns_str_not_coroutine_sync(self):
        client = QdrantClient(":memory:")
        r = repr(client)
        assert isinstance(r, str)
        assert "QdrantClient" in r

    @pytest.mark.asyncio
    async def test_repr_returns_str_not_coroutine_async(self):
        client = AsyncQdrantClient(":memory:")
        r = repr(client)
        assert isinstance(r, str)
        assert "AsyncQdrantClient" in r
