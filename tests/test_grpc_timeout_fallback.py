"""Tests for the gRPC timeout-fallback fix (issue #948).

The bug: when a method on QdrantRemote / AsyncQdrantRemote is called without
a per-call `timeout`, the protobuf `timeout` field on the gRPC request was
left as `None`. The gRPC transport deadline (a separate argument) was set
from `self._timeout` correctly, but the Qdrant server uses the protobuf
field for its own internal operation budget, so a user-set global timeout
was effectively ignored.

The fix: every gRPC call site in qdrant_remote.py / async_qdrant_remote.py
now uses the same `effective_timeout = timeout if timeout is not None
else self._timeout` for both the protobuf field and the gRPC deadline.
Methods that don't take a per-call `timeout` use `self._timeout` directly.

These tests exercise the protobuf field and the gRPC deadline without
making a real network call, by monkey-patching the gRPC stub. They call
the gRPC method on the remote directly, then assert on the captured
request and deadline (any post-call processing errors are irrelevant
to the fix being tested).
"""

import inspect
from unittest.mock import MagicMock

import pytest

from qdrant_client import AsyncQdrantClient, QdrantClient


def _patch_stub_pool(remote, async_mode: bool) -> list:
    """Replace every gRPC stub method on every stub in the pool. The remote
    property round-robins across the pool, so we must patch all of them.
    Returns a list that will be populated with (method_name, request, deadline)
    tuples as calls happen."""
    # Force lazy initialization.
    _ = remote.grpc_points
    _ = remote.grpc_collections

    captured: list = []

    def make_recorder(name: str):
        if async_mode:
            async def recorder(request, timeout=None, **_):
                captured.append((name, request, timeout))
                resp = MagicMock()
                resp.result = []
                # Scroll response shape: next_page_offset and result.
                resp.next_page_offset = None
                resp.points = []
                return resp
        else:
            def recorder(request, timeout=None, **_):
                captured.append((name, request, timeout))
                resp = MagicMock()
                resp.result = []
                resp.next_page_offset = None
                resp.points = []
                return resp
        return recorder

    pools = [remote._grpc_points_client_pool, remote._grpc_collections_client_pool]
    for pool in pools:
        if pool is None:
            continue
        for stub in pool:
            for method_name in dir(stub):
                if method_name.startswith("_"):
                    continue
                attr = getattr(stub, method_name)
                if not callable(attr):
                    continue
                # Both sync and async stubs expose plain callables; the
                # async variant returns a coroutine on call. We replace
                # every callable with a recorder; in async mode the
                # recorder itself is an async coroutine function.
                if async_mode and inspect.iscoroutinefunction(attr):
                    continue  # never the case for grpc.aio stubs
                setattr(stub, method_name, make_recorder(method_name))
    return captured


# ---------------------------------------------------------------------------
# sync
# ---------------------------------------------------------------------------


class TestSyncGrpcTimeoutFallback:
    def test_query_points_propagates_global_timeout(self):
        client = QdrantClient(prefer_grpc=True, check_compatibility=False, timeout=42)
        remote = client._client
        captured = _patch_stub_pool(remote, async_mode=False)

        # Suppress post-call conversion; the test cares about the request.
        try:
            client.query_points(collection_name="test", query=[0.1, 0.2, 0.3])
        except Exception:
            pass

        method_name, request, deadline = captured[0]
        assert method_name == "Query"
        assert request.timeout == 42
        assert deadline == 42

    def test_query_points_per_call_timeout_overrides_global(self):
        client = QdrantClient(prefer_grpc=True, check_compatibility=False, timeout=42)
        remote = client._client
        captured = _patch_stub_pool(remote, async_mode=False)

        try:
            client.query_points(
                collection_name="test", query=[0.1, 0.2, 0.3], timeout=7
            )
        except Exception:
            pass

        method_name, request, deadline = captured[0]
        assert method_name == "Query"
        assert request.timeout == 7
        assert deadline == 7

    def test_query_points_default_timeout_used_when_neither_set(self):
        # When neither a per-call timeout nor a global QdrantClient timeout is
        # set, the QdrantRemote default (DEFAULT_GRPC_TIMEOUT = 5) is used
        # for both the protobuf field and the gRPC deadline.
        client = QdrantClient(prefer_grpc=True, check_compatibility=False)
        remote = client._client
        captured = _patch_stub_pool(remote, async_mode=False)

        try:
            client.query_points(collection_name="test", query=[0.1, 0.2, 0.3])
        except Exception:
            pass

        method_name, request, deadline = captured[0]
        assert method_name == "Query"
        assert request.timeout == 5
        assert deadline == 5

    def test_scroll_propagates_global_timeout(self):
        client = QdrantClient(prefer_grpc=True, check_compatibility=False, timeout=99)
        remote = client._client
        captured = _patch_stub_pool(remote, async_mode=False)

        try:
            client.scroll(collection_name="test")
        except Exception:
            pass

        # Find the Scroll call (round-robin may pick a different pool entry).
        scroll_calls = [c for c in captured if c[0] == "Scroll"]
        assert scroll_calls, f"expected Scroll call, got {[c[0] for c in captured]}"
        _method, request, deadline = scroll_calls[0]
        assert request.timeout == 99
        assert deadline == 99

    def test_count_propagates_global_timeout(self):
        client = QdrantClient(prefer_grpc=True, check_compatibility=False, timeout=11)
        remote = client._client
        captured = _patch_stub_pool(remote, async_mode=False)

        try:
            client.count(collection_name="test")
        except Exception:
            pass

        count_calls = [c for c in captured if c[0] == "Count"]
        assert count_calls, f"expected Count call, got {[c[0] for c in captured]}"
        _method, request, deadline = count_calls[0]
        assert request.timeout == 11
        assert deadline == 11

    def test_upsert_propagates_global_timeout(self):
        # Write methods originally used `timeout=self._timeout` for the gRPC
        # deadline, ignoring the per-call timeout. The fix applies the
        # fallback to both the protobuf field and the deadline.
        from qdrant_client import models

        client = QdrantClient(prefer_grpc=True, check_compatibility=False, timeout=33)
        remote = client._client
        captured = _patch_stub_pool(remote, async_mode=False)

        try:
            client.upsert(
                collection_name="test",
                points=[models.PointStruct(id=1, vector=[0.1, 0.2])],
            )
        except Exception:
            pass

        upsert_calls = [c for c in captured if c[0] == "Upsert"]
        assert upsert_calls, f"expected Upsert call, got {[c[0] for c in captured]}"
        _method, request, deadline = upsert_calls[0]
        assert request.timeout == 33
        assert deadline == 33

    def test_collection_method_uses_self_timeout_only(self):
        # collection_exists takes no per-call timeout. The local is
        # `effective_timeout = self._timeout` (not the with-fallback form).
        # CollectionExistsRequest has no protobuf `timeout` field, so we
        # verify the gRPC deadline equals the global timeout instead.
        client = QdrantClient(prefer_grpc=True, check_compatibility=False, timeout=55)
        remote = client._client
        captured = _patch_stub_pool(remote, async_mode=False)

        try:
            client.collection_exists(collection_name="test")
        except Exception:
            pass

        assert captured, "expected a gRPC call"
        deadlines = [d for _m, _r, d in captured]
        assert any(d == 55 for d in deadlines), (
            f"expected at least one gRPC deadline == 55, got {deadlines}"
        )


# ---------------------------------------------------------------------------
# async
# ---------------------------------------------------------------------------


class TestAsyncGrpcTimeoutFallback:
    @pytest.mark.asyncio
    async def test_query_points_propagates_global_timeout(self):
        client = AsyncQdrantClient(
            prefer_grpc=True, check_compatibility=False, timeout=42
        )
        remote = client._client
        captured = _patch_stub_pool(remote, async_mode=True)

        try:
            await client.query_points(collection_name="test", query=[0.1, 0.2, 0.3])
        except Exception:
            pass

        method_name, request, deadline = captured[0]
        assert method_name == "Query"
        assert request.timeout == 42
        assert deadline == 42

    @pytest.mark.asyncio
    async def test_query_points_per_call_timeout_overrides_global(self):
        client = AsyncQdrantClient(
            prefer_grpc=True, check_compatibility=False, timeout=42
        )
        remote = client._client
        captured = _patch_stub_pool(remote, async_mode=True)

        try:
            await client.query_points(
                collection_name="test", query=[0.1, 0.2, 0.3], timeout=7
            )
        except Exception:
            pass

        method_name, request, deadline = captured[0]
        assert method_name == "Query"
        assert request.timeout == 7
        assert deadline == 7

    @pytest.mark.asyncio
    async def test_scroll_propagates_global_timeout(self):
        client = AsyncQdrantClient(
            prefer_grpc=True, check_compatibility=False, timeout=99
        )
        remote = client._client
        captured = _patch_stub_pool(remote, async_mode=True)

        try:
            await client.scroll(collection_name="test")
        except Exception:
            pass

        scroll_calls = [c for c in captured if c[0] == "Scroll"]
        assert scroll_calls, f"expected Scroll call, got {[c[0] for c in captured]}"
        _method, request, deadline = scroll_calls[0]
        assert request.timeout == 99
        assert deadline == 99

    @pytest.mark.asyncio
    async def test_upsert_propagates_global_timeout(self):
        from qdrant_client import models

        client = AsyncQdrantClient(
            prefer_grpc=True, check_compatibility=False, timeout=33
        )
        remote = client._client
        captured = _patch_stub_pool(remote, async_mode=True)

        try:
            await client.upsert(
                collection_name="test",
                points=[models.PointStruct(id=1, vector=[0.1, 0.2])],
            )
        except Exception:
            pass

        upsert_calls = [c for c in captured if c[0] == "Upsert"]
        assert upsert_calls, f"expected Upsert call, got {[c[0] for c in captured]}"
        _method, request, deadline = upsert_calls[0]
        assert request.timeout == 33
        assert deadline == 33
