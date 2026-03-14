import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest
from httpx import Request

from qdrant_client.context_headers import (
    async_headers,
    async_rest_headers_middleware,
    get_context_headers,
    headers,
    rest_headers_middleware,
)


class TestContextHeaders:
    def test_default_is_empty(self):
        assert get_context_headers() == {}

    def test_sync_headers_sets_and_resets(self):
        assert get_context_headers() == {}
        with headers({"x-tracing-id": "trace-1"}):
            assert get_context_headers() == {"x-tracing-id": "trace-1"}
        assert get_context_headers() == {}

    def test_multiple_headers(self):
        with headers({"x-tracing-id": "trace-1", "x-request-id": "req-1"}):
            h = get_context_headers()
            assert h["x-tracing-id"] == "trace-1"
            assert h["x-request-id"] == "req-1"
        assert get_context_headers() == {}

    def test_nested_sync_headers(self):
        with headers({"x-tracing-id": "outer"}):
            assert get_context_headers() == {"x-tracing-id": "outer"}
            with headers({"x-tracing-id": "inner"}):
                assert get_context_headers()["x-tracing-id"] == "inner"
            assert get_context_headers()["x-tracing-id"] == "outer"
        assert get_context_headers() == {}

    def test_nested_headers_merge(self):
        with headers({"x-tracing-id": "trace-1"}):
            with headers({"x-request-id": "req-1"}):
                h = get_context_headers()
                assert h["x-tracing-id"] == "trace-1"
                assert h["x-request-id"] == "req-1"
            assert get_context_headers() == {"x-tracing-id": "trace-1"}

    def test_sync_headers_resets_on_exception(self):
        with pytest.raises(RuntimeError):
            with headers({"x-tracing-id": "error-trace"}):
                assert get_context_headers()["x-tracing-id"] == "error-trace"
                raise RuntimeError("test")
        assert get_context_headers() == {}

    @pytest.mark.asyncio
    async def test_async_headers_sets_and_resets(self):
        assert get_context_headers() == {}
        async with async_headers({"x-tracing-id": "async-trace-1"}):
            assert get_context_headers()["x-tracing-id"] == "async-trace-1"
        assert get_context_headers() == {}

    @pytest.mark.asyncio
    async def test_nested_async_headers(self):
        async with async_headers({"x-tracing-id": "outer"}):
            assert get_context_headers()["x-tracing-id"] == "outer"
            async with async_headers({"x-tracing-id": "inner"}):
                assert get_context_headers()["x-tracing-id"] == "inner"
            assert get_context_headers()["x-tracing-id"] == "outer"
        assert get_context_headers() == {}

    @pytest.mark.asyncio
    async def test_async_headers_resets_on_exception(self):
        with pytest.raises(RuntimeError):
            async with async_headers({"x-tracing-id": "error-trace"}):
                assert get_context_headers()["x-tracing-id"] == "error-trace"
                raise RuntimeError("test")
        assert get_context_headers() == {}

    @pytest.mark.asyncio
    async def test_async_tasks_get_isolated_context(self):
        results = {}

        async def task(name: str, trace_id: str):
            async with async_headers({"x-tracing-id": trace_id}):
                await asyncio.sleep(0.01)
                results[name] = get_context_headers()["x-tracing-id"]

        await asyncio.gather(
            task("a", "trace-a"),
            task("b", "trace-b"),
        )
        assert results["a"] == "trace-a"
        assert results["b"] == "trace-b"
        assert get_context_headers() == {}


class TestHttpMiddleware:
    def test_sync_middleware_adds_headers(self):
        request = Request("GET", "http://localhost:6333/collections")
        call_next = MagicMock(return_value="response")

        # Without context - no extra headers
        result = rest_headers_middleware(request, call_next)
        assert "x-tracing-id" not in request.headers
        assert result == "response"

        # With context - headers added
        request2 = Request("GET", "http://localhost:6333/collections")
        with headers({"x-tracing-id": "test-123", "x-request-id": "req-456"}):
            rest_headers_middleware(request2, call_next)
            assert request2.headers["x-tracing-id"] == "test-123"
            assert request2.headers["x-request-id"] == "req-456"

    @pytest.mark.asyncio
    async def test_async_middleware_adds_headers(self):
        request = Request("GET", "http://localhost:6333/collections")
        call_next = AsyncMock(return_value="response")

        # Without context - no extra headers
        await async_rest_headers_middleware(request, call_next)
        assert "x-tracing-id" not in request.headers

        # With context - headers added
        request2 = Request("GET", "http://localhost:6333/collections")
        async with async_headers({"x-tracing-id": "async-123"}):
            await async_rest_headers_middleware(request2, call_next)
            assert request2.headers["x-tracing-id"] == "async-123"


class TestGrpcInterceptor:
    def test_sync_grpc_context_headers(self):
        from qdrant_client.connection import header_adder_interceptor

        interceptor = header_adder_interceptor(new_metadata=[("api-key", "test")])
        intercept_fn = interceptor._fn

        class FakeCallDetails:
            method = "/qdrant.Points/Search"
            timeout = 5
            metadata = None
            credentials = None

        details = FakeCallDetails()

        # Without context headers
        new_details, _, _ = intercept_fn(details, iter(["request"]), False, False)
        metadata_dict = dict(new_details.metadata)
        assert "x-tracing-id" not in metadata_dict
        assert metadata_dict["api-key"] == "test"

        # With context headers
        with headers({"x-tracing-id": "grpc-trace-123", "x-custom": "value"}):
            new_details, _, _ = intercept_fn(details, iter(["request"]), False, False)
            metadata_dict = dict(new_details.metadata)
            assert metadata_dict["x-tracing-id"] == "grpc-trace-123"
            assert metadata_dict["x-custom"] == "value"

        # After context - no extra headers
        new_details, _, _ = intercept_fn(details, iter(["request"]), False, False)
        metadata_dict = dict(new_details.metadata)
        assert "x-tracing-id" not in metadata_dict

    @pytest.mark.asyncio
    async def test_async_grpc_context_headers(self):
        from qdrant_client.connection import header_adder_async_interceptor

        interceptor = header_adder_async_interceptor(new_metadata=[("api-key", "test")])
        intercept_fn = interceptor._fn

        class FakeCallDetails:
            method = "/qdrant.Points/Search"
            timeout = 5
            metadata = None
            credentials = None

            def _replace(self, **kwargs):
                import copy

                new = copy.copy(self)
                for k, v in kwargs.items():
                    setattr(new, k, v)
                return new

        details = FakeCallDetails()

        # Without context headers
        new_details, _, _ = await intercept_fn(details, iter(["request"]), False, False)
        metadata_dict = dict(new_details.metadata)
        assert "x-tracing-id" not in metadata_dict

        # With context headers
        async with async_headers({"x-tracing-id": "async-grpc-trace"}):
            new_details, _, _ = await intercept_fn(details, iter(["request"]), False, False)
            metadata_dict = dict(new_details.metadata)
            assert metadata_dict["x-tracing-id"] == "async-grpc-trace"

        # After context - no extra headers
        new_details, _, _ = await intercept_fn(details, iter(["request"]), False, False)
        metadata_dict = dict(new_details.metadata)
        assert "x-tracing-id" not in metadata_dict
