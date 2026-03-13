"""Tests for retry / resilience middleware."""

import asyncio
import time
from unittest.mock import MagicMock, AsyncMock, patch

import grpc
import grpc.aio
import pytest
from httpx import ConnectError, Request, Response

from qdrant_client.common.client_exceptions import ResourceExhaustedResponse
from qdrant_client.common.retry import (
    RetryConfig,
    async_retry_middleware,
    compute_backoff,
    is_retryable_grpc_code,
    is_retryable_status,
    retry_interceptor,
    retry_middleware,
)


# ---------------------------------------------------------------------------
# compute_backoff
# ---------------------------------------------------------------------------


class TestComputeBackoff:
    def test_first_attempt(self):
        config = RetryConfig(initial_backoff=1.0, backoff_multiplier=2.0, jitter=0.0)
        assert compute_backoff(0, config) == 1.0

    def test_exponential_growth(self):
        config = RetryConfig(initial_backoff=1.0, backoff_multiplier=2.0, jitter=0.0)
        assert compute_backoff(0, config) == 1.0
        assert compute_backoff(1, config) == 2.0
        assert compute_backoff(2, config) == 4.0

    def test_max_backoff_cap(self):
        config = RetryConfig(initial_backoff=1.0, backoff_multiplier=10.0, max_backoff=5.0, jitter=0.0)
        assert compute_backoff(2, config) == 5.0

    def test_retry_after_floor(self):
        config = RetryConfig(initial_backoff=0.1, jitter=0.0)
        result = compute_backoff(0, config, retry_after=10.0)
        assert result == 10.0

    def test_jitter_adds_randomness(self):
        config = RetryConfig(initial_backoff=1.0, jitter=1.0, backoff_multiplier=1.0)
        values = {compute_backoff(0, config) for _ in range(50)}
        # With jitter=1.0, values should vary (not all identical).
        assert len(values) > 1


# ---------------------------------------------------------------------------
# is_retryable_status / is_retryable_grpc_code
# ---------------------------------------------------------------------------


class TestRetryableChecks:
    def test_default_retryable_statuses(self):
        config = RetryConfig()
        assert is_retryable_status(429, config)
        assert is_retryable_status(503, config)
        assert not is_retryable_status(200, config)
        assert not is_retryable_status(404, config)

    def test_custom_retryable_statuses(self):
        config = RetryConfig(retryable_status_codes=frozenset({418}))
        assert is_retryable_status(418, config)
        assert not is_retryable_status(429, config)

    def test_default_retryable_grpc_codes(self):
        config = RetryConfig()
        assert is_retryable_grpc_code(grpc.StatusCode.UNAVAILABLE, config)
        assert is_retryable_grpc_code(grpc.StatusCode.DEADLINE_EXCEEDED, config)
        assert not is_retryable_grpc_code(grpc.StatusCode.NOT_FOUND, config)


# ---------------------------------------------------------------------------
# REST sync middleware
# ---------------------------------------------------------------------------


class TestSyncRetryMiddleware:
    def test_no_retry_on_success(self):
        config = RetryConfig(max_retries=3)
        mw = retry_middleware(config)

        call_count = 0

        def call_next(request: Request) -> Response:
            nonlocal call_count
            call_count += 1
            return Response(200)

        request = Request("GET", "http://localhost/test")
        response = mw(request, call_next)
        assert response.status_code == 200
        assert call_count == 1

    def test_retries_on_503(self):
        config = RetryConfig(max_retries=2, initial_backoff=0.001, jitter=0.0)
        mw = retry_middleware(config)

        call_count = 0

        def call_next(request: Request) -> Response:
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                return Response(503)
            return Response(200)

        request = Request("GET", "http://localhost/test")
        response = mw(request, call_next)
        assert response.status_code == 200
        assert call_count == 3

    def test_gives_up_after_max_retries(self):
        config = RetryConfig(max_retries=2, initial_backoff=0.001, jitter=0.0)
        mw = retry_middleware(config)

        call_count = 0

        def call_next(request: Request) -> Response:
            nonlocal call_count
            call_count += 1
            return Response(503)

        request = Request("GET", "http://localhost/test")
        response = mw(request, call_next)
        assert response.status_code == 503
        assert call_count == 3  # 1 original + 2 retries

    def test_retries_on_connection_error(self):
        config = RetryConfig(max_retries=2, initial_backoff=0.001, jitter=0.0)
        mw = retry_middleware(config)

        call_count = 0

        def call_next(request: Request) -> Response:
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectError("Connection refused")
            return Response(200)

        request = Request("GET", "http://localhost/test")
        response = mw(request, call_next)
        assert response.status_code == 200
        assert call_count == 3

    def test_connection_error_raises_after_max_retries(self):
        config = RetryConfig(max_retries=1, initial_backoff=0.001, jitter=0.0)
        mw = retry_middleware(config)

        def call_next(request: Request) -> Response:
            raise ConnectError("Connection refused")

        request = Request("GET", "http://localhost/test")
        with pytest.raises(ConnectError):
            mw(request, call_next)

    def test_no_retry_on_non_retryable_status(self):
        config = RetryConfig(max_retries=3)
        mw = retry_middleware(config)

        call_count = 0

        def call_next(request: Request) -> Response:
            nonlocal call_count
            call_count += 1
            return Response(404)

        request = Request("GET", "http://localhost/test")
        response = mw(request, call_next)
        assert response.status_code == 404
        assert call_count == 1

    def test_zero_retries_means_no_retry(self):
        config = RetryConfig(max_retries=0)
        mw = retry_middleware(config)

        call_count = 0

        def call_next(request: Request) -> Response:
            nonlocal call_count
            call_count += 1
            return Response(503)

        request = Request("GET", "http://localhost/test")
        response = mw(request, call_next)
        assert response.status_code == 503
        assert call_count == 1

    def test_retry_after_header_respected(self):
        config = RetryConfig(max_retries=1, initial_backoff=0.001, jitter=0.0)
        mw = retry_middleware(config)

        call_count = 0

        def call_next(request: Request) -> Response:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return Response(429, headers={"Retry-After": "0.01"})
            return Response(200)

        request = Request("GET", "http://localhost/test")
        start = time.monotonic()
        response = mw(request, call_next)
        elapsed = time.monotonic() - start

        assert response.status_code == 200
        assert elapsed >= 0.01


# ---------------------------------------------------------------------------
# REST async middleware
# ---------------------------------------------------------------------------


class TestAsyncRetryMiddleware:
    @pytest.mark.asyncio
    async def test_no_retry_on_success(self):
        config = RetryConfig(max_retries=3)
        mw = async_retry_middleware(config)

        call_count = 0

        async def call_next(request: Request) -> Response:
            nonlocal call_count
            call_count += 1
            return Response(200)

        request = Request("GET", "http://localhost/test")
        response = await mw(request, call_next)
        assert response.status_code == 200
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_retries_on_503(self):
        config = RetryConfig(max_retries=2, initial_backoff=0.001, jitter=0.0)
        mw = async_retry_middleware(config)

        call_count = 0

        async def call_next(request: Request) -> Response:
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                return Response(503)
            return Response(200)

        request = Request("GET", "http://localhost/test")
        response = await mw(request, call_next)
        assert response.status_code == 200
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_retries_on_connection_error(self):
        config = RetryConfig(max_retries=2, initial_backoff=0.001, jitter=0.0)
        mw = async_retry_middleware(config)

        call_count = 0

        async def call_next(request: Request) -> Response:
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectError("Connection refused")
            return Response(200)

        request = Request("GET", "http://localhost/test")
        response = await mw(request, call_next)
        assert response.status_code == 200
        assert call_count == 3


# ---------------------------------------------------------------------------
# gRPC sync retry interceptor
# ---------------------------------------------------------------------------


class TestGrpcRetryInterceptor:
    def test_no_retry_on_success(self):
        config = RetryConfig(max_retries=3)
        interceptor = retry_interceptor(config)

        mock_response = MagicMock()
        mock_response.code.return_value = grpc.StatusCode.OK

        def continuation(details, request):
            return mock_response

        result = interceptor.intercept_unary_unary(continuation, MagicMock(), MagicMock())
        assert result is mock_response

    def test_retries_on_unavailable(self):
        config = RetryConfig(max_retries=2, initial_backoff=0.001, jitter=0.0)
        interceptor = retry_interceptor(config)

        call_count = 0
        mock_success = MagicMock()
        mock_success.code.return_value = grpc.StatusCode.OK

        mock_fail = MagicMock()
        mock_fail.code.return_value = grpc.StatusCode.UNAVAILABLE

        def continuation(details, request):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                return mock_fail
            return mock_success

        result = interceptor.intercept_unary_unary(continuation, MagicMock(), MagicMock())
        assert result is mock_success
        assert call_count == 3

    def test_retries_on_rpc_error(self):
        config = RetryConfig(max_retries=2, initial_backoff=0.001, jitter=0.0)
        interceptor = retry_interceptor(config)

        call_count = 0

        class FakeRpcError(grpc.RpcError):
            def code(self):
                return grpc.StatusCode.UNAVAILABLE

        mock_success = MagicMock()
        mock_success.code.return_value = grpc.StatusCode.OK

        def continuation(details, request):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise FakeRpcError()
            return mock_success

        result = interceptor.intercept_unary_unary(continuation, MagicMock(), MagicMock())
        assert result is mock_success
        assert call_count == 3

    def test_does_not_retry_stream_requests(self):
        """Stream-unary and stream-stream just pass through."""
        config = RetryConfig(max_retries=3)
        interceptor = retry_interceptor(config)

        mock_response = MagicMock()

        def continuation(details, request_iterator):
            return mock_response

        result = interceptor.intercept_stream_unary(continuation, MagicMock(), iter([]))
        assert result is mock_response

    def test_retries_on_resource_exhausted_exception(self):
        config = RetryConfig(max_retries=1, initial_backoff=0.001, jitter=0.0)
        interceptor = retry_interceptor(config)

        call_count = 0
        mock_success = MagicMock()
        mock_success.code.return_value = grpc.StatusCode.OK

        def continuation(details, request):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ResourceExhaustedResponse(message="Rate limited", retry_after_s=0)
            return mock_success

        result = interceptor.intercept_unary_unary(continuation, MagicMock(), MagicMock())
        assert result is mock_success
        assert call_count == 2


# ---------------------------------------------------------------------------
# RetryConfig defaults
# ---------------------------------------------------------------------------


class TestRetryConfig:
    def test_defaults(self):
        config = RetryConfig()
        assert config.max_retries == 3
        assert config.initial_backoff == 0.1
        assert config.max_backoff == 30.0
        assert config.backoff_multiplier == 2.0
        assert config.jitter == 0.1
        assert 429 in config.retryable_status_codes
        assert grpc.StatusCode.UNAVAILABLE in config.retryable_grpc_codes

    def test_frozen(self):
        config = RetryConfig()
        with pytest.raises(AttributeError):
            config.max_retries = 5  # type: ignore

    def test_custom_values(self):
        config = RetryConfig(max_retries=5, initial_backoff=0.5, max_backoff=60.0)
        assert config.max_retries == 5
        assert config.initial_backoff == 0.5
        assert config.max_backoff == 60.0
