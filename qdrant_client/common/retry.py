"""Retry / resilience middleware for qdrant-client.

Provides automatic retries with exponential backoff for transient failures
on both REST (via httpx middleware) and gRPC (via interceptors).

Disabled by default — pass ``retry_config=RetryConfig(...)`` to ``QdrantClient``
to enable.

Note: Most Qdrant write operations (upsert, delete by ID, set payload, …) are
idempotent, so retrying them is generally safe.
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Set

import grpc
import grpc.aio
from httpx import ConnectError, ConnectTimeout, ReadTimeout, Request, Response, WriteTimeout

from qdrant_client.common.client_exceptions import ResourceExhaustedResponse

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_DEFAULT_RETRYABLE_STATUS_CODES: frozenset[int] = frozenset({429, 500, 502, 503, 504})

_DEFAULT_RETRYABLE_GRPC_CODES: frozenset[grpc.StatusCode] = frozenset(
    {
        grpc.StatusCode.UNAVAILABLE,
        grpc.StatusCode.DEADLINE_EXCEEDED,
        grpc.StatusCode.RESOURCE_EXHAUSTED,
    }
)

# httpx exceptions that indicate a transient connection-level problem.
_RETRYABLE_HTTPX_EXCEPTIONS = (ConnectError, ConnectTimeout, ReadTimeout, WriteTimeout)


@dataclass(frozen=True)
class RetryConfig:
    """User-facing retry configuration.

    Args:
        max_retries: Maximum number of retry attempts (0 = no retries). Default: 3.
        initial_backoff: Base backoff duration in seconds. Default: 0.1.
        max_backoff: Upper bound on backoff duration in seconds. Default: 30.
        backoff_multiplier: Multiplier applied per attempt. Default: 2.
        jitter: Jitter factor (0–1) added to backoff to avoid thundering herd. Default: 0.1.
        retryable_status_codes: HTTP status codes eligible for retry.
    """

    max_retries: int = 3
    initial_backoff: float = 0.1
    max_backoff: float = 30.0
    backoff_multiplier: float = 2.0
    jitter: float = 0.1
    retryable_status_codes: frozenset[int] = _DEFAULT_RETRYABLE_STATUS_CODES
    retryable_grpc_codes: frozenset[grpc.StatusCode] = _DEFAULT_RETRYABLE_GRPC_CODES


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def compute_backoff(
    attempt: int,
    config: RetryConfig,
    retry_after: float | None = None,
) -> float:
    """Return the backoff duration (seconds) for a given *attempt* (0-indexed).

    If *retry_after* is provided (e.g. from a ``Retry-After`` header), it is
    used as a floor.
    """
    base = config.initial_backoff * (config.backoff_multiplier ** attempt)
    jitter_amount = random.random() * config.jitter * base  # noqa: S311 – not crypto
    backoff = min(base + jitter_amount, config.max_backoff)
    if retry_after is not None:
        backoff = max(backoff, retry_after)
    return backoff


def is_retryable_status(status_code: int, config: RetryConfig) -> bool:
    return status_code in config.retryable_status_codes


def is_retryable_grpc_code(code: grpc.StatusCode, config: RetryConfig) -> bool:
    return code in config.retryable_grpc_codes


def _parse_retry_after(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (ValueError, TypeError):
        return None


# ---------------------------------------------------------------------------
# REST middleware (httpx)
# ---------------------------------------------------------------------------

# Signature matches ``MiddlewareT`` from ``api_client.py``:
#   Callable[[Request, Send], Response]
# where Send = Callable[[Request], Response]


def retry_middleware(
    config: RetryConfig,
) -> Callable[[Request, Callable[[Request], Response]], Response]:
    """Return a sync REST middleware that retries transient failures."""

    def middleware(
        request: Request,
        call_next: Callable[[Request], Response],
    ) -> Response:
        last_exc: Exception | None = None
        for attempt in range(config.max_retries + 1):
            try:
                response = call_next(request)
            except _RETRYABLE_HTTPX_EXCEPTIONS as exc:
                last_exc = exc
                if attempt >= config.max_retries:
                    raise
                backoff = compute_backoff(attempt, config)
                logger.debug(
                    "Retry %d/%d after connection error: %s (backoff=%.2fs)",
                    attempt + 1,
                    config.max_retries,
                    exc,
                    backoff,
                )
                time.sleep(backoff)
                continue

            if is_retryable_status(response.status_code, config) and attempt < config.max_retries:
                retry_after = _parse_retry_after(response.headers.get("Retry-After"))
                backoff = compute_backoff(attempt, config, retry_after=retry_after)
                logger.debug(
                    "Retry %d/%d after HTTP %d (backoff=%.2fs)",
                    attempt + 1,
                    config.max_retries,
                    response.status_code,
                    backoff,
                )
                time.sleep(backoff)
                continue

            return response

        # Shouldn't be reached, but satisfy type checker.
        if last_exc is not None:
            raise last_exc
        return response  # type: ignore[possibly-undefined]

    return middleware


def async_retry_middleware(
    config: RetryConfig,
) -> Callable[[Request, Callable[[Request], Awaitable[Response]]], Awaitable[Response]]:
    """Return an async REST middleware that retries transient failures."""

    async def middleware(
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        last_exc: Exception | None = None
        for attempt in range(config.max_retries + 1):
            try:
                response = await call_next(request)
            except _RETRYABLE_HTTPX_EXCEPTIONS as exc:
                last_exc = exc
                if attempt >= config.max_retries:
                    raise
                backoff = compute_backoff(attempt, config)
                logger.debug(
                    "Retry %d/%d after connection error: %s (backoff=%.2fs)",
                    attempt + 1,
                    config.max_retries,
                    exc,
                    backoff,
                )
                await asyncio.sleep(backoff)
                continue

            if is_retryable_status(response.status_code, config) and attempt < config.max_retries:
                retry_after = _parse_retry_after(response.headers.get("Retry-After"))
                backoff = compute_backoff(attempt, config, retry_after=retry_after)
                logger.debug(
                    "Retry %d/%d after HTTP %d (backoff=%.2fs)",
                    attempt + 1,
                    config.max_retries,
                    response.status_code,
                    backoff,
                )
                await asyncio.sleep(backoff)
                continue

            return response

        if last_exc is not None:
            raise last_exc
        return response  # type: ignore[possibly-undefined]

    return middleware


# ---------------------------------------------------------------------------
# REST install helper
# ---------------------------------------------------------------------------


def install_retry_middleware(openapi_client: Any, config: RetryConfig) -> None:
    """Install retry middleware on a ``SyncApis`` or ``AsyncApis`` wrapper.

    Runtime-dispatches on the underlying client type so generated async code
    does not need special-cased imports.
    """
    from qdrant_client.http.api_client import ApiClient, AsyncApiClient

    client = openapi_client.client
    if isinstance(client, AsyncApiClient):
        client.add_middleware(async_retry_middleware(config))
    elif isinstance(client, ApiClient):
        client.add_middleware(retry_middleware(config))


# ---------------------------------------------------------------------------
# gRPC interceptors
# ---------------------------------------------------------------------------


def retry_interceptor(config: RetryConfig) -> Any:
    """Return a sync gRPC client interceptor that retries transient failures."""
    from qdrant_client.connection import (
        _ClientCallDetails,
        _GenericClientInterceptor,
    )

    def intercept_call(
        client_call_details: Any,
        request_iterator: Any,
        request_streaming: bool,
        response_streaming: bool,
    ) -> tuple:
        # Only retry unary-unary and unary-stream calls.
        # Stream requests consume the iterator on the first attempt.
        if request_streaming:
            return client_call_details, request_iterator, None

        def postprocess(response: Any) -> Any:
            return response

        return client_call_details, request_iterator, None

    # We cannot use the generic interceptor pattern for retry because we need
    # to re-invoke `continuation`.  Instead, build a custom interceptor class.

    class _RetryInterceptor(
        grpc.UnaryUnaryClientInterceptor,
        grpc.UnaryStreamClientInterceptor,
        grpc.StreamUnaryClientInterceptor,
        grpc.StreamStreamClientInterceptor,
    ):
        def intercept_unary_unary(
            self, continuation: Any, client_call_details: Any, request: Any
        ) -> Any:
            for attempt in range(config.max_retries + 1):
                try:
                    response = continuation(client_call_details, request)
                    # Force the call to complete so we can inspect the status.
                    code = response.code()
                    if code is not None and is_retryable_grpc_code(code, config) and attempt < config.max_retries:
                        backoff = compute_backoff(attempt, config)
                        logger.debug(
                            "gRPC retry %d/%d after %s (backoff=%.2fs)",
                            attempt + 1,
                            config.max_retries,
                            code,
                            backoff,
                        )
                        time.sleep(backoff)
                        continue
                    return response
                except grpc.RpcError as exc:
                    if is_retryable_grpc_code(exc.code(), config) and attempt < config.max_retries:
                        backoff = compute_backoff(attempt, config)
                        logger.debug(
                            "gRPC retry %d/%d after %s (backoff=%.2fs)",
                            attempt + 1,
                            config.max_retries,
                            exc.code(),
                            backoff,
                        )
                        time.sleep(backoff)
                        continue
                    raise
                except ResourceExhaustedResponse as exc:
                    if attempt < config.max_retries:
                        backoff = compute_backoff(
                            attempt, config, retry_after=float(exc.retry_after_s) if exc.retry_after_s else None
                        )
                        logger.debug(
                            "gRPC retry %d/%d after ResourceExhausted (backoff=%.2fs)",
                            attempt + 1,
                            config.max_retries,
                            backoff,
                        )
                        time.sleep(backoff)
                        continue
                    raise
            return response  # type: ignore[possibly-undefined]

        def intercept_unary_stream(
            self, continuation: Any, client_call_details: Any, request: Any
        ) -> Any:
            # Don't retry streaming responses – just pass through.
            return continuation(client_call_details, request)

        def intercept_stream_unary(
            self, continuation: Any, client_call_details: Any, request_iterator: Any
        ) -> Any:
            return continuation(client_call_details, request_iterator)

        def intercept_stream_stream(
            self, continuation: Any, client_call_details: Any, request_iterator: Any
        ) -> Any:
            return continuation(client_call_details, request_iterator)

    return _RetryInterceptor()


def retry_async_interceptor(config: RetryConfig) -> Any:
    """Return an async gRPC client interceptor that retries transient failures."""

    class _AsyncRetryInterceptor(
        grpc.aio.UnaryUnaryClientInterceptor,
        grpc.aio.UnaryStreamClientInterceptor,
        grpc.aio.StreamUnaryClientInterceptor,
        grpc.aio.StreamStreamClientInterceptor,
    ):
        async def intercept_unary_unary(
            self, continuation: Any, client_call_details: Any, request: Any
        ) -> Any:
            for attempt in range(config.max_retries + 1):
                try:
                    response = await continuation(client_call_details, request)
                    return response
                except grpc.aio.AioRpcError as exc:
                    if is_retryable_grpc_code(exc.code(), config) and attempt < config.max_retries:
                        backoff = compute_backoff(attempt, config)
                        logger.debug(
                            "async gRPC retry %d/%d after %s (backoff=%.2fs)",
                            attempt + 1,
                            config.max_retries,
                            exc.code(),
                            backoff,
                        )
                        await asyncio.sleep(backoff)
                        continue
                    raise
                except ResourceExhaustedResponse as exc:
                    if attempt < config.max_retries:
                        backoff = compute_backoff(
                            attempt, config, retry_after=float(exc.retry_after_s) if exc.retry_after_s else None
                        )
                        logger.debug(
                            "async gRPC retry %d/%d after ResourceExhausted (backoff=%.2fs)",
                            attempt + 1,
                            config.max_retries,
                            backoff,
                        )
                        await asyncio.sleep(backoff)
                        continue
                    raise
            return response  # type: ignore[possibly-undefined]

        async def intercept_unary_stream(
            self, continuation: Any, client_call_details: Any, request: Any
        ) -> Any:
            return await continuation(client_call_details, request)

        async def intercept_stream_unary(
            self, continuation: Any, client_call_details: Any, request_iterator: Any
        ) -> Any:
            return await continuation(client_call_details, request_iterator)

        async def intercept_stream_stream(
            self, continuation: Any, client_call_details: Any, request_iterator: Any
        ) -> Any:
            return await continuation(client_call_details, request_iterator)

    return _AsyncRetryInterceptor()
