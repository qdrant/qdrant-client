"""Retry support for the remote Qdrant client.

This module provides a transport-level retry mechanism with exponential backoff
and jitter for both the REST (httpx) and gRPC transports:

* REST: :func:`retry_middleware` / :func:`async_retry_middleware` are injected
  into the generated ``ApiClient`` middleware chain. They re-send the request on
  transient transport errors and on a configurable set of HTTP status codes,
  honouring the server provided ``Retry-After`` header when present.
* gRPC: :func:`retry_to_grpc_options` translates a :class:`RetryConfig` into the
  native gRPC ``service_config`` retry policy, which is handled by gRPC core.

Retries are opt-in: when no :class:`RetryConfig` is provided the client behaves
exactly as before.
"""

import asyncio
import json
import random
import time
from email.utils import parsedate_to_datetime
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Optional, Tuple, Union

import httpx
from pydantic import BaseModel, Field

from qdrant_client.http.exceptions import ResponseHandlingException

DEFAULT_RETRY_ON_STATUS: Tuple[int, ...] = (429, 502, 503, 504)


class RetryConfig(BaseModel):
    """Configuration for automatic request retries with exponential backoff.

    Args:
        max_retries: Maximum number of retries *after* the initial attempt.
            The total number of attempts is ``max_retries + 1``. Default: 3.
        backoff_factor: Base delay (in seconds) for the exponential backoff.
            The delay before retry ``n`` (0-based) is ``backoff_factor * 2 ** n``,
            clamped to ``max_backoff``. Default: 0.5.
        max_backoff: Maximum delay (in seconds) between attempts. Default: 10.0.
        jitter: If ``True``, apply full jitter to the computed backoff to avoid
            thundering-herd retries. Default: ``True``.
        retry_on_status: HTTP status codes that should trigger a retry (REST).
            Default: ``(429, 502, 503, 504)``.
        retry_on_timeout: If ``True``, retry on transient transport errors such
            as connection errors and timeouts. Default: ``True``.
        respect_retry_after: If ``True``, honour the server provided
            ``Retry-After`` header (REST) instead of the computed backoff.
            Default: ``True``.
    """

    max_retries: int = Field(default=3, ge=0)
    backoff_factor: float = Field(default=0.5, ge=0.0)
    max_backoff: float = Field(default=10.0, ge=0.0)
    jitter: bool = True
    retry_on_status: Tuple[int, ...] = DEFAULT_RETRY_ON_STATUS
    retry_on_timeout: bool = True
    respect_retry_after: bool = True


RestSend = Callable[[httpx.Request], httpx.Response]
RestSendAsync = Callable[[httpx.Request], Awaitable[httpx.Response]]


def _compute_backoff(config: RetryConfig, attempt: int) -> float:
    """Exponential backoff (with optional full jitter) for a 0-based attempt."""
    delay = config.backoff_factor * (2**attempt)
    delay = min(delay, config.max_backoff)
    if config.jitter:
        delay = random.uniform(0.0, delay)
    return delay


def _is_retryable_exception(exc: Optional[BaseException]) -> bool:
    """Whether a transport-level exception is safe to retry.

    ``httpx.TransportError`` is the base class for connection errors, timeouts,
    network errors and protocol errors. Higher level errors (e.g. invalid URL,
    too many redirects) are intentionally not retried.
    """
    return isinstance(exc, httpx.TransportError)


def _retry_after_seconds(response: httpx.Response) -> Optional[float]:
    """Parse the ``Retry-After`` header (delta-seconds or HTTP-date)."""
    value = response.headers.get("Retry-After")
    if not value:
        return None
    value = value.strip()
    try:
        return max(0.0, float(value))
    except ValueError:
        pass
    try:
        retry_date = parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None
    if retry_date is None:
        return None
    if retry_date.tzinfo is None:
        retry_date = retry_date.replace(tzinfo=timezone.utc)
    return max(0.0, (retry_date - datetime.now(timezone.utc)).total_seconds())


def _next_delay(config: RetryConfig, attempt: int, response: Optional[httpx.Response]) -> float:
    if response is not None and config.respect_retry_after:
        retry_after = _retry_after_seconds(response)
        if retry_after is not None:
            return min(retry_after, config.max_backoff) if config.max_backoff else retry_after
    return _compute_backoff(config, attempt)


def retry_middleware(config: RetryConfig) -> Callable[[httpx.Request, RestSend], httpx.Response]:
    """Build a synchronous REST middleware that retries transient failures."""

    def middleware(request: httpx.Request, call_next: RestSend) -> httpx.Response:
        response: Optional[httpx.Response] = None
        for attempt in range(config.max_retries + 1):
            is_last = attempt >= config.max_retries
            try:
                response = call_next(request)
            except ResponseHandlingException as exc:
                if (
                    config.retry_on_timeout
                    and not is_last
                    and _is_retryable_exception(getattr(exc, "source", None))
                ):
                    time.sleep(_compute_backoff(config, attempt))
                    continue
                raise

            if response.status_code in config.retry_on_status and not is_last:
                time.sleep(_next_delay(config, attempt, response))
                continue
            return response

        assert response is not None  # max_retries >= 0 guarantees at least one attempt
        return response

    return middleware


def async_retry_middleware(
    config: RetryConfig,
) -> Callable[[httpx.Request, RestSendAsync], Awaitable[httpx.Response]]:
    """Build an asynchronous REST middleware that retries transient failures."""

    async def middleware(request: httpx.Request, call_next: RestSendAsync) -> httpx.Response:
        response: Optional[httpx.Response] = None
        for attempt in range(config.max_retries + 1):
            is_last = attempt >= config.max_retries
            try:
                response = await call_next(request)
            except ResponseHandlingException as exc:
                if (
                    config.retry_on_timeout
                    and not is_last
                    and _is_retryable_exception(getattr(exc, "source", None))
                ):
                    await asyncio.sleep(_compute_backoff(config, attempt))
                    continue
                raise

            if response.status_code in config.retry_on_status and not is_last:
                await asyncio.sleep(_next_delay(config, attempt, response))
                continue
            return response

        assert response is not None
        return response

    return middleware


def retry_to_grpc_options(config: RetryConfig) -> dict[str, Any]:
    """Translate a :class:`RetryConfig` into native gRPC retry channel options.

    gRPC core handles the backoff (with built-in jitter) once the channel is
    configured with ``grpc.enable_retries`` and a ``grpc.service_config`` that
    contains a ``retryPolicy``.
    """
    status_codes = ["UNAVAILABLE"]
    if 429 in config.retry_on_status:
        status_codes.append("RESOURCE_EXHAUSTED")
    if config.retry_on_timeout:
        status_codes.append("DEADLINE_EXCEEDED")

    # gRPC requires 2 <= maxAttempts; the default hard cap in gRPC core is 5.
    max_attempts = max(2, min(config.max_retries + 1, 5))
    initial_backoff = max(config.backoff_factor, 0.001)
    max_backoff = max(config.max_backoff, initial_backoff)

    service_config = {
        "methodConfig": [
            {
                "name": [{}],  # apply to every method on the channel
                "retryPolicy": {
                    "maxAttempts": max_attempts,
                    "initialBackoff": f"{initial_backoff}s",
                    "maxBackoff": f"{max_backoff}s",
                    "backoffMultiplier": 2.0,
                    "retryableStatusCodes": status_codes,
                },
            }
        ]
    }

    return {
        "grpc.enable_retries": 1,
        "grpc.service_config": json.dumps(service_config),
    }


def coerce_retry_config(
    retry: Union[RetryConfig, dict[str, Any], None],
) -> Optional[RetryConfig]:
    """Normalise the public ``retry`` argument into a :class:`RetryConfig`."""
    if retry is None:
        return None
    if isinstance(retry, RetryConfig):
        return retry
    if isinstance(retry, dict):
        return RetryConfig(**retry)
    raise TypeError(f"`retry` must be a RetryConfig, dict or None, but got {type(retry)}")
