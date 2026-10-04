import json

import httpx
import pytest

from qdrant_client.common.retry import (
    RetryConfig,
    _compute_backoff,
    _retry_after_seconds,
    async_retry_middleware,
    coerce_retry_config,
    retry_middleware,
    retry_to_grpc_options,
)
from qdrant_client.http.exceptions import ResponseHandlingException


def _request() -> httpx.Request:
    return httpx.Request("GET", "http://localhost:6333/collections")


def _response(status_code: int, headers: dict | None = None) -> httpx.Response:
    return httpx.Response(status_code=status_code, headers=headers or {})


class _FlakyCallNext:
    """Callable that fails a fixed number of times before succeeding."""

    def __init__(self, outcomes: list):
        self.outcomes = list(outcomes)
        self.calls = 0

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.calls += 1
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    monkeypatch.setattr("qdrant_client.common.retry.time.sleep", lambda *_: None)
    monkeypatch.setattr("qdrant_client.common.retry.asyncio.sleep", _async_noop)


async def _async_noop(*_args, **_kwargs):
    return None


def test_retry_config_defaults():
    config = RetryConfig()
    assert config.max_retries == 3
    assert config.retry_on_status == (429, 502, 503, 504)
    assert config.retry_on_timeout is True


def test_coerce_retry_config():
    assert coerce_retry_config(None) is None
    cfg = coerce_retry_config({"max_retries": 7})
    assert isinstance(cfg, RetryConfig)
    assert cfg.max_retries == 7
    existing = RetryConfig(max_retries=1)
    assert coerce_retry_config(existing) is existing
    with pytest.raises(TypeError):
        coerce_retry_config(42)


def test_compute_backoff_is_bounded():
    config = RetryConfig(backoff_factor=1.0, max_backoff=4.0, jitter=False)
    assert _compute_backoff(config, 0) == 1.0
    assert _compute_backoff(config, 1) == 2.0
    assert _compute_backoff(config, 2) == 4.0
    assert _compute_backoff(config, 10) == 4.0  # clamped to max_backoff


def test_retry_after_parsing():
    assert _retry_after_seconds(_response(429, {"Retry-After": "5"})) == 5.0
    assert _retry_after_seconds(_response(429)) is None


def test_rest_middleware_retries_then_succeeds():
    config = RetryConfig(max_retries=3, jitter=False)
    call_next = _FlakyCallNext([_response(503), _response(503), _response(200)])
    middleware = retry_middleware(config)

    response = middleware(_request(), call_next)

    assert response.status_code == 200
    assert call_next.calls == 3


def test_rest_middleware_gives_up_and_returns_last_response():
    config = RetryConfig(max_retries=2, jitter=False)
    call_next = _FlakyCallNext([_response(503), _response(503), _response(503)])
    middleware = retry_middleware(config)

    response = middleware(_request(), call_next)

    assert response.status_code == 503
    assert call_next.calls == 3  # 1 initial + 2 retries


def test_rest_middleware_does_not_retry_success():
    config = RetryConfig(max_retries=3, jitter=False)
    call_next = _FlakyCallNext([_response(200), _response(503)])
    middleware = retry_middleware(config)

    response = middleware(_request(), call_next)

    assert response.status_code == 200
    assert call_next.calls == 1


def test_rest_middleware_retries_transient_exception():
    config = RetryConfig(max_retries=2, jitter=False)
    transient = ResponseHandlingException(httpx.ConnectError("boom"))
    call_next = _FlakyCallNext([transient, _response(200)])
    middleware = retry_middleware(config)

    response = middleware(_request(), call_next)

    assert response.status_code == 200
    assert call_next.calls == 2


def test_rest_middleware_reraises_exhausted_exception():
    config = RetryConfig(max_retries=1, jitter=False)
    transient = ResponseHandlingException(httpx.ConnectError("boom"))
    call_next = _FlakyCallNext([transient, transient])
    middleware = retry_middleware(config)

    with pytest.raises(ResponseHandlingException):
        middleware(_request(), call_next)
    assert call_next.calls == 2


def test_rest_middleware_does_not_retry_non_transient_exception():
    config = RetryConfig(max_retries=3, retry_on_timeout=True, jitter=False)
    # ValueError is not an httpx.TransportError -> must not be retried
    non_transient = ResponseHandlingException(ValueError("nope"))
    call_next = _FlakyCallNext([non_transient, _response(200)])
    middleware = retry_middleware(config)

    with pytest.raises(ResponseHandlingException):
        middleware(_request(), call_next)
    assert call_next.calls == 1


@pytest.mark.asyncio
async def test_async_rest_middleware_retries_then_succeeds():
    config = RetryConfig(max_retries=3, jitter=False)
    outcomes = [_response(503), _response(200)]
    calls = {"n": 0}

    async def call_next(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return outcomes.pop(0)

    middleware = async_retry_middleware(config)
    response = await middleware(_request(), call_next)

    assert response.status_code == 200
    assert calls["n"] == 2


def test_retry_to_grpc_options():
    config = RetryConfig(max_retries=4, backoff_factor=0.5, max_backoff=10.0)
    options = retry_to_grpc_options(config)

    assert options["grpc.enable_retries"] == 1
    service_config = json.loads(options["grpc.service_config"])
    policy = service_config["methodConfig"][0]["retryPolicy"]
    assert policy["maxAttempts"] == 5  # capped at gRPC default hard limit
    assert "UNAVAILABLE" in policy["retryableStatusCodes"]
    assert "RESOURCE_EXHAUSTED" in policy["retryableStatusCodes"]
    assert "DEADLINE_EXCEEDED" in policy["retryableStatusCodes"]


def test_retry_to_grpc_options_respects_status_set():
    config = RetryConfig(retry_on_status=(502, 503), retry_on_timeout=False)
    policy = json.loads(retry_to_grpc_options(config)["grpc.service_config"])["methodConfig"][0][
        "retryPolicy"
    ]
    assert "RESOURCE_EXHAUSTED" not in policy["retryableStatusCodes"]
    assert "DEADLINE_EXCEEDED" not in policy["retryableStatusCodes"]
