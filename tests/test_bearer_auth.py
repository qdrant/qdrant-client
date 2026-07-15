"""Tests for ``qdrant_client.auth.bearer_auth.BearerAuth``.

The auth class plugs into httpx's ``Auth`` protocol via ``sync_auth_flow``
and ``async_auth_flow``. These tests drive the flows directly so no live
server is required and so we can inspect the outgoing ``Authorization``
header.

Covers:

* Sync provider is detected when the callable is not a coroutine function.
* Async provider is detected when the callable is a coroutine function.
* Non-callable providers raise ``ValueError`` at construction time.
* ``sync_auth_flow`` injects the bearer header and re-fetches the token
  on every invocation (no stale cache).
* ``sync_auth_flow`` raises ``ValueError`` if no sync provider is set
  (e.g. when only an async provider was supplied).
* ``async_auth_flow`` injects the bearer header from the async provider.
* ``async_auth_flow`` falls back to the sync provider when only a sync
  provider was supplied.
"""

from __future__ import annotations

import asyncio

import httpx
import pytest

from qdrant_client.auth.bearer_auth import BearerAuth


def _run_sync(auth: BearerAuth, request: httpx.Request) -> httpx.Request:
    """Drive the sync auth flow to completion and return the final request."""
    flow = auth.sync_auth_flow(request)
    try:
        first = next(flow)
    except StopIteration:
        return request
    # httpx.Auth flows can yield more than once; drain.
    for sent in flow:
        first = sent
    return first


def test_sync_provider_sets_bearer_header() -> None:
    auth = BearerAuth(lambda: "static-token")

    request = httpx.Request("GET", "http://test/")
    out = _run_sync(auth, request)

    assert out.headers["Authorization"] == "Bearer static-token"


def test_sync_provider_called_each_invocation() -> None:
    counter = {"n": 0}

    def provider() -> str:
        counter["n"] += 1
        return f"token-{counter['n']}"

    auth = BearerAuth(provider)

    first = _run_sync(auth, httpx.Request("GET", "http://test/"))
    second = _run_sync(auth, httpx.Request("GET", "http://test/"))

    assert first.headers["Authorization"] == "Bearer token-1"
    assert second.headers["Authorization"] == "Bearer token-2"


def test_sync_flow_without_sync_provider_raises() -> None:
    async def async_provider() -> str:
        return "async-only"

    auth = BearerAuth(async_provider)
    assert auth.sync_token is None

    with pytest.raises(ValueError) as excinfo:
        list(auth.sync_auth_flow(httpx.Request("GET", "http://test/")))
    assert "Synchronous token provider is not set" in str(excinfo.value)


def test_non_callable_provider_raises_value_error() -> None:
    with pytest.raises(ValueError) as excinfo:
        BearerAuth("not-a-callable")  # type: ignore[arg-type]
    assert "must be a callable or awaitable" in str(excinfo.value)


def test_async_provider_sets_bearer_header() -> None:
    async def provider() -> str:
        return "async-token"

    auth = BearerAuth(provider)
    assert auth.async_token is not None
    assert auth.sync_token is None

    async def drive() -> httpx.Request:
        gen = auth.async_auth_flow(httpx.Request("GET", "http://test/"))
        try:
            request = await gen.__anext__()
        except StopAsyncIteration:
            return httpx.Request("GET", "http://test/")
        async for sent in gen:
            request = sent
        return request

    out = asyncio.run(drive())
    assert out.headers["Authorization"] == "Bearer async-token"


def test_async_provider_called_each_invocation() -> None:
    counter = {"n": 0}

    async def provider() -> str:
        counter["n"] += 1
        return f"at-{counter['n']}"

    auth = BearerAuth(provider)

    async def drive() -> httpx.Request:
        gen = auth.async_auth_flow(httpx.Request("GET", "http://test/"))
        try:
            request = await gen.__anext__()
        except StopAsyncIteration:
            return httpx.Request("GET", "http://test/")
        async for sent in gen:
            request = sent
        return request

    first = asyncio.run(drive())
    second = asyncio.run(drive())
    assert first.headers["Authorization"] == "Bearer at-1"
    assert second.headers["Authorization"] == "Bearer at-2"


def test_async_flow_falls_back_to_sync_provider() -> None:
    auth = BearerAuth(lambda: "fallback-token")
    assert auth.async_token is None
    assert auth.sync_token is not None

    async def drive() -> httpx.Request:
        gen = auth.async_auth_flow(httpx.Request("GET", "http://test/"))
        try:
            request = await gen.__anext__()
        except StopAsyncIteration:
            return httpx.Request("GET", "http://test/")
        async for sent in gen:
            request = sent
        return request

    out = asyncio.run(drive())
    assert out.headers["Authorization"] == "Bearer fallback-token"


def test_async_flow_without_any_provider_raises() -> None:
    # Construct then strip both providers to exercise the sync-missing branch.
    auth = BearerAuth(lambda: "x")
    auth.async_token = None
    auth.sync_token = None

    async def drive() -> None:
        gen = auth.async_auth_flow(httpx.Request("GET", "http://test/"))
        await gen.__anext__()

    with pytest.raises(ValueError) as excinfo:
        asyncio.run(drive())
    assert "Synchronous token provider is not set" in str(excinfo.value)


def test_bearer_auth_subclass_of_httpx_auth() -> None:
    auth = BearerAuth(lambda: "t")
    assert isinstance(auth, httpx.Auth)