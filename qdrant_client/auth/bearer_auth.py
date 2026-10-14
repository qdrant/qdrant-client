import inspect
from typing import Any, Awaitable, Callable

import httpx


class BearerAuth(httpx.Auth):
    def __init__(
        self,
        auth_token_provider: Callable[[], str] | Callable[[], Awaitable[str]],
    ):
        if not callable(auth_token_provider):
            raise ValueError("auth_token_provider must be a callable or awaitable")
        self.auth_token_provider = auth_token_provider

    def _sync_get_token(self) -> str:
        # Whether the provider is async is only known from what it returns: a lambda or an
        # object with `async def __call__` returns a coroutine without being a coroutine function
        token: Any = self.auth_token_provider()
        if inspect.isawaitable(token):
            if inspect.iscoroutine(token):
                token.close()  # never awaited, close it to avoid a RuntimeWarning
            raise ValueError("Synchronous token provider is not set.")
        return token

    def sync_auth_flow(self, request: httpx.Request) -> httpx.Request:
        token = self._sync_get_token()
        request.headers["Authorization"] = f"Bearer {token}"
        yield request

    async def _async_get_token(self) -> str:
        token: Any = self.auth_token_provider()
        if inspect.isawaitable(token):
            token = await token
        return token

    async def async_auth_flow(self, request: httpx.Request) -> httpx.Request:
        token = await self._async_get_token()
        request.headers["Authorization"] = f"Bearer {token}"
        yield request
