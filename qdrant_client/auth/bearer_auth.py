import asyncio
import threading
from typing import Awaitable, Callable, Optional, Union

import httpx


class BearerAuth(httpx.Auth):
    def __init__(
        self,
        auth_token_provider: Union[Callable[[], str], Callable[[], Awaitable[str]]],
    ):
        self._sync_lock = threading.RLock()
        self._async_lock = asyncio.Lock()
        self.async_token: Optional[Callable[[], Awaitable[str]]] = None
        self.sync_token: Optional[Callable[[], str]] = None

        if asyncio.iscoroutinefunction(auth_token_provider):
            self.async_token = auth_token_provider
        else:
            if callable(auth_token_provider):
                # type narrowing does not work here, so we use `# type: ignore`
                self.sync_token = auth_token_provider  # type: ignore
            else:
                raise ValueError("auth_token_provider must be a callable or awaitable")

    def sync_get_token(self) -> str:
        if self.sync_token is None:
            raise ValueError("Synchronous token provider is not set.")
        with self._sync_lock:
            return self.sync_token()

    def sync_auth_flow(self, request: httpx.Request):
        token = self.sync_get_token()
        request.headers["Authorization"] = f"Bearer {token}"
        yield request

    async def async_get_token(self) -> str:
        if self.async_token is not None:
            async with self._async_lock:
                return await self.async_token()
        # Fallback to synchronous token if asynchronous token is not available
        return self.sync_get_token()

    async def async_auth_flow(self, request: httpx.Request):
        token = await self.async_get_token()
        request.headers["Authorization"] = f"Bearer {token}"
        yield request
