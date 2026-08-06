from unittest.mock import AsyncMock, Mock

import pytest

from qdrant_client import AsyncQdrantClient, QdrantClient
from qdrant_client.async_qdrant_remote import AsyncQdrantRemote
from qdrant_client.local.async_qdrant_local import AsyncQdrantLocal
from qdrant_client.local.qdrant_local import QdrantLocal
from qdrant_client.qdrant_remote import QdrantRemote


def _package_version(_: str) -> str:
    return "1.0.0"


def test_qdrant_client_health_check_tracks_local_lifecycle() -> None:
    client = QdrantClient(":memory:")
    try:
        assert client.health_check() is True

        client.close()

        assert client.health_check() is False
    finally:
        client.close()


@pytest.mark.asyncio
async def test_async_qdrant_client_health_check_tracks_local_lifecycle() -> None:
    client = AsyncQdrantClient(":memory:")
    try:
        assert await client.health_check() is True

        await client.close()

        assert await client.health_check() is False
    finally:
        await client.close()


def test_qdrant_local_health_check_tracks_lifecycle() -> None:
    client = QdrantLocal(":memory:")
    try:
        assert client.health_check() is True

        client.close()

        assert client.health_check() is False
    finally:
        client.close()


@pytest.mark.asyncio
async def test_async_qdrant_local_health_check_tracks_lifecycle() -> None:
    client = AsyncQdrantLocal(":memory:")
    try:
        assert await client.health_check() is True

        await client.close()

        assert await client.health_check() is False
    finally:
        await client.close()


def test_qdrant_remote_health_check_folds_service_errors_into_false(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "qdrant_client.qdrant_remote.importlib.metadata.version", _package_version
    )
    client = QdrantRemote(url="http://localhost:6333", check_compatibility=False)
    try:
        healthz = Mock(return_value="all is good")
        monkeypatch.setattr(client.openapi_client.service_api, "healthz", healthz)
        assert client.health_check() is True
        healthz.assert_called_once_with()

        healthz.side_effect = RuntimeError("server unavailable")
        assert client.health_check() is False
    finally:
        client.close()


@pytest.mark.asyncio
async def test_async_qdrant_remote_health_check_folds_service_errors_into_false(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "qdrant_client.async_qdrant_remote.importlib.metadata.version",
        _package_version,
    )
    client = AsyncQdrantRemote(url="http://localhost:6333", check_compatibility=False)
    try:
        healthz = AsyncMock(return_value="all is good")
        monkeypatch.setattr(client.http.service_api, "healthz", healthz)

        assert await client.health_check() is True
        healthz.assert_awaited_once_with()

        healthz.side_effect = RuntimeError("server unavailable")
        assert await client.health_check() is False
    finally:
        await client.close()
