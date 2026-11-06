import pytest
from unittest.mock import MagicMock, AsyncMock

from qdrant_client import QdrantClient, AsyncQdrantClient
from qdrant_client.qdrant_remote import QdrantRemote
from qdrant_client.async_qdrant_remote import AsyncQdrantRemote
from qdrant_client.local.qdrant_local import QdrantLocal
from qdrant_client.local.async_qdrant_local import AsyncQdrantLocal


def test_qdrant_local_health_check():
    client = QdrantClient(":memory:")
    assert client.health_check() is True
    assert client._client.health_check() is True

    client.close()
    assert client.health_check() is False
    assert client._client.health_check() is False


@pytest.mark.asyncio
async def test_async_qdrant_local_health_check():
    client = AsyncQdrantClient(":memory:")
    assert await client.health_check() is True
    assert await client._client.health_check() is True

    await client.close()
    assert await client.health_check() is False
    assert await client._client.health_check() is False


def test_qdrant_remote_health_check_success():
    remote = QdrantRemote(url="http://localhost:6333", check_compatibility=False)
    mock_service_api = MagicMock()
    mock_service_api.healthz.return_value = "all is good"
    remote.rest.service_api = mock_service_api

    assert remote.health_check() is True
    mock_service_api.healthz.assert_called_once()


def test_qdrant_remote_health_check_failure():
    remote = QdrantRemote(url="http://localhost:6333", check_compatibility=False)
    mock_service_api = MagicMock()
    mock_service_api.healthz.side_effect = Exception("Connection refused")
    remote.rest.service_api = mock_service_api

    assert remote.health_check() is False
    mock_service_api.healthz.assert_called_once()


@pytest.mark.asyncio
async def test_async_qdrant_remote_health_check_success():
    remote = AsyncQdrantRemote(url="http://localhost:6333", check_compatibility=False)
    mock_service_api = MagicMock()
    mock_service_api.healthz = AsyncMock(return_value="all is good")
    remote.rest.service_api = mock_service_api

    assert await remote.health_check() is True
    mock_service_api.healthz.assert_called_once()


@pytest.mark.asyncio
async def test_async_qdrant_remote_health_check_failure():
    remote = AsyncQdrantRemote(url="http://localhost:6333", check_compatibility=False)
    mock_service_api = MagicMock()
    mock_service_api.healthz = AsyncMock(side_effect=Exception("Connection refused"))
    remote.rest.service_api = mock_service_api

    assert await remote.health_check() is False
    mock_service_api.healthz.assert_called_once()
