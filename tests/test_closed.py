import pytest

from qdrant_client import AsyncQdrantClient, QdrantClient


def test_local_client_reports_closed_state():
    client = QdrantClient(":memory:")
    assert client.closed is False

    client.close()
    assert client.closed is True


def test_persistent_local_client_reports_closed_state(tmp_path):
    client = QdrantClient(path=str(tmp_path / "storage"))
    assert client.closed is False

    client.close()
    assert client.closed is True


def test_remote_client_reports_closed_state():
    client = QdrantClient("localhost", port=6333, check_compatibility=False)
    assert client.closed is False

    client.close()
    assert client.closed is True


def test_closed_matches_the_inner_client():
    client = QdrantClient(":memory:")
    assert client.closed is client._client.closed

    client.close()
    assert client.closed is client._client.closed


@pytest.mark.asyncio
async def test_async_local_client_reports_closed_state():
    client = AsyncQdrantClient(":memory:")
    assert client.closed is False

    await client.close()
    assert client.closed is True


@pytest.mark.asyncio
async def test_async_remote_client_reports_closed_state():
    client = AsyncQdrantClient("localhost", port=6333, check_compatibility=False)
    assert client.closed is False

    await client.close()
    assert client.closed is True
