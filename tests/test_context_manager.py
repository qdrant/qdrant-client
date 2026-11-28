import pytest

from qdrant_client import AsyncQdrantClient, QdrantClient, models


def test_sync_context_manager_closes_on_exit():
    with QdrantClient(":memory:") as client:
        assert isinstance(client, QdrantClient)
        client.create_collection(
            collection_name="ctx",
            vectors_config=models.VectorParams(size=2, distance=models.Distance.DOT),
        )
        assert client._client.closed is False

    assert client._client.closed is True


def test_sync_context_manager_closes_when_body_raises():
    client = QdrantClient(":memory:")

    with pytest.raises(RuntimeError, match="boom"):
        with client:
            raise RuntimeError("boom")

    assert client._client.closed is True


def test_sync_context_manager_exit_is_idempotent():
    with QdrantClient(":memory:") as client:
        pass

    # A second close() must not raise: QdrantLocal/QdrantRemote.close() are idempotent.
    client.close()
    assert client._client.closed is True


@pytest.mark.asyncio
async def test_async_context_manager_closes_on_exit():
    async with AsyncQdrantClient(":memory:") as client:
        assert isinstance(client, AsyncQdrantClient)
        await client.create_collection(
            collection_name="ctx",
            vectors_config=models.VectorParams(size=2, distance=models.Distance.DOT),
        )
        assert client._client.closed is False

    assert client._client.closed is True


@pytest.mark.asyncio
async def test_async_context_manager_closes_when_body_raises():
    client = AsyncQdrantClient(":memory:")

    with pytest.raises(RuntimeError, match="boom"):
        async with client:
            raise RuntimeError("boom")

    assert client._client.closed is True


@pytest.mark.asyncio
async def test_async_context_manager_exit_is_idempotent():
    async with AsyncQdrantClient(":memory:") as client:
        pass

    await client.close()
    assert client._client.closed is True
