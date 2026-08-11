import pytest

from qdrant_client import AsyncQdrantClient, QdrantClient


def test_collection_exists_rejects_empty_name() -> None:
    client = QdrantClient(location=":memory:")
    try:
        with pytest.raises(ValueError, match="Collection name must not be empty"):
            client.collection_exists("")
    finally:
        client.close()


@pytest.mark.asyncio
async def test_async_collection_exists_rejects_empty_name() -> None:
    client = AsyncQdrantClient(location=":memory:")
    try:
        with pytest.raises(ValueError, match="Collection name must not be empty"):
            await client.collection_exists("")
    finally:
        await client.close()
