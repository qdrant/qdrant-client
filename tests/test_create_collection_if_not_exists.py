import pytest
from qdrant_client import QdrantClient, models
from qdrant_client.async_qdrant_client import AsyncQdrantClient


def test_create_collection_if_not_exists_sync():
    """Test that create_collection with if_not_exists parameter works correctly for sync client"""
    client = QdrantClient(":memory:")
    collection_name = "test_collection_if_not_exists"

    # Create collection with if_not_exists=True (collection doesn't exist yet)
    result = client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(size=10, distance=models.Distance.COSINE),
        if_not_exists=True,
    )
    assert result is True
    assert client.collection_exists(collection_name)

    # Try to create again with if_not_exists=True (should not fail, return True)
    result = client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(size=10, distance=models.Distance.COSINE),
        if_not_exists=True,
    )
    assert result is True

    # Clean up
    client.delete_collection(collection_name)


def test_create_collection_if_not_exists_false_sync():
    """Test that create_collection with if_not_exists=False still creates collection"""
    client = QdrantClient(":memory:")
    collection_name = "test_collection_if_not_exists_false"

    # Create collection without if_not_exists (default behavior)
    result = client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(size=10, distance=models.Distance.COSINE),
    )
    assert result is True
    assert client.collection_exists(collection_name)

    # Clean up
    client.delete_collection(collection_name)


@pytest.mark.asyncio
async def test_create_collection_if_not_exists_async():
    """Test that create_collection with if_not_exists parameter works correctly for async client"""
    client = AsyncQdrantClient(":memory:")
    collection_name = "test_collection_if_not_exists_async"

    # Create collection with if_not_exists=True (collection doesn't exist yet)
    result = await client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(size=10, distance=models.Distance.COSINE),
        if_not_exists=True,
    )
    assert result is True
    assert await client.collection_exists(collection_name)

    # Try to create again with if_not_exists=True (should not fail, return True)
    result = await client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(size=10, distance=models.Distance.COSINE),
        if_not_exists=True,
    )
    assert result is True

    # Clean up
    await client.delete_collection(collection_name)


@pytest.mark.asyncio
async def test_create_collection_if_not_exists_false_async():
    """Test that create_collection with if_not_exists=False still creates collection"""
    client = AsyncQdrantClient(":memory:")
    collection_name = "test_collection_if_not_exists_false_async"

    # Create collection without if_not_exists (default behavior)
    result = await client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(size=10, distance=models.Distance.COSINE),
    )
    assert result is True
    assert await client.collection_exists(collection_name)

    # Clean up
    await client.delete_collection(collection_name)
