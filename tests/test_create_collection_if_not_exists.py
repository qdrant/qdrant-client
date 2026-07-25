import pytest
from qdrant_client import QdrantClient, models
from qdrant_client.async_qdrant_client import AsyncQdrantClient


@pytest.fixture
def sync_client():
    client = QdrantClient(":memory:")
    yield client
    client.close()


@pytest.fixture
async def async_client():
    client = AsyncQdrantClient(":memory:")
    yield client
    await client.close()


def test_create_collection_if_not_exists_sync(sync_client):
    """Test that create_collection with if_not_exists parameter works correctly for sync client"""
    collection_name = "test_collection_if_not_exists"

    # Create collection with if_not_exists=True (collection doesn't exist yet)
    result = sync_client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(size=10, distance=models.Distance.COSINE),
        if_not_exists=True,
    )
    assert result is True
    assert sync_client.collection_exists(collection_name)

    # Try to create again with if_not_exists=True (should not fail, return True)
    result = sync_client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(size=10, distance=models.Distance.COSINE),
        if_not_exists=True,
    )
    assert result is True


def test_create_collection_if_not_exists_false_sync(sync_client):
    """Test that create_collection with if_not_exists=False still creates collection"""
    collection_name = "test_collection_if_not_exists_false"

    # Create collection with explicit if_not_exists=False
    result = sync_client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(size=10, distance=models.Distance.COSINE),
        if_not_exists=False,
    )
    assert result is True
    assert sync_client.collection_exists(collection_name)


@pytest.mark.asyncio
async def test_create_collection_if_not_exists_async(async_client):
    """Test that create_collection with if_not_exists parameter works correctly for async client"""
    collection_name = "test_collection_if_not_exists_async"

    # Create collection with if_not_exists=True (collection doesn't exist yet)
    result = await async_client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(size=10, distance=models.Distance.COSINE),
        if_not_exists=True,
    )
    assert result is True
    assert await async_client.collection_exists(collection_name)

    # Try to create again with if_not_exists=True (should not fail, return True)
    result = await async_client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(size=10, distance=models.Distance.COSINE),
        if_not_exists=True,
    )
    assert result is True


@pytest.mark.asyncio
async def test_create_collection_if_not_exists_false_async(async_client):
    """Test that create_collection with if_not_exists=False still creates collection"""
    collection_name = "test_collection_if_not_exists_false_async"

    # Create collection with explicit if_not_exists=False
    result = await async_client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(size=10, distance=models.Distance.COSINE),
        if_not_exists=False,
    )
    assert result is True
    assert await async_client.collection_exists(collection_name)


def test_create_collection_without_if_not_exists_raises_on_duplicate_sync(sync_client):
    """Test that creating an existing collection without if_not_exists=True raises an error."""
    collection_name = "test_duplicate_error"

    sync_client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(size=10, distance=models.Distance.COSINE),
    )

    with pytest.raises(ValueError, match="already exists"):
        sync_client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(size=10, distance=models.Distance.COSINE),
        )


@pytest.mark.asyncio
async def test_create_collection_without_if_not_exists_raises_on_duplicate_async(async_client):
    """Test that creating an existing collection without if_not_exists=True raises an error (async)."""
    collection_name = "test_duplicate_error_async"

    await async_client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(size=10, distance=models.Distance.COSINE),
    )

    with pytest.raises(ValueError, match="already exists"):
        await async_client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(size=10, distance=models.Distance.COSINE),
        )
