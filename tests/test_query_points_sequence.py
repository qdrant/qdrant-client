import pytest
import numpy as np
from qdrant_client import QdrantClient, models
from qdrant_client.async_qdrant_client import AsyncQdrantClient


def test_query_points_with_tuple_sync():
    """Test that query_points accepts tuple (Sequence[float]) for sync client"""
    client = QdrantClient(":memory:")
    collection_name = "test_query_tuple"

    # Create collection
    client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(size=10, distance=models.Distance.COSINE),
    )

    # Add some test points
    client.upsert(
        collection_name=collection_name,
        points=[
            models.PointStruct(
                id=i,
                vector=[float(i)] * 10,
                payload={"value": i},
            )
            for i in range(5)
        ],
    )

    # Test with tuple (which is a Sequence but not a list)
    query_tuple = tuple([1.0] * 10)
    result = client.query_points(
        collection_name=collection_name,
        query=query_tuple,
        limit=5,
    )
    assert len(result.points) == 5

    # Test with list (existing behavior)
    query_list = [1.0] * 10
    result = client.query_points(
        collection_name=collection_name,
        query=query_list,
        limit=5,
    )
    assert len(result.points) == 5

    # Test with numpy array (existing behavior)
    query_numpy = np.array([1.0] * 10)
    result = client.query_points(
        collection_name=collection_name,
        query=query_numpy,
        limit=5,
    )
    assert len(result.points) == 5

    # Clean up
    client.delete_collection(collection_name)


@pytest.mark.asyncio
async def test_query_points_with_tuple_async():
    """Test that query_points accepts tuple (Sequence[float]) for async client"""
    client = AsyncQdrantClient(":memory:")
    collection_name = "test_query_tuple_async"

    # Create collection
    await client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(size=10, distance=models.Distance.COSINE),
    )

    # Add some test points
    await client.upsert(
        collection_name=collection_name,
        points=[
            models.PointStruct(
                id=i,
                vector=[float(i)] * 10,
                payload={"value": i},
            )
            for i in range(5)
        ],
    )

    # Test with tuple (which is a Sequence but not a list)
    query_tuple = tuple([1.0] * 10)
    result = await client.query_points(
        collection_name=collection_name,
        query=query_tuple,
        limit=5,
    )
    assert len(result.points) == 5

    # Test with list (existing behavior)
    query_list = [1.0] * 10
    result = await client.query_points(
        collection_name=collection_name,
        query=query_list,
        limit=5,
    )
    assert len(result.points) == 5

    # Test with numpy array (existing behavior)
    query_numpy = np.array([1.0] * 10)
    result = await client.query_points(
        collection_name=collection_name,
        query=query_numpy,
        limit=5,
    )
    assert len(result.points) == 5

    # Clean up
    await client.delete_collection(collection_name)


def test_query_points_groups_with_tuple_sync():
    """Test that query_points_groups accepts tuple (Sequence[float]) for sync client"""
    client = QdrantClient(":memory:")
    collection_name = "test_query_groups_tuple"

    # Create collection
    client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(size=10, distance=models.Distance.COSINE),
    )

    # Add some test points with groups
    client.upsert(
        collection_name=collection_name,
        points=[
            models.PointStruct(
                id=i,
                vector=[float(i % 3)] * 10,
                payload={"group": i % 3, "value": i},
            )
            for i in range(9)
        ],
    )

    # Test with tuple (which is a Sequence but not a list)
    query_tuple = tuple([1.0] * 10)
    result = client.query_points_groups(
        collection_name=collection_name,
        query=query_tuple,
        group_by="group",
        limit=3,
    )
    assert len(result.groups) >= 1

    # Clean up
    client.delete_collection(collection_name)


@pytest.mark.asyncio
async def test_query_points_groups_with_tuple_async():
    """Test that query_points_groups accepts tuple (Sequence[float]) for async client"""
    client = AsyncQdrantClient(":memory:")
    collection_name = "test_query_groups_tuple_async"

    # Create collection
    await client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(size=10, distance=models.Distance.COSINE),
    )

    # Add some test points with groups
    await client.upsert(
        collection_name=collection_name,
        points=[
            models.PointStruct(
                id=i,
                vector=[float(i % 3)] * 10,
                payload={"group": i % 3, "value": i},
            )
            for i in range(9)
        ],
    )

    # Test with tuple (which is a Sequence but not a list)
    query_tuple = tuple([1.0] * 10)
    result = await client.query_points_groups(
        collection_name=collection_name,
        query=query_tuple,
        group_by="group",
        limit=3,
    )
    assert len(result.groups) >= 1

    # Clean up
    await client.delete_collection(collection_name)


def test_query_points_with_deque_sync():
    """Test that query_points accepts deque (generic Sequence[float]) for sync client"""
    from collections import deque

    client = QdrantClient(":memory:")
    collection_name = "test_query_deque"

    client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(size=10, distance=models.Distance.COSINE),
    )

    client.upsert(
        collection_name=collection_name,
        points=[
            models.PointStruct(
                id=i,
                vector=[float(i)] * 10,
                payload={"value": i},
            )
            for i in range(5)
        ],
    )

    # Test with deque (a Sequence that is not list/tuple/ndarray)
    query_deque = deque([1.0] * 10)
    result = client.query_points(
        collection_name=collection_name,
        query=query_deque,
        limit=5,
    )
    assert len(result.points) == 5

    # Verify that bytes, bytearray, and range are rejected
    with pytest.raises(ValueError, match="Unsupported query type"):
        client.query_points(
            collection_name=collection_name,
            query=b"\x00" * 10,
            limit=5,
        )

    with pytest.raises(ValueError, match="Unsupported query type"):
        client.query_points(
            collection_name=collection_name,
            query=bytearray(10),
            limit=5,
        )

    with pytest.raises(ValueError, match="Unsupported query type"):
        client.query_points(
            collection_name=collection_name,
            query=range(10),
            limit=5,
        )

    client.delete_collection(collection_name)


@pytest.mark.asyncio
async def test_query_points_with_deque_async():
    """Test that query_points accepts deque (generic Sequence[float]) for async client"""
    from collections import deque

    client = AsyncQdrantClient(":memory:")
    collection_name = "test_query_deque_async"

    await client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(size=10, distance=models.Distance.COSINE),
    )

    await client.upsert(
        collection_name=collection_name,
        points=[
            models.PointStruct(
                id=i,
                vector=[float(i)] * 10,
                payload={"value": i},
            )
            for i in range(5)
        ],
    )

    # Test with deque (a Sequence that is not list/tuple/ndarray)
    query_deque = deque([1.0] * 10)
    result = await client.query_points(
        collection_name=collection_name,
        query=query_deque,
        limit=5,
    )
    assert len(result.points) == 5

    # Verify that bytes, bytearray, and range are rejected
    with pytest.raises(ValueError, match="Unsupported query type"):
        await client.query_points(
            collection_name=collection_name,
            query=b"\x00" * 10,
            limit=5,
        )

    with pytest.raises(ValueError, match="Unsupported query type"):
        await client.query_points(
            collection_name=collection_name,
            query=bytearray(10),
            limit=5,
        )

    with pytest.raises(ValueError, match="Unsupported query type"):
        await client.query_points(
            collection_name=collection_name,
            query=range(10),
            limit=5,
        )

    await client.delete_collection(collection_name)
