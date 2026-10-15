import asyncio
import time

import numpy as np
import pytest
import pytest_asyncio

from qdrant_client import AsyncQdrantClient, models

COLLECTION_NAME = "test_collection"
DIM = 32
NUM_POINTS = 20_000
NUM_SEARCHES = 20
NUM_HEARTBEATS = 20


@pytest_asyncio.fixture
async def qdrant() -> AsyncQdrantClient:
    client = AsyncQdrantClient(":memory:")
    await client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=models.VectorParams(size=DIM, distance=models.Distance.COSINE),
    )
    await client.upsert(
        collection_name=COLLECTION_NAME,
        points=[
            models.PointStruct(id=i, vector=np.random.rand(DIM).tolist())
            for i in range(NUM_POINTS)
        ],
    )
    return client


@pytest.mark.asyncio
async def test_async_local_query_points_does_not_block_event_loop(qdrant: AsyncQdrantClient):
    """AsyncQdrantLocal.query_points runs a synchronous brute-force numpy scan with no
    await inside it, so it monopolizes the event loop for its full duration. A concurrent
    coroutine doing nothing but asyncio.sleep(0.01) should still wake up close to on time;
    if the loop is blocked, its wakeup gets delayed by however long the scan underneath it
    runs."""
    heartbeat_gaps = []

    async def heartbeat():
        for _ in range(NUM_HEARTBEATS):
            start = time.monotonic()
            await asyncio.sleep(0.01)
            heartbeat_gaps.append(time.monotonic() - start)

    async def search():
        for _ in range(NUM_SEARCHES):
            await qdrant.query_points(
                collection_name=COLLECTION_NAME,
                query=np.random.rand(DIM).tolist(),
                limit=10,
            )

    await asyncio.gather(heartbeat(), search())

    assert max(heartbeat_gaps) < 0.2, (
        f"heartbeat delayed to {max(heartbeat_gaps):.3f}s (scheduled every 0.01s) -- "
        "the event loop was blocked by a concurrent query_points call"
    )
