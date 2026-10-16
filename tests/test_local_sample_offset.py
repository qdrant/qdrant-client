import pytest

from qdrant_client import AsyncQdrantClient, QdrantClient, models


@pytest.mark.parametrize("offset, expected", [(0, 3), (2, 3), (4, 1), (5, 0), (10, 0)])
@pytest.mark.parametrize("prefetched", [False, True])
def test_random_sample_offset(offset, expected, prefetched):
    client = QdrantClient(":memory:")
    try:
        client.create_collection(
            "sample_offset",
            vectors_config=models.VectorParams(size=2, distance=models.Distance.DOT),
        )
        client.upsert(
            "sample_offset",
            [models.PointStruct(id=i, vector=[float(i), 1.0]) for i in range(5)],
        )
        points = client.query_points(
            "sample_offset",
            prefetch=models.Prefetch(query=[1.0, 0.0], limit=5) if prefetched else None,
            query=models.SampleQuery(sample=models.Sample.RANDOM),
            limit=3,
            offset=offset,
        ).points
        assert len(points) == (expected if prefetched else 3)
        assert len({point.id for point in points}) == len(points)
    finally:
        client.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("offset, expected", [(0, 3), (2, 3), (4, 1), (5, 0), (10, 0)])
@pytest.mark.parametrize("prefetched", [False, True])
async def test_async_random_sample_offset(offset, expected, prefetched):
    client = AsyncQdrantClient(":memory:")
    try:
        await client.create_collection(
            "sample_offset",
            vectors_config=models.VectorParams(size=2, distance=models.Distance.DOT),
        )
        await client.upsert(
            "sample_offset",
            [models.PointStruct(id=i, vector=[float(i), 1.0]) for i in range(5)],
        )
        points = (
            await client.query_points(
                "sample_offset",
                prefetch=models.Prefetch(query=[1.0, 0.0], limit=5) if prefetched else None,
                query=models.SampleQuery(sample=models.Sample.RANDOM),
                limit=3,
                offset=offset,
            )
        ).points
        assert len(points) == (expected if prefetched else 3)
        assert len({point.id for point in points}) == len(points)
    finally:
        await client.close()
