import asyncio

import numpy as np
import pytest

from qdrant_client import QdrantClient, grpc
from qdrant_client.conversions.conversion import payload_to_grpc
from tests.fixtures.payload import one_random_payload_please

NUM_VECTORS = 100
NUM_QUERIES = 100
DIM = 32


@pytest.mark.asyncio
async def test_async_grpc():
    records = (
        grpc.PointStruct(
            id=grpc.PointId(num=idx),
            vectors=grpc.Vectors(vector=grpc.Vector(data=np.random.rand(DIM).tolist())),
            payload=payload_to_grpc(one_random_payload_please(idx)),
        )
        for idx in range(NUM_VECTORS)
    )

    client = QdrantClient(prefer_grpc=True, timeout=3.0)

    grpc_collections = client.async_grpc_collections

    res = await grpc_collections.List(grpc.ListCollectionsRequest(), timeout=1.0)

    for collection in res.collections:
        print(collection.name)
        await grpc_collections.Delete(grpc.DeleteCollection(collection_name=collection.name))

    await grpc_collections.Create(
        grpc.CreateCollection(
            collection_name="test_collection",
            vectors_config=grpc.VectorsConfig(
                params=grpc.VectorParams(size=DIM, distance=grpc.Distance.Cosine)
            ),
        )
    )

    grpc_points = client.async_grpc_points

    upload_features = []

    # Upload vectors in parallel
    for record in records:
        upload_features.append(
            grpc_points.Upsert(
                grpc.UpsertPoints(collection_name="test_collection", wait=True, points=[record])
            )
        )
    await asyncio.gather(*upload_features)

    queries = [np.random.rand(DIM).tolist() for _ in range(NUM_QUERIES)]

    # Make async queries
    search_queries = []
    for query in queries:
        search_query = grpc_points.Search(
            grpc.SearchPoints(
                collection_name="test_collection",
                vector=query,
                limit=10,
            )
        )
        search_queries.append(search_query)
    results = await asyncio.gather(*search_queries)  # All queries are running in parallel now

    assert len(results) == NUM_QUERIES

    for result in results:
        assert len(result.result) == 10

    client.close()
