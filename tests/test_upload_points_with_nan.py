import asyncio
import os
import random
import time
import uuid
import grpc.aio._call
import numpy as np
import pytest

import qdrant_client.http.exceptions
from qdrant_client import QdrantClient
from qdrant_client import grpc as qdrant_grpc
from qdrant_client import models
from qdrant_client.http import models as rest_models
from qdrant_client.async_qdrant_client import AsyncQdrantClient
from qdrant_client.conversions.conversion import payload_to_grpc
from tests.fixtures.payload import one_random_payload_please

NUM_VECTORS = 100
NUM_QUERIES = 100
DIM = 32
COLLECTION_NAME = "nan_test_collection"


@pytest.mark.asyncio
async def test_async_grpc():
    points = (
        qdrant_grpc.PointStruct(
            id=qdrant_grpc.PointId(num=idx),
            vectors=qdrant_grpc.Vectors(
                vector=qdrant_grpc.Vector(data=np.random.rand(DIM).tolist())
            ),
            payload=payload_to_grpc(one_random_payload_please(idx)),
        )
        for idx in range(NUM_VECTORS)
    )

    client = QdrantClient(prefer_grpc=True, timeout=3.0)

    grpc_collections = client.async_grpc_collections

    res = await grpc_collections.List(qdrant_grpc.ListCollectionsRequest(), timeout=1.0)

    for collection in res.collections:
        print(collection.name)
        await grpc_collections.Delete(
            qdrant_grpc.DeleteCollection(collection_name=collection.name)
        )

    await grpc_collections.Create(
        qdrant_grpc.CreateCollection(
            collection_name=COLLECTION_NAME,
            vectors_config=qdrant_grpc.VectorsConfig(
                params=qdrant_grpc.VectorParams(size=DIM, distance=qdrant_grpc.Distance.Cosine)
            ),
        )
    )

    grpc_points = client.async_grpc_points

    upload_features = []

    # Upload vectors in parallel
    for point in points:
        upload_features.append(
            grpc_points.Upsert(
                qdrant_grpc.UpsertPoints(
                    collection_name=COLLECTION_NAME, wait=True, points=[point]
                )
            )
        )
    await asyncio.gather(*upload_features)

    queries = [np.random.rand(DIM).tolist() for _ in range(NUM_QUERIES)]

    # Make async queries
    search_queries = []
    for query in queries:
        search_query = grpc_points.Search(
            qdrant_grpc.SearchPoints(
                collection_name=COLLECTION_NAME,
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

@pytest.mark.asyncio
async def test_upload_points_with_nan():
    client = AsyncQdrantClient(":memory:")
    await client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=rest_models.VectorParams(
            size=3, distance=rest_models.Distance.COSINE
        ),
    )

    # Generate a valid UUID for the point id
    valid_uuid = str(uuid.uuid4())

    # Define a point with NaN values in its vector
    point_with_nan = rest_models.PointStruct(
        id=valid_uuid, vector=np.array([np.nan, 0.1, 0.2]).tolist(), payload={}
    )

    # Attempt to upload the point with NaN values
    await client.upload_points(collection_name=COLLECTION_NAME, points=[point_with_nan])

    # Clean up any resources if needed
    await client.delete_collection(collection_name=COLLECTION_NAME)