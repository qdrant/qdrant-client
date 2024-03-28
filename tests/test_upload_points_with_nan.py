
import uuid

import numpy as np
import pytest

from qdrant_client.http import models as rest_models
from qdrant_client.async_qdrant_client import AsyncQdrantClient
NUM_VECTORS = 100
NUM_QUERIES = 100
DIM = 32
COLLECTION_NAME = 'nan_test_collection'



@pytest.mark.asyncio
async def test_upload_points_with_nan():
    client = AsyncQdrantClient(':memory:')
    await client.create_collection(collection_name=COLLECTION_NAME, vectors_config=rest_models.VectorParams(size=3, distance=rest_models.Distance.COSINE))
    valid_uuid = str(uuid.uuid4())
    point_with_nan = rest_models.PointStruct(id=valid_uuid, vector=np.array([np.nan, 0.1, 0.2]).tolist(), payload={})
    await client.upload_points(collection_name=COLLECTION_NAME, points=[point_with_nan])
    await client.delete_collection(collection_name=COLLECTION_NAME)