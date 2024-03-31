import uuid
import numpy as np
import pytest
from qdrant_client.http import models as rest_models
from qdrant_client.async_qdrant_client import AsyncQdrantClient

COLLECTION_NAME = 'nan_test_collection'


def preprocess_vector(vector):
    # Replace NaN values with 0
    return np.nan_to_num(vector)


@pytest.mark.asyncio
async def test_upload_points_with_nan():
    client = AsyncQdrantClient(':memory:')

    try:
        await client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=rest_models.VectorParams(size=3, distance=rest_models.Distance.COSINE)
        )

        valid_uuid = str(uuid.uuid4())
        raw_vector = np.array([np.nan, 0.1, 0.2])
        processed_vector = preprocess_vector(raw_vector)  # Preprocess the vector
        point_with_nan = rest_models.PointStruct(
            id=valid_uuid,
            vector=processed_vector.tolist(),
            payload={}
        )

        await client.upload_points(collection_name=COLLECTION_NAME, points=[point_with_nan])

    finally:
        await client.delete_collection(collection_name=COLLECTION_NAME)
        await client.close()