import time

import numpy as np
import pytest

from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

COLLECTION_NAME = "test_rest_upload"
VECTOR_SIZE = 256
BATCH_SIZE = 64


def get_data(num_vectors: int):
    return np.random.rand(num_vectors, VECTOR_SIZE)


def prepare_collection_rest():
    client = QdrantClient(timeout=30)
    if client.collection_exists(COLLECTION_NAME):
        client.delete_collection(COLLECTION_NAME)
    client.create_collection(
        COLLECTION_NAME,
        vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
    )


def upload_data(data):
    client = QdrantClient(timeout=30)
    client.upload_collection(
        collection_name=COLLECTION_NAME,
        vectors=data,
        payload=None,
        ids=None,
        batch_size=BATCH_SIZE,
        parallel=2,
    )


@pytest.mark.skip(reason="skip slow benchmark")
def test_rest_upload():
    print("")
    prepare_collection_rest()

    data = get_data(num_vectors=50_000)
    start = time.time()
    upload_data(data)
    end = time.time()
    print("Elapsed", end - start)
