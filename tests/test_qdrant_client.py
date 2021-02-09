import os
import random
import uuid
from pprint import pprint
from tempfile import mkdtemp
from time import sleep

import numpy as np

from qdrant_client.qdrant_client import QdrantClient
from qdrant_openapi_client.models.models import PayloadInterfaceAnyOf

DIM = 100
NUM_VECTORS = 1_000
COLLECTION_NAME = 'client_test'


def random_payload():
    for i in range(NUM_VECTORS):
        yield {
            "id": i + 100,
            "text_data": uuid.uuid4().hex,
            "rand_number": random.random()
        }


def create_random_vectors():
    vectors_path = os.path.join(mkdtemp(), 'vectors.npy')
    fp = np.memmap(vectors_path, dtype='float32', mode='w+', shape=(NUM_VECTORS, DIM))

    data = np.random.rand(NUM_VECTORS, DIM).astype(np.float32)
    fp[:] = data[:]
    fp.flush()
    return vectors_path


def test_qdrant_client_integration():
    vectors_path = create_random_vectors()
    vectors = np.memmap(vectors_path, dtype='float32', mode='r', shape=(NUM_VECTORS, DIM))
    payload = random_payload()

    client = QdrantClient()

    client.recreate_collection(
        collection_name=COLLECTION_NAME,
        vector_size=DIM
    )

    # Call Qdrant API to retrieve list of existing collections
    collections = client.http.collections_api.get_collections().result.collections

    # Print all existing collections
    for collection in collections:
        print(collection.dict())

    # Retrieve detailed information about newly created collection
    test_collection = client.http.collections_api.get_collection(COLLECTION_NAME)
    pprint(test_collection.dict())

    # Upload data to a new collection
    client.upload_collection(
        collection_name=COLLECTION_NAME,
        vectors=vectors,
        payload=payload,
        ids=None,  # Let client auto-assign sequential ids
    )

    # By default, Qdrant indexes data updates asynchronously, so client don't need to wait before sending next batch
    # Let's give it a second to actually add all points to a collection.
    # If want need to change this behaviour - simply enable synchronous processing by enabling `wait=true`
    sleep(1)

    # Let's now check details about our new collection
    test_collection = client.http.collections_api.get_collection(COLLECTION_NAME)
    pprint(test_collection.dict())

    # Now we can actually search in the collection
    # Let's create some random vector
    query_vector = np.random.rand(DIM)

    #  and use it as a query
    hits = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vector,
        query_filter=None,  # Don't use any filters for now, search across all indexed points
        append_payload=True,  # Also return a stored payload for found points
        top=5  # Return 5 closest points
    )

    # Print found results
    for hit in hits:
        print(hit)


if __name__ == '__main__':
    test_qdrant_client_integration()
