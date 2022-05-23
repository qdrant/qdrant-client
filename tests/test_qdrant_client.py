import os
import random
import uuid
from pprint import pprint
from tempfile import mkdtemp
from time import sleep

import numpy as np
import pytest

from qdrant_client import QdrantClient
from qdrant_client.conversions.conversion import grpc_to_payload, json_to_value
from qdrant_client.http.models import Filter, FieldCondition, Range, PointsList, PointStruct, PointRequest, \
    SetPayload, HasIdCondition, PointIdsList, PayloadSchemaType, MatchValue, Distance
from qdrant_client.uploader.grpc_uploader import payload_to_grpc

DIM = 100
NUM_VECTORS = 1_000
COLLECTION_NAME = 'client_test'


def random_payload():
    for i in range(NUM_VECTORS):
        yield {
            "id": i + 100,
            "text_data": uuid.uuid4().hex,
            "rand_number": random.random(),
            "text_array": [uuid.uuid4().hex, uuid.uuid4().hex]
        }


def create_random_vectors():
    vectors_path = os.path.join(mkdtemp(), 'vectors.npy')
    fp = np.memmap(vectors_path, dtype='float32', mode='w+', shape=(NUM_VECTORS, DIM))

    data = np.random.rand(NUM_VECTORS, DIM).astype(np.float32)
    fp[:] = data[:]
    fp.flush()
    return vectors_path


@pytest.mark.parametrize(
    "prefer_grpc", [False, True]
)
def test_qdrant_client_integration(prefer_grpc):
    vectors_path = create_random_vectors()
    vectors = np.memmap(vectors_path, dtype='float32', mode='r', shape=(NUM_VECTORS, DIM))
    payload = random_payload()

    client = QdrantClient(prefer_grpc=prefer_grpc)

    client.recreate_collection(
        collection_name=COLLECTION_NAME,
        vector_size=DIM,
        distance=Distance.DOT,
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
        parallel=2
    )

    # By default, Qdrant indexes data updates asynchronously, so client don't need to wait before sending next batch
    # Let's give it a second to actually add all points to a collection.
    # If you need to change this behaviour - simply enable synchronous processing by enabling `wait=true`
    sleep(1)

    # Create payload index for field `random_num`
    # If indexed field appear in filtering condition - search operation could be performed faster
    index_create_result = client.create_payload_index(COLLECTION_NAME, "random_num", PayloadSchemaType.FLOAT)
    pprint(index_create_result.dict())

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

    assert len(hits) == 5

    # Print found results
    print("Search result:")
    for hit in hits:
        print(hit)

    # Let's now query same vector with filter condition
    hits = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vector,
        query_filter=Filter(
            must=[  # These conditions are required for search results
                FieldCondition(
                    key='rand_number',  # Condition based on values of `rand_number` field.
                    range=Range(
                        gte=0.5  # Select only those results where `rand_number` >= 0.5
                    )
                )
            ]
        ),
        append_payload=True,  # Also return a stored payload for found points
        top=5  # Return 5 closest points
    )

    print("Filtered search result (`random_num` >= 0.5):")
    for hit in hits:
        print(hit)


def test_points_crud():
    client = QdrantClient()

    client.recreate_collection(
        collection_name=COLLECTION_NAME,
        vector_size=DIM,
        distance=Distance.DOT
    )

    # Create a single point

    client.http.points_api.upsert_points(
        collection_name=COLLECTION_NAME,
        wait=True,
        point_insert_operations=PointsList(points=[
            PointStruct(
                id=123,
                payload={"test": "value"},
                vector=np.random.rand(DIM).tolist()
            )
        ])
    )

    # Read a single point

    points = client.http.points_api.get_points(COLLECTION_NAME, point_request=PointRequest(ids=[123]))

    print("read a single point", points)

    # Update a single point

    client.http.points_api.set_payload(
        collection_name=COLLECTION_NAME,
        set_payload=SetPayload(
            payload={
                "test2": ["value2", "value3"]
            },
            points=[123]
        )
    )

    # Delete a single point

    client.http.points_api.delete_points(
        collection_name=COLLECTION_NAME,
        points_selector=PointIdsList(points=[123])
    )


def test_has_id_condition():
    query = Filter(
        must=[
            HasIdCondition(has_id=[42, 43]),
            FieldCondition(key="field_name", match=MatchValue(value="field_value_42")),
        ]
    ).dict()

    assert query['must'][0]['has_id'] == [42, 43]


def test_insert_float():
    point = PointStruct(
        id=123,
        payload={'value': 0.123},
        vector=np.random.rand(DIM).tolist()
    )

    assert isinstance(point.payload['value'], float)


def test_legacy_imports():
    try:
        from qdrant_openapi_client.models.models import Filter, FieldCondition
        from qdrant_openapi_client.api.points_api import SyncPointsApi
        from qdrant_openapi_client.exceptions import UnexpectedResponse
    except ImportError:
        assert False  # can't import, fail


def test_value_serialization():
    v = json_to_value(123)
    print(v)


def test_serialization():
    from qdrant_client.grpc import PointStruct as PointStructGrpc
    from qdrant_client.grpc import PointId as PointIdGrpc

    point = PointStructGrpc(
        id=PointIdGrpc(num=1),
        vector=[1., 2., 3., 4.],
        payload=payload_to_grpc({
            "a": 123,
            "b": "text",
            "c": [1, 2, 3],
            "d": {
                "val1": "val2",
                "val2": [1, 2, 3],
            },
            "e": True,
            "f": None,
        })
    )
    print("\n")
    print(point.payload)
    data = point.SerializeToString()
    res = PointStructGrpc().parse(data)
    print(res.payload)
    print(grpc_to_payload(res.payload))


if __name__ == '__main__':
    test_qdrant_client_integration()
    test_points_crud()
    test_has_id_condition()
    test_insert_float()
    test_legacy_imports()
