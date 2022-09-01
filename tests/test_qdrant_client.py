import os
import random
import uuid
from pprint import pprint
from tempfile import mkdtemp
from time import sleep

import numpy as np
import pytest

from qdrant_client import QdrantClient
from qdrant_client.conversions.common_types import Record
from qdrant_client.conversions.conversion import grpc_to_payload, json_to_value
from qdrant_client.http.models import Filter, FieldCondition, Range, PointStruct, HasIdCondition, PointIdsList, \
    PayloadSchemaType, MatchValue, Distance, CreateAliasOperation, CreateAlias, OptimizersConfigDiff
from qdrant_client.uploader.grpc_uploader import payload_to_grpc

DIM = 100
NUM_VECTORS = 1_000
COLLECTION_NAME = 'client_test'
COLLECTION_NAME_ALIAS = 'client_test_alias'

def one_random_payload_please(idx):
    return {
        "id": idx + 100,
        "id_str": [str(idx + 100), str(idx + 1000)],
        "text_data": uuid.uuid4().hex,
        "rand_number": random.random(),
        "text_array": [uuid.uuid4().hex, uuid.uuid4().hex]
    }

def random_payload():
    for i in range(NUM_VECTORS):
        yield one_random_payload_please(i)


def create_random_vectors():
    vectors_path = os.path.join(mkdtemp(), 'vectors.npy')
    fp = np.memmap(vectors_path, dtype='float32', mode='w+', shape=(NUM_VECTORS, DIM))

    data = np.random.rand(NUM_VECTORS, DIM).astype(np.float32)
    fp[:] = data[:]
    fp.flush()
    return vectors_path


@pytest.mark.parametrize("prefer_grpc", [False, True])
def test_record_upload(prefer_grpc):
    records = (
        Record(
            id=idx,
            vector=np.random.rand(DIM).tolist(),
            payload=one_random_payload_please(idx)
        )
        for idx in range(NUM_VECTORS)
    )

    client = QdrantClient(prefer_grpc=prefer_grpc)

    client.recreate_collection(
        collection_name=COLLECTION_NAME,
        vector_size=DIM,
        distance=Distance.DOT,
    )

    client.upload_records(
        collection_name=COLLECTION_NAME,
        records=records,
        parallel=2
    )

    # By default, Qdrant indexes data updates asynchronously, so client don't need to wait before sending next batch
    # Let's give it a second to actually add all points to a collection.
    # If you need to change this behaviour - simply enable synchronous processing by enabling `wait=true`
    sleep(1)

    collection_info = client.get_collection(collection_name=COLLECTION_NAME)

    assert collection_info.points_count == NUM_VECTORS

    result_count = client.count(
        COLLECTION_NAME,
        count_filter=Filter(
            must=[
                FieldCondition(
                    key='rand_number',  # Condition based on values of `rand_number` field.
                    range=Range(
                        gte=0.5  # Select only those results where `rand_number` >= 0.5
                    )
                )
            ]
        )
    )

    assert result_count.count < 900
    assert result_count.count > 100



@pytest.mark.parametrize("prefer_grpc", [False, True])
@pytest.mark.parametrize("numpy_upload", [False, True])
def test_qdrant_client_integration(prefer_grpc, numpy_upload):
    vectors_path = create_random_vectors()

    if numpy_upload:
        vectors = np.memmap(vectors_path, dtype='float32', mode='r', shape=(NUM_VECTORS, DIM))
        vectors_2 = vectors[2].tolist()
    else:
        vectors = [
            np.random.rand(DIM).tolist()
            for _ in range(NUM_VECTORS)
        ]
        vectors_2 = vectors[2]

    payload = random_payload()

    client = QdrantClient(prefer_grpc=prefer_grpc)

    client.recreate_collection(
        collection_name=COLLECTION_NAME,
        vector_size=DIM,
        distance=Distance.DOT,
    )

    # Call Qdrant API to retrieve list of existing collections
    collections = client.get_collections().collections

    # Print all existing collections
    for collection in collections:
        print(collection.dict())

    # Retrieve detailed information about newly created collection
    test_collection = client.get_collection(COLLECTION_NAME)
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

    result_count = client.count(
        COLLECTION_NAME,
        count_filter=Filter(
            must=[
                FieldCondition(
                    key='rand_number',  # Condition based on values of `rand_number` field.
                    range=Range(
                        gte=0.5  # Select only those results where `rand_number` >= 0.5
                    )
                )
            ]
        )
    )

    assert result_count.count < 900
    assert result_count.count > 100

    client.update_collection_aliases(
        change_aliases_operations=[
            CreateAliasOperation(
                create_alias=CreateAlias(
                    collection_name=COLLECTION_NAME,
                    alias_name=COLLECTION_NAME_ALIAS
                )
            )
        ]
    )

    # Create payload index for field `random_num`
    # If indexed field appear in filtering condition - search operation could be performed faster
    index_create_result = client.create_payload_index(COLLECTION_NAME, "random_num", PayloadSchemaType.FLOAT)
    pprint(index_create_result.dict())

    # Let's now check details about our new collection
    test_collection = client.get_collection(COLLECTION_NAME_ALIAS)
    pprint(test_collection.dict())

    # Now we can actually search in the collection
    # Let's create some random vector
    query_vector = np.random.rand(DIM)

    #  and use it as a query
    hits = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vector,
        query_filter=None,  # Don't use any filters for now, search across all indexed points
        with_payload=True,  # Also return a stored payload for found points
        limit=5  # Return 5 closest points
    )

    assert len(hits) == 5

    # Print found results
    print("Search result:")
    for hit in hits:
        print(hit)

    client.create_payload_index(COLLECTION_NAME, "id_str", PayloadSchemaType.KEYWORD)
    #  and use it as a query
    hits = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vector,
        query_filter=Filter(
            must=[
                FieldCondition(
                    key="id_str",
                    match=MatchValue(value="123")
                )
            ]
        ),
        with_payload=True,
        limit=5
    )

    assert len(hits) == 1
    assert (hits[0].payload['id_str'] == '123') or ('123' in hits[0].payload['id_str'])

    client.update_collection(
        collection_name=COLLECTION_NAME,
        optimizer_config=OptimizersConfigDiff(
            max_segment_size=10000
        )
    )

    assert client.get_collection(COLLECTION_NAME).config.optimizer_config.max_segment_size == 10000

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
        limit=5  # Return 5 closest points
    )

    print("Filtered search result (`random_num` >= 0.5):")
    for hit in hits:
        print(hit)

    got_points = client.retrieve(
        collection_name=COLLECTION_NAME,
        ids=[1, 2, 3],
        with_payload=True,
        with_vector=True
    )

    assert len(got_points) == 3

    client.delete(
        collection_name=COLLECTION_NAME,
        wait=True,
        points_selector=PointIdsList(points=[2, 3])
    )

    got_points = client.retrieve(
        collection_name=COLLECTION_NAME,
        ids=[1, 2, 3],
        with_payload=True,
        with_vector=True
    )

    assert len(got_points) == 1

    client.upsert(
        collection_name=COLLECTION_NAME,
        wait=True,
        points=[PointStruct(id=2, payload={"hello": "world"}, vector=vectors_2)]
    )

    got_points = client.retrieve(
        collection_name=COLLECTION_NAME,
        ids=[1, 2, 3],
        with_payload=True,
        with_vector=True
    )

    assert len(got_points) == 2

    client.set_payload(
        collection_name=COLLECTION_NAME,
        payload={
            "new_key": 123
        },
        points=[1, 2],
        wait=True
    )

    got_points = client.retrieve(
        collection_name=COLLECTION_NAME,
        ids=[1, 2],
        with_payload=True,
        with_vector=True
    )

    for point in got_points:
        assert point.payload.get("new_key") == 123

    client.delete_payload(
        collection_name=COLLECTION_NAME,
        keys=["new_key"],
        points=[1],
    )

    got_points = client.retrieve(
        collection_name=COLLECTION_NAME,
        ids=[1],
        with_payload=True,
        with_vector=True
    )

    for point in got_points:
        assert "new_key" not in point.payload

    client.clear_payload(
        collection_name=COLLECTION_NAME,
        points_selector=PointIdsList(points=[1, 2]),
    )

    got_points = client.retrieve(
        collection_name=COLLECTION_NAME,
        ids=[1, 2],
        with_payload=True,
        with_vector=True
    )

    for point in got_points:
        assert not point.payload

    recommended_points = client.recommend(
        collection_name=COLLECTION_NAME,
        positive=[1, 2],
        query_filter=Filter(
            must=[  # These conditions are required for recommend results
                FieldCondition(
                    key='rand_number',  # Condition based on values of `rand_number` field.
                    range=Range(
                        lte=0.5  # Select only those results where `rand_number` >= 0.5
                    )
                )
            ]
        ),
        limit=5,
        with_payload=True,
        with_vector=False
    )

    assert len(recommended_points) == 5

    scrolled_points, next_page = client.scroll(
        collection_name=COLLECTION_NAME,
        scroll_filter=Filter(
            must=[  # These conditions are required for scroll results
                FieldCondition(
                    key='rand_number',  # Condition based on values of `rand_number` field.
                    range=Range(
                        lte=0.5  # Return only those results where `rand_number` <= 0.5
                    )
                )
            ]
        ),
        limit=5,
        offset=None,
        with_payload=True,
        with_vector=False
    )

    assert len(scrolled_points) == 5


@pytest.mark.parametrize(
    "prefer_grpc", [False, True]
)
def test_points_crud(prefer_grpc):
    client = QdrantClient(prefer_grpc=prefer_grpc)

    client.recreate_collection(
        collection_name=COLLECTION_NAME,
        vector_size=DIM,
        distance=Distance.DOT
    )

    # Create a single point
    client.upsert(
        collection_name=COLLECTION_NAME,
        points=[
            PointStruct(
                id=123,
                payload={"test": "value"},
                vector=np.random.rand(DIM).tolist()
            )
        ],
        wait=True,
    )

    # Read a single point

    points = client.retrieve(COLLECTION_NAME, ids=[123])

    print("read a single point", points)

    # Update a single point

    client.set_payload(
        collection_name=COLLECTION_NAME,
        payload={
            "test2": ["value2", "value3"]
        },
        points=[123]
    )

    # Delete a single point
    client.delete(
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
