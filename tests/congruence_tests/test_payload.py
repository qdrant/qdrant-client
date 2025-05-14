import datetime
import random
import uuid

import grpc
import numpy as np
import pytest

from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import PointStruct
from qdrant_client.http.exceptions import UnexpectedResponse
from tests.congruence_tests.test_common import (
    COLLECTION_NAME,
    compare_collections,
    generate_fixtures,
    init_local,
    init_remote,
    text_vector_size,
    initialize_fixture_collection,
)

NUM_VECTORS = 100


def upload(client_1: QdrantClient, client_2: QdrantClient, num_vectors=NUM_VECTORS):
    points = generate_fixtures(num_vectors)

    client_1.upload_points(COLLECTION_NAME, points, wait=True)
    client_2.upload_points(COLLECTION_NAME, points, wait=True)
    return points


@pytest.mark.parametrize("prefer_grpc", [True, False])
def test_delete_payload(prefer_grpc):
    local_client: QdrantClient = init_local()
    initialize_fixture_collection(local_client)
    remote_client: QdrantClient = init_remote(prefer_grpc=prefer_grpc)
    initialize_fixture_collection(remote_client)

    points = upload(local_client, remote_client)

    # region delete one point
    id_ = points[0].id
    local_point = local_client.retrieve(COLLECTION_NAME, [id_])
    remote_point = remote_client.retrieve(COLLECTION_NAME, [id_])

    assert local_point == remote_point

    key = "text_data"
    local_client.delete_payload(COLLECTION_NAME, keys=[key], points=[id_])
    remote_client.delete_payload(COLLECTION_NAME, keys=[key], points=[id_], wait=True)

    assert local_client.retrieve(COLLECTION_NAME, [id_]) == remote_client.retrieve(
        COLLECTION_NAME, [id_]
    )
    # endregion

    # region delete multiple points
    keys_to_delete = ["rand_number", "text_array"]
    ids = [points[1].id, points[2].id]
    local_client.delete_payload(COLLECTION_NAME, keys=keys_to_delete, points=ids)
    remote_client.delete_payload(COLLECTION_NAME, keys=keys_to_delete, points=ids, wait=True)

    compare_collections(local_client, remote_client, NUM_VECTORS)
    # endregion

    # region delete by filter
    payload = points[2].payload
    key = "text_data"
    value = payload[key]
    delete_filter = models.Filter(
        must=[models.FieldCondition(key=key, match=models.MatchValue(value=value))]
    )

    local_client.delete_payload(COLLECTION_NAME, keys=["text_data"], points=delete_filter)
    remote_client.delete_payload(
        COLLECTION_NAME, keys=["text_data"], points=delete_filter, wait=True
    )

    compare_collections(local_client, remote_client, NUM_VECTORS)
    # endregion


@pytest.mark.parametrize("prefer_grpc", [True, False])
def test_clear_payload(prefer_grpc):
    local_client: QdrantClient = init_local()
    initialize_fixture_collection(local_client)
    remote_client: QdrantClient = init_remote(prefer_grpc=prefer_grpc)
    initialize_fixture_collection(remote_client)

    points = upload(local_client, remote_client)

    points_selector = [point.id for point in points[:5]]
    local_client.clear_payload(COLLECTION_NAME, points_selector)
    remote_client.clear_payload(COLLECTION_NAME, points_selector)

    compare_collections(local_client, remote_client, NUM_VECTORS)

    payload = points[42].payload
    key = "text_data"
    value = payload[key]
    points_selector = models.Filter(
        must=[models.FieldCondition(key=key, match=models.MatchValue(value=value))]
    )
    local_client.clear_payload(COLLECTION_NAME, points_selector)
    remote_client.clear_payload(COLLECTION_NAME, points_selector)

    compare_collections(local_client, remote_client, NUM_VECTORS)


@pytest.mark.parametrize("prefer_grpc", [True, False])
def test_update_payload(prefer_grpc):
    local_client: QdrantClient = init_local()
    initialize_fixture_collection(local_client)
    remote_client: QdrantClient = init_remote(prefer_grpc=prefer_grpc)
    initialize_fixture_collection(remote_client)
    points = upload(local_client, remote_client)

    # region fetch point
    id_ = points[0].id
    id_filter = models.Filter(must=[models.HasIdCondition(has_id=[id_])])
    local_point = local_client.scroll(
        COLLECTION_NAME,
        scroll_filter=id_filter,
        limit=1,
    )
    remote_point = remote_client.scroll(
        COLLECTION_NAME,
        scroll_filter=id_filter,
        limit=1,
    )

    assert local_point == remote_point
    # endregion

    # region set payload
    local_client.set_payload(COLLECTION_NAME, {"new_field": "new_value"}, id_filter)
    remote_client.set_payload(COLLECTION_NAME, {"new_field": "new_value"}, id_filter)

    local_new_point = local_client.scroll(
        COLLECTION_NAME,
        scroll_filter=id_filter,
        limit=1,
    )
    remote_new_point = remote_client.scroll(
        COLLECTION_NAME,
        scroll_filter=id_filter,
        limit=1,
    )

    assert local_new_point == remote_new_point
    # endregion

    # region overwrite payload
    local_client.overwrite_payload(COLLECTION_NAME, {"new_field": "overwritten_value"}, id_filter)
    remote_client.overwrite_payload(COLLECTION_NAME, {"new_field": "overwritten_value"}, id_filter)

    local_new_point = local_client.scroll(
        COLLECTION_NAME,
        scroll_filter=id_filter,
        limit=1,
    )
    remote_new_point = remote_client.scroll(
        COLLECTION_NAME,
        scroll_filter=id_filter,
        limit=1,
    )

    assert local_new_point == remote_new_point
    # endregion

    compare_collections(local_client, remote_client, NUM_VECTORS)  # sanity check


def test_not_jsonable_payload():
    local_client = init_local()
    remote_client = init_remote()

    vector_size = 2
    vectors_config = models.VectorParams(size=vector_size, distance=models.Distance.COSINE)

    initialize_fixture_collection(local_client, vectors_config=vectors_config)
    initialize_fixture_collection(remote_client, vectors_config=vectors_config)

    # subset of types from pydantic.json.ENCODERS_BY_TYPE (pydantic v1)
    # is not supported by grpc

    payloads = [
        {"bytes": b"123"},
        {"date": datetime.date(2021, 1, 1)},
        {"datetime": datetime.datetime(2021, 1, 1, 1, 1, 1)},
        {"time": datetime.time(1, 1, 1)},
        {"timedelta": datetime.timedelta(seconds=1)},
        {"decimal": 1.0},
        {"frozenset": frozenset([1, 2])},
        {"set": {1, 2}},
        {"uuid": uuid.uuid4()},
    ]

    points = [
        models.PointStruct(id=i, vector=[random.random(), random.random()], payload=payload)
        for i, payload in enumerate(payloads)
    ]

    for point in points:  # for better debugging
        local_client.upsert(COLLECTION_NAME, [point])
        remote_client.upsert(COLLECTION_NAME, [point])

    compare_collections(local_client, remote_client, len(points))

    local_client.delete_collection(collection_name=COLLECTION_NAME)
    local_client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=vectors_config,
    )
    remote_client.delete_collection(collection_name=COLLECTION_NAME)
    remote_client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=vectors_config,
    )

    point_ids = []
    for point in points:
        point.payload = None
        point_ids.append(point.id)
        local_client.upsert(COLLECTION_NAME, [point])
        remote_client.upsert(COLLECTION_NAME, [point])

    for point_id, payload in zip(point_ids, payloads):
        local_client.set_payload(
            COLLECTION_NAME,
            payload,
            models.Filter(must=[models.HasIdCondition(has_id=[point_id])]),
        )
        remote_client.set_payload(
            COLLECTION_NAME,
            payload,
            models.Filter(must=[models.HasIdCondition(has_id=[point_id])]),
        )

    compare_collections(local_client, remote_client, len(points))

    for point_id, payload in zip(point_ids[::-1], payloads):
        local_client.overwrite_payload(
            COLLECTION_NAME,
            payload,
            models.Filter(must=[models.HasIdCondition(has_id=[point_id])]),
        )
        remote_client.overwrite_payload(
            COLLECTION_NAME,
            payload,
            models.Filter(must=[models.HasIdCondition(has_id=[point_id])]),
        )

    compare_collections(local_client, remote_client, len(points))


@pytest.mark.parametrize("prefer_grpc", [True, False])
def test_set_payload_with_key(prefer_grpc):
    local_client = init_local()
    remote_client = init_remote(prefer_grpc=prefer_grpc)

    vector_size = 2
    vectors_config = models.VectorParams(size=vector_size, distance=models.Distance.COSINE)

    initialize_fixture_collection(local_client, vectors_config=vectors_config)
    initialize_fixture_collection(remote_client, vectors_config=vectors_config)

    vector = np.random.rand(vector_size).tolist()

    def set_payload(payload, new_payload, key):
        local_client.upsert(
            collection_name=COLLECTION_NAME,
            points=[
                PointStruct(
                    id=9999,
                    payload=payload,
                    vector=vector,
                ),
            ],
            wait=True,
        )
        remote_client.upsert(
            collection_name=COLLECTION_NAME,
            points=[
                PointStruct(
                    id=9999,
                    payload=payload,
                    vector=vector,
                ),
            ],
            wait=True,
        )

        local_client.set_payload(
            collection_name=COLLECTION_NAME,
            payload=new_payload,
            points=[9999],
            key=key,
        )
        remote_client.set_payload(
            collection_name=COLLECTION_NAME,
            payload=new_payload,
            points=[9999],
            key=key,
        )
        compare_collections(local_client, remote_client, 1)

    # update an existing field in nested array
    payload = {"nest": [{"a": "100", "b": "200"}]}
    new_payload = {"a": "101"}
    key = "nest[0]"
    set_payload(payload, new_payload, key)

    # can't modify a non-existing array element
    key = "nest[1]"
    new_payload = {"d": "404"}
    set_payload(payload, new_payload, key)

    # add new field to a dict in nested array
    key = "nest[].nest"
    set_payload(payload, new_payload, key)

    # add new field to a dict in nested array for all array elements
    key = "nest[]"
    set_payload(payload, new_payload, key)

    # add new key to an empty payload
    payload = {}
    set_payload(
        payload, new_payload, key
    )  # todo: uncomment when https://github.com/qdrant/qdrant/issues/6449 is resolved

    # can't add fields to an array
    payload = {"nest": [{"a": [], "b": "200"}]}
    new_payload = {"a": "101"}
    key = "nest[0].a[]"
    set_payload(payload, new_payload, key)

    # add key to a deeply nested dict
    payload = {"a": {"b": {"c": {"d": {"e": 1}}}}}
    new_payload = {"f": 2}
    key = "a.b.c.d"
    set_payload(payload, new_payload, key)

    # replace an array with a dict
    payload = {"a": []}
    new_payload = {}
    key = "a.b"
    set_payload(payload, new_payload, key)

    # replace an array with a dict of arrays
    payload = {"a": []}
    new_payload = {}
    key = "a.b[0]"
    set_payload(payload, new_payload, key)

    # can't replace a dict with an empty dict
    payload = {"a": [[{"a": 1}]]}
    new_payload = {}
    key = "a[0][0]"
    set_payload(payload, new_payload, key)

    # modify a dict in a deeply nested array
    payload = {"a": [[{"a": "w"}]]}
    new_payload = {"b": "q"}
    key = "a[0][0]"
    set_payload(payload, new_payload, key)

    # replace an array with an empty dict
    payload = {"a": []}
    new_payload = {}
    key = "a.b"
    set_payload(payload, new_payload, key)

    # replace a dict with a nested array
    payload = {"a": {"c": [{"d": 1}]}}
    new_payload = {"a": 1}
    key = "a.c[][]"
    set_payload(payload, new_payload, key)

    payload = {"": "xc"}
    new_payload = {"": "bbb"}
    key = ""
    local_client.upsert(
        collection_name=COLLECTION_NAME,
        points=[
            PointStruct(
                id=9999,
                payload=payload,
                vector=vector,
            ),
        ],
        wait=True,
    )
    remote_client.upsert(
        collection_name=COLLECTION_NAME,
        points=[
            PointStruct(
                id=9999,
                payload=payload,
                vector=vector,
            ),
        ],
        wait=True,
    )

    # region invalid path blank key
    with pytest.raises(ValueError):
        local_client.set_payload(
            collection_name=COLLECTION_NAME,
            payload=new_payload,
            points=[9999],
            key=key,
        )
    with pytest.raises((UnexpectedResponse, grpc.RpcError)):  # type: ignore
        remote_client.set_payload(
            collection_name=COLLECTION_NAME,
            payload=new_payload,
            points=[9999],
            key=key,
        )
    # endregion

    # region invalid path blank key in filter
    filter_ = models.Filter(
        must=[models.FieldCondition(key="", match=models.MatchValue(value="xc"))]
    )

    with pytest.raises(ValueError):
        local_client.set_payload(
            collection_name=COLLECTION_NAME, payload=new_payload, points=filter_
        )
    with pytest.raises((UnexpectedResponse, grpc.RpcError)):  # type: ignore
        remote_client.set_payload(
            collection_name=COLLECTION_NAME, payload=new_payload, points=filter_
        )
    # endregion

    # region correct way of setting payload for a blank key
    filter_ = models.Filter(
        must=[models.FieldCondition(key='""', match=models.MatchValue(value="xc"))]
    )

    remote_client.set_payload(collection_name=COLLECTION_NAME, payload=new_payload, points=filter_)
    local_client.set_payload(collection_name=COLLECTION_NAME, payload=new_payload, points=filter_)
    compare_collections(local_client, remote_client, 1)
    # endregion


@pytest.mark.parametrize("prefer_grpc", [True, False])
def test_upsert_operation(prefer_grpc):
    local_client: QdrantClient = init_local()
    initialize_fixture_collection(local_client)
    remote_client: QdrantClient = init_remote(prefer_grpc=prefer_grpc)
    initialize_fixture_collection(remote_client)

    def do_upsert_operation(client: QdrantClient, op: models.UpsertOperation):
        client.batch_update_points(collection_name=COLLECTION_NAME, update_operations=[op])

    vector = np.random.rand(text_vector_size).tolist()

    upsert_points_batch = models.UpsertOperation(
        upsert=models.PointsBatch(
            batch=models.Batch(
                ids=[
                    1,
                ],
                vectors={"text": [vector]},
                payloads=[{"key": "value"}],
            )
        )
    )
    do_upsert_operation(local_client, upsert_points_batch)
    do_upsert_operation(remote_client, upsert_points_batch)
    compare_collections(local_client, remote_client, 10)

    upsert_points_list = models.UpsertOperation(
        upsert=models.PointsList(
            points=[models.PointStruct(id=2, vector={"text": vector}, payload={"key": "value"})]
        )
    )
    do_upsert_operation(local_client, upsert_points_list)
    do_upsert_operation(remote_client, upsert_points_list)
    compare_collections(local_client, remote_client, 10)


@pytest.mark.parametrize("prefer_grpc", [True, False])
def test_delete_operation(prefer_grpc):
    local_client: QdrantClient = init_local()
    initialize_fixture_collection(local_client)
    remote_client: QdrantClient = init_remote(prefer_grpc=prefer_grpc)
    initialize_fixture_collection(remote_client)
    num_vectors = 100
    upload(local_client, remote_client, num_vectors)

    ids_to_delete = [1, 2, 3]
    op = models.DeleteOperation(delete=models.PointIdsList(points=ids_to_delete))
    local_client.batch_update_points(collection_name=COLLECTION_NAME, update_operations=[op])
    remote_client.batch_update_points(collection_name=COLLECTION_NAME, update_operations=[op])

    compare_collections(local_client, remote_client, num_vectors)
    assert local_client.count(collection_name=COLLECTION_NAME).count == (
        num_vectors - len(ids_to_delete)
    )


@pytest.mark.parametrize("prefer_grpc", [True, False])
def test_delete_and_clear_payload_operation(prefer_grpc):
    local_client: QdrantClient = init_local()
    remote_client: QdrantClient = init_remote(prefer_grpc=prefer_grpc)
    initialize_fixture_collection(
        local_client, vectors_config=models.VectorParams(size=2, distance=models.Distance.COSINE)
    )
    initialize_fixture_collection(
        remote_client, vectors_config=models.VectorParams(size=2, distance=models.Distance.COSINE)
    )

    points = [
        models.PointStruct(
            id=i,
            vector=[random.random(), random.random()],
            payload={"random_digit": random.randint(0, 9)},
        )
        for i in range(10)
    ]
    local_client.upsert(COLLECTION_NAME, points)
    remote_client.upsert(COLLECTION_NAME, points)

    ids_to_delete = [1, 2, 3]
    delete_payload_op_points = models.DeletePayloadOperation(
        delete_payload=models.DeletePayload(keys=["random_digit"], points=ids_to_delete)
    )
    local_client.batch_update_points(
        collection_name=COLLECTION_NAME, update_operations=[delete_payload_op_points]
    )
    remote_client.batch_update_points(
        collection_name=COLLECTION_NAME, update_operations=[delete_payload_op_points]
    )
    compare_collections(local_client, remote_client, 10)

    ids_to_delete = [4, 5, 6]
    delete_payload_op_filter = models.DeletePayloadOperation(
        delete_payload=models.DeletePayload(
            keys=["random_digit"],
            filter=models.Filter(must=[models.HasIdCondition(has_id=ids_to_delete)]),
        )
    )
    local_client.batch_update_points(
        collection_name=COLLECTION_NAME, update_operations=[delete_payload_op_filter]
    )
    remote_client.batch_update_points(
        collection_name=COLLECTION_NAME, update_operations=[delete_payload_op_filter]
    )
    compare_collections(local_client, remote_client, 10)

    ids_to_clear = [7, 8, 9]
    clear_payload_op = models.ClearPayloadOperation(
        clear_payload=models.PointIdsList(points=ids_to_clear)
    )
    local_client.batch_update_points(
        collection_name=COLLECTION_NAME, update_operations=[clear_payload_op]
    )
    remote_client.batch_update_points(
        collection_name=COLLECTION_NAME, update_operations=[clear_payload_op]
    )
    compare_collections(local_client, remote_client, 10)


@pytest.mark.parametrize("prefer_grpc", [True, False])
def test_update_or_delete_vectors_operation(prefer_grpc):
    local_client: QdrantClient = init_local()
    initialize_fixture_collection(local_client)
    remote_client: QdrantClient = init_remote(prefer_grpc=prefer_grpc)
    initialize_fixture_collection(remote_client)

    num_vectors = 10
    upload(local_client, remote_client, num_vectors)

    new_vector = {"text": np.random.randn(text_vector_size).round(3).tolist()}
    update_vectors_op = models.UpdateVectorsOperation(
        update_vectors=models.UpdateVectors(points=[models.PointVectors(id=1, vector=new_vector)])
    )

    local_client.batch_update_points(
        collection_name=COLLECTION_NAME, update_operations=[update_vectors_op]
    )
    remote_client.batch_update_points(
        collection_name=COLLECTION_NAME, update_operations=[update_vectors_op]
    )
    compare_collections(local_client, remote_client, num_vectors)

    ids_to_delete = [1, 2, 3]
    delete_vectors_op_points = models.DeleteVectorsOperation(
        delete_vectors=models.DeleteVectors(points=ids_to_delete, vector=["text"])
    )
    local_client.batch_update_points(
        collection_name=COLLECTION_NAME, update_operations=[delete_vectors_op_points]
    )
    remote_client.batch_update_points(
        collection_name=COLLECTION_NAME, update_operations=[delete_vectors_op_points]
    )
    compare_collections(local_client, remote_client, num_vectors)

    ids_to_delete = [4, 5, 6]
    delete_vectors_op_filter = models.DeleteVectorsOperation(
        delete_vectors=models.DeleteVectors(
            filter=models.Filter(must=[models.HasIdCondition(has_id=ids_to_delete)]),
            vector=["text"],
        )
    )
    local_client.batch_update_points(
        collection_name=COLLECTION_NAME, update_operations=[delete_vectors_op_filter]
    )
    remote_client.batch_update_points(
        collection_name=COLLECTION_NAME, update_operations=[delete_vectors_op_filter]
    )
    compare_collections(local_client, remote_client, num_vectors)


@pytest.mark.parametrize("prefer_grpc", [True, False])
def test_set_or_overwrite_payload_operation(prefer_grpc):
    local_client: QdrantClient = init_local()
    initialize_fixture_collection(local_client)
    remote_client: QdrantClient = init_remote(prefer_grpc=prefer_grpc)
    initialize_fixture_collection(remote_client)
    num_vectors = 10
    upload(local_client, remote_client, num_vectors)

    new_payload = {"text_data": "new_value"}
    set_payload_op_points_no_key = models.SetPayloadOperation(
        set_payload=models.SetPayload(points=[1], payload=new_payload)
    )

    local_client.batch_update_points(
        collection_name=COLLECTION_NAME, update_operations=[set_payload_op_points_no_key]
    )
    remote_client.batch_update_points(
        collection_name=COLLECTION_NAME, update_operations=[set_payload_op_points_no_key]
    )
    compare_collections(local_client, remote_client, num_vectors)

    new_nested_payload = {"nested_data": "new_nested_value"}
    set_payload_op_filter_key = models.SetPayloadOperation(
        set_payload=models.SetPayload(
            filter=models.Filter(must=[models.HasIdCondition(has_id=[1])]),
            payload=new_nested_payload,
            key="text_data",
        )
    )

    local_client.batch_update_points(
        collection_name=COLLECTION_NAME, update_operations=[set_payload_op_filter_key]
    )
    remote_client.batch_update_points(
        collection_name=COLLECTION_NAME, update_operations=[set_payload_op_filter_key]
    )
    compare_collections(local_client, remote_client, num_vectors)

    new_payload = {"text_data": {"some_key": "overwritten_value"}}
    overwrite_payload_op_points_no_key = models.OverwritePayloadOperation(
        overwrite_payload=models.SetPayload(points=[1], payload=new_payload)
    )

    local_client.batch_update_points(
        collection_name=COLLECTION_NAME, update_operations=[overwrite_payload_op_points_no_key]
    )
    remote_client.batch_update_points(
        collection_name=COLLECTION_NAME, update_operations=[overwrite_payload_op_points_no_key]
    )
    compare_collections(local_client, remote_client, num_vectors)
