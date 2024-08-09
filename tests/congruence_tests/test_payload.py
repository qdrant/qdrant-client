import datetime
import random
import uuid

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
)

NUM_VECTORS = 100


def upload(client_1: QdrantClient, client_2: QdrantClient, num_vectors=NUM_VECTORS):
    points = generate_fixtures(num_vectors)

    client_1.upload_points(COLLECTION_NAME, points, wait=True)
    client_2.upload_points(COLLECTION_NAME, points, wait=True)
    return points


def test_delete_payload(local_client: QdrantClient, remote_client: QdrantClient):
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


def test_clear_payload(local_client: QdrantClient, remote_client: QdrantClient):
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


def test_update_payload(local_client: QdrantClient, remote_client: QdrantClient):
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

    local_client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=vectors_config,
    )
    if remote_client.collection_exists(COLLECTION_NAME):
        remote_client.delete_collection(collection_name=COLLECTION_NAME)
    remote_client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=vectors_config,
    )

    # subset of types from pydantic.json.ENCODERS_BY_TYPE (pydantic v1)

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


def test_set_payload_with_key():
    local_client = init_local()
    remote_client = init_remote()

    vector_size = 2
    vectors_config = models.VectorParams(size=vector_size, distance=models.Distance.COSINE)

    local_client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=vectors_config,
    )
    if remote_client.collection_exists(COLLECTION_NAME):
        remote_client.delete_collection(collection_name=COLLECTION_NAME)
    remote_client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=vectors_config,
    )

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

    payload = {"nest": [{"a": "100", "b": "200"}]}
    new_payload = {"a": "101"}
    key = "nest[0]"
    set_payload(payload, new_payload, key)

    key = "nest[1]"
    new_payload = {"d": "404"}
    set_payload(payload, new_payload, key)

    key = "nest[].nest"
    set_payload(payload, new_payload, key)

    key = "nest[]"
    set_payload(payload, new_payload, key)

    payload = {}
    set_payload(payload, new_payload, key)

    payload = {"nest": [{"a": [], "b": "200"}]}
    new_payload = {"a": "101"}
    key = "nest[0].a[]"
    set_payload(payload, new_payload, key)

    payload = {"a": []}
    new_payload = {"b": {"c": 1}}
    key = "a[0]"
    set_payload(payload, new_payload, key)

    payload = {"a": {"b": {"c": {"d": {"e": 1}}}}}
    new_payload = {"f": 2}
    key = "a.b.c.d"
    set_payload(payload, new_payload, key)

    payload = {"a": []}
    new_payload = {}
    key = "a.b[0]"
    set_payload(payload, new_payload, key)

    payload = {"a": []}
    new_payload = {}
    key = "a.b"
    set_payload(payload, new_payload, key)

    payload = {"a": [[{"a": 1}]]}
    new_payload = {}
    key = "a[0][0]"
    set_payload(payload, new_payload, key)

    payload = {"a": [[{"a": "w"}]]}
    new_payload = {"b": "q"}
    key = "a[0][0]"
    set_payload(payload, new_payload, key)

    payload = {"a": []}
    new_payload = {}
    key = "a.b"
    set_payload(payload, new_payload, key)

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

    with pytest.raises(ValueError):
        local_client.set_payload(
            collection_name=COLLECTION_NAME,
            payload=new_payload,
            points=[9999],
            key=key,
        )
    with pytest.raises(UnexpectedResponse):
        remote_client.set_payload(
            collection_name=COLLECTION_NAME,
            payload=new_payload,
            points=[9999],
            key=key,
        )

    filter_ = models.Filter(
        must=[models.FieldCondition(key="", match=models.MatchValue(value="xc"))]
    )
    with pytest.raises(ValueError):
        local_client.set_payload(
            collection_name=COLLECTION_NAME, payload=new_payload, points=filter_
        )

    with pytest.raises(UnexpectedResponse):
        remote_client.set_payload(
            collection_name=COLLECTION_NAME, payload=new_payload, points=filter_
        )

    filter_ = models.Filter(
        must=[models.FieldCondition(key='""', match=models.MatchValue(value="xc"))]
    )

    remote_client.set_payload(collection_name=COLLECTION_NAME, payload=new_payload, points=filter_)
    local_client.set_payload(collection_name=COLLECTION_NAME, payload=new_payload, points=filter_)
    compare_collections(local_client, remote_client, 1)
