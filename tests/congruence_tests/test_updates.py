import itertools
import uuid
from collections import defaultdict
from typing import Dict, List

import numpy as np
import pytest

from qdrant_client.http import models
from tests.congruence_tests.test_common import (
    COLLECTION_NAME,
    compare_collections,
    generate_fixtures,
    init_local,
    init_remote,
)
from tests.fixtures.payload import one_random_payload_please

UPLOAD_NUM_VECTORS = 100


def test_upsert(local_client, remote_client):
    # region upload data
    records = generate_fixtures(UPLOAD_NUM_VECTORS)
    ids, payload = [], []
    vectors = {}
    for record in records:
        ids.append(record.id)
        payload.append(record.payload)
        for vector_name, vector in record.vector.items():
            if vector_name not in vectors:
                vectors[vector_name] = []
            vectors[vector_name].append(vector)

    points = models.Batch(
        ids=ids,
        vectors=vectors,
        payloads=payload,
    )

    local_client.upsert(COLLECTION_NAME, points)
    remote_client.upsert(COLLECTION_NAME, points)

    id_ = ids[0]
    vector = {k: v[0] for k, v in vectors.items()}
    old_payload = payload[0]

    id_filter = models.Filter(must=[models.HasIdCondition(has_id=[id_])])

    local_old_point = local_client.scroll(
        COLLECTION_NAME,
        scroll_filter=id_filter,
        limit=1,
    )[
        0
    ][0]
    remote_old_point = remote_client.scroll(COLLECTION_NAME, scroll_filter=id_filter, limit=1)[0][
        0
    ]

    assert local_old_point == remote_old_point
    # endregion

    # region update point
    new_payload = one_random_payload_please(id_)
    assert old_payload != new_payload

    local_client.upsert(
        COLLECTION_NAME,
        [models.PointStruct(id=id_, vector=vector, payload=new_payload)],
    )
    remote_client.upsert(
        COLLECTION_NAME,
        [models.PointStruct(id=id_, vector=vector, payload=new_payload)],
    )

    local_new_point = local_client.scroll(COLLECTION_NAME, scroll_filter=id_filter, limit=1)[0][0]
    remote_new_point = remote_client.scroll(COLLECTION_NAME, scroll_filter=id_filter, limit=1)[0][
        0
    ]

    assert local_new_point == remote_new_point
    # endregion

    compare_collections(local_client, remote_client, UPLOAD_NUM_VECTORS)


def test_upload_collection(local_client, remote_client):
    records = generate_fixtures(UPLOAD_NUM_VECTORS)

    vectors = []
    payload = []
    for record in records:
        vectors.append(record.vector)
        payload.append(record.payload)

    local_client.upload_collection(COLLECTION_NAME, vectors, payload)
    remote_client.upload_collection(COLLECTION_NAME, vectors, payload)

    compare_collections(local_client, remote_client, UPLOAD_NUM_VECTORS)


@pytest.mark.timeout(15)  # normally takes less than a second
def test_upload_collection_generators(local_client, remote_client):
    records = generate_fixtures(UPLOAD_NUM_VECTORS)
    vectors = []
    payload = []
    for record in records:
        vectors.append(record.vector)
        payload.append(record.payload)

    payload = itertools.cycle(payload)
    local_client.upload_collection(COLLECTION_NAME, vectors, payload, ids=itertools.count())
    remote_client.upload_collection(COLLECTION_NAME, vectors, payload, ids=itertools.count())

    compare_collections(local_client, remote_client, UPLOAD_NUM_VECTORS)


def test_upload_records(local_client, remote_client):
    records = generate_fixtures(UPLOAD_NUM_VECTORS)

    local_client.upload_records(COLLECTION_NAME, records)
    remote_client.upload_records(COLLECTION_NAME, records)

    compare_collections(local_client, remote_client, UPLOAD_NUM_VECTORS)


def test_upload_uuid_in_batches(local_client, remote_client):
    records = generate_fixtures(UPLOAD_NUM_VECTORS)
    vectors = defaultdict(list)

    for record in records:
        for vector_name, vector in record.vector.items():
            vectors[vector_name].append(vector)

    batch = models.Batch(
        ids=[str(uuid.uuid4()) for _ in records],
        vectors=vectors,
        payloads=[record.payload for record in records],
    )

    local_client.upsert(COLLECTION_NAME, batch)
    remote_client.upsert(COLLECTION_NAME, batch)

    compare_collections(local_client, remote_client, UPLOAD_NUM_VECTORS)


def test_upload_collection_float_list():
    vectors_dim = 50
    local_client = init_local()
    remote_client = init_remote()

    vectors = np.random.randn(UPLOAD_NUM_VECTORS, vectors_dim).tolist()
    vectors_config = models.VectorParams(size=vectors_dim, distance=models.Distance.EUCLID)
    local_client.recreate_collection(COLLECTION_NAME, vectors_config=vectors_config)
    remote_client.recreate_collection(COLLECTION_NAME, vectors_config=vectors_config)

    local_client.upload_collection(COLLECTION_NAME, vectors)
    remote_client.upload_collection(COLLECTION_NAME, vectors)

    compare_collections(local_client, remote_client, UPLOAD_NUM_VECTORS)


def test_upload_collection_named_float_list_vectors(local_client, remote_client):
    records = generate_fixtures(UPLOAD_NUM_VECTORS)
    vectors: List[Dict[str, float]] = []
    for record in records:
        vectors.append(record.vector)

    local_client.upload_collection(COLLECTION_NAME, vectors)
    remote_client.upload_collection(COLLECTION_NAME, vectors)
    compare_collections(local_client, remote_client, UPLOAD_NUM_VECTORS)


def test_upload_collection_np_array_2d():
    vectors_dim = 50
    local_client = init_local()
    remote_client = init_remote()

    vectors = np.random.randn(UPLOAD_NUM_VECTORS, vectors_dim)
    vectors_config = models.VectorParams(size=vectors_dim, distance=models.Distance.EUCLID)
    local_client.recreate_collection(COLLECTION_NAME, vectors_config=vectors_config)
    remote_client.recreate_collection(COLLECTION_NAME, vectors_config=vectors_config)

    local_client.upload_collection(COLLECTION_NAME, vectors)
    remote_client.upload_collection(COLLECTION_NAME, vectors)

    compare_collections(local_client, remote_client, UPLOAD_NUM_VECTORS)


def test_upload_collection_list_np_arrays():
    vectors_dim = 50
    local_client = init_local()
    remote_client = init_remote()

    vectors = np.random.randn(UPLOAD_NUM_VECTORS, vectors_dim).tolist()
    vectors = [np.array(vector) for vector in vectors]
    vectors_config = models.VectorParams(size=vectors_dim, distance=models.Distance.EUCLID)
    local_client.recreate_collection(COLLECTION_NAME, vectors_config=vectors_config)
    remote_client.recreate_collection(COLLECTION_NAME, vectors_config=vectors_config)

    local_client.upload_collection(COLLECTION_NAME, vectors)
    remote_client.upload_collection(COLLECTION_NAME, vectors)

    compare_collections(local_client, remote_client, UPLOAD_NUM_VECTORS)
