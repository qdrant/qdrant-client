import itertools
import uuid
from collections import defaultdict

import pytest

from qdrant_client.http import models
from tests.congruence_tests.test_common import (
    COLLECTION_NAME,
    compare_collections,
    generate_sparse_fixtures,
    init_client,
    init_local,
    init_remote,
    sparse_vectors_config,
)
from tests.fixtures.payload import one_random_payload_please

UPLOAD_NUM_VECTORS = 100


def test_upsert():
    # region upload data
    points = generate_sparse_fixtures(UPLOAD_NUM_VECTORS)
    local_client = init_local()
    init_client(local_client, points, sparse_vectors_config=sparse_vectors_config)

    remote_client = init_remote()
    init_client(remote_client, points, sparse_vectors_config=sparse_vectors_config)

    ids, payload = [], []
    vectors = {}
    for point in points:
        ids.append(point.id)
        payload.append(point.payload)
        for vector_name, vector in point.vector.items():
            if vector_name not in vectors:
                vectors[vector_name] = []
            vectors[vector_name].append(vector)

    points_batch = models.Batch(
        ids=ids,
        vectors=vectors,
        payloads=payload,
    )

    local_client.upsert(COLLECTION_NAME, points_batch)
    remote_client.upsert(COLLECTION_NAME, points_batch)

    id_ = ids[0]
    vector = {k: v[0] for k, v in vectors.items()}
    old_payload = payload[0]

    id_filter = models.Filter(must=[models.HasIdCondition(has_id=[id_])])

    local_old_point = local_client.scroll(
        COLLECTION_NAME,
        scroll_filter=id_filter,
        limit=1,
    )[0][0]
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

    compare_collections(
        local_client,
        remote_client,
        UPLOAD_NUM_VECTORS,
        attrs=("points_count",),
    )


def test_upload_collection():
    points = generate_sparse_fixtures(UPLOAD_NUM_VECTORS)

    local_client = init_local()
    init_client(local_client, points, sparse_vectors_config=sparse_vectors_config)

    remote_client = init_remote()
    init_client(remote_client, points, sparse_vectors_config=sparse_vectors_config)

    vectors = []
    payload = []
    for point in points:
        vectors.append(point.vector)
        payload.append(point.payload)

    ids = list(range(len(vectors)))
    local_client.upload_collection(COLLECTION_NAME, vectors, payload, ids=ids)
    remote_client.upload_collection(COLLECTION_NAME, vectors, payload, ids=ids, wait=True)

    compare_collections(
        local_client,
        remote_client,
        UPLOAD_NUM_VECTORS,
        attrs=("points_count",),
    )


@pytest.mark.timeout(60)  # normally takes less than a second
def test_upload_collection_generators():
    points = generate_sparse_fixtures(UPLOAD_NUM_VECTORS)

    local_client = init_local()
    init_client(local_client, points, sparse_vectors_config=sparse_vectors_config)

    remote_client = init_remote()
    init_client(remote_client, points, sparse_vectors_config=sparse_vectors_config)

    vectors = []
    payload = []
    for point in points:
        vectors.append(point.vector)
        payload.append(point.payload)

    payload = itertools.cycle(payload)
    local_client.upload_collection(COLLECTION_NAME, vectors, payload, ids=itertools.count())
    remote_client.upload_collection(
        COLLECTION_NAME, vectors, payload, ids=itertools.count(), wait=True
    )

    compare_collections(
        local_client,
        remote_client,
        UPLOAD_NUM_VECTORS,
        attrs=("points_count",),
    )


def test_upload_points():
    points = generate_sparse_fixtures(UPLOAD_NUM_VECTORS)

    local_client = init_local()
    init_client(local_client, points, sparse_vectors_config=sparse_vectors_config)

    remote_client = init_remote()
    init_client(remote_client, points, sparse_vectors_config=sparse_vectors_config)

    local_client.upload_points(COLLECTION_NAME, points)
    remote_client.upload_points(COLLECTION_NAME, points, wait=True)

    compare_collections(
        local_client,
        remote_client,
        UPLOAD_NUM_VECTORS,
        attrs=("points_count",),
    )


def test_upload_uuid_in_batches():
    points = generate_sparse_fixtures(UPLOAD_NUM_VECTORS)

    local_client = init_local()
    init_client(local_client, points, sparse_vectors_config=sparse_vectors_config)

    remote_client = init_remote()
    init_client(remote_client, points, sparse_vectors_config=sparse_vectors_config)

    vectors = defaultdict(list)

    for point in points:
        for vector_name, vector in point.vector.items():
            vectors[vector_name].append(vector)

    batch = models.Batch(
        ids=[str(uuid.uuid4()) for _ in points],
        vectors=vectors,
        payloads=[point.payload for point in points],
    )

    local_client.upsert(COLLECTION_NAME, batch)
    remote_client.upsert(COLLECTION_NAME, batch)

    compare_collections(
        local_client,
        remote_client,
        UPLOAD_NUM_VECTORS,
        attrs=("points_count",),
    )
