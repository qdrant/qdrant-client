import itertools
import uuid
from collections import defaultdict
from typing import Dict, List

import numpy as np
import pytest

import qdrant_client.http.exceptions
from qdrant_client.http import models
from tests.congruence_tests.settings import TIMEOUT
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
    points = generate_fixtures(UPLOAD_NUM_VECTORS)
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

    compare_collections(local_client, remote_client, UPLOAD_NUM_VECTORS)


def test_upload_collection(local_client, remote_client):
    points = generate_fixtures(UPLOAD_NUM_VECTORS)

    vectors = []
    payload = []
    ids = []
    for point in points:
        (ids.append(point.id),)
        vectors.append(point.vector)
        payload.append(point.payload)

    local_client.upload_collection(COLLECTION_NAME, vectors, payload, ids=ids)
    remote_client.upload_collection(COLLECTION_NAME, vectors, payload, ids=ids, wait=True)

    compare_collections(local_client, remote_client, UPLOAD_NUM_VECTORS)


@pytest.mark.timeout(60)  # normally takes less than a second
def test_upload_collection_generators(local_client, remote_client):
    points = generate_fixtures(UPLOAD_NUM_VECTORS)
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

    compare_collections(local_client, remote_client, UPLOAD_NUM_VECTORS)


def test_upload_points(local_client, remote_client):
    points = generate_fixtures(UPLOAD_NUM_VECTORS)

    local_client.upload_points(COLLECTION_NAME, points)
    remote_client.upload_points(COLLECTION_NAME, points, wait=True)

    compare_collections(local_client, remote_client, UPLOAD_NUM_VECTORS)


def test_upload_uuid_in_batches(local_client, remote_client):
    points = generate_fixtures(UPLOAD_NUM_VECTORS)
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

    compare_collections(local_client, remote_client, UPLOAD_NUM_VECTORS)


def test_upload_collection_float_list():
    vectors_dim = 50
    local_client = init_local()
    remote_client = init_remote()

    vectors = np.random.randn(UPLOAD_NUM_VECTORS, vectors_dim).tolist()
    vectors_config = models.VectorParams(size=vectors_dim, distance=models.Distance.EUCLID)

    local_client.create_collection(COLLECTION_NAME, vectors_config=vectors_config, timeout=TIMEOUT)
    if remote_client.collection_exists(COLLECTION_NAME):
        remote_client.delete_collection(COLLECTION_NAME, timeout=TIMEOUT)
    remote_client.create_collection(
        COLLECTION_NAME, vectors_config=vectors_config, timeout=TIMEOUT
    )

    ids = list(range(len(vectors)))
    local_client.upload_collection(COLLECTION_NAME, vectors, ids=ids)
    remote_client.upload_collection(COLLECTION_NAME, vectors, ids=ids, wait=True)

    compare_collections(local_client, remote_client, UPLOAD_NUM_VECTORS)
    local_client.delete_collection(COLLECTION_NAME)
    remote_client.delete_collection(COLLECTION_NAME)


def test_upload_collection_named_float_list_vectors(local_client, remote_client):
    points = generate_fixtures(UPLOAD_NUM_VECTORS)
    vectors = []  # List[Dict[str, float]]
    for point in points:
        vectors.append(point.vector)
    ids = [point.id for point in points]
    local_client.upload_collection(COLLECTION_NAME, vectors, ids=ids)
    remote_client.upload_collection(COLLECTION_NAME, vectors, ids=ids, wait=True)
    compare_collections(local_client, remote_client, UPLOAD_NUM_VECTORS)


def test_upload_collection_np_array_2d():
    vectors_dim = 50
    local_client = init_local()
    remote_client = init_remote()

    vectors = np.random.randn(UPLOAD_NUM_VECTORS, vectors_dim)
    ids = list(range(len(vectors)))
    vectors_config = models.VectorParams(size=vectors_dim, distance=models.Distance.EUCLID)

    local_client.create_collection(
        COLLECTION_NAME,
        vectors_config=vectors_config,
        timeout=TIMEOUT,
    )
    if remote_client.collection_exists(COLLECTION_NAME):
        remote_client.delete_collection(COLLECTION_NAME, timeout=TIMEOUT)
    remote_client.create_collection(
        COLLECTION_NAME,
        vectors_config=vectors_config,
        timeout=TIMEOUT,
    )

    local_client.upload_collection(COLLECTION_NAME, vectors, ids=ids)
    remote_client.upload_collection(COLLECTION_NAME, vectors, ids=ids, wait=True)

    compare_collections(local_client, remote_client, UPLOAD_NUM_VECTORS)
    local_client.delete_collection(COLLECTION_NAME)
    remote_client.delete_collection(COLLECTION_NAME)


def test_upload_collection_list_np_arrays():
    vectors_dim = 50
    local_client = init_local()
    remote_client = init_remote()

    vectors = np.random.randn(UPLOAD_NUM_VECTORS, vectors_dim).tolist()
    vectors = [np.array(vector) for vector in vectors]
    vectors_config = models.VectorParams(size=vectors_dim, distance=models.Distance.EUCLID)
    ids = list(range(len(vectors)))

    local_client.create_collection(
        COLLECTION_NAME,
        vectors_config=vectors_config,
        timeout=TIMEOUT,
    )
    if remote_client.collection_exists(COLLECTION_NAME):
        remote_client.delete_collection(COLLECTION_NAME, timeout=TIMEOUT)
    remote_client.create_collection(
        COLLECTION_NAME,
        vectors_config=vectors_config,
        timeout=TIMEOUT,
    )

    local_client.upload_collection(COLLECTION_NAME, vectors, ids=ids)
    remote_client.upload_collection(COLLECTION_NAME, vectors, ids=ids, wait=True)

    compare_collections(local_client, remote_client, UPLOAD_NUM_VECTORS)
    local_client.delete_collection(COLLECTION_NAME)
    remote_client.delete_collection(COLLECTION_NAME)


def test_upload_collection_dict_np_arrays(local_client, remote_client):
    points = generate_fixtures(UPLOAD_NUM_VECTORS)
    intermediate_vectors: Dict[str, List[float]] = defaultdict(list)
    vectors: Dict[str, np.ndarray] = {}
    ids = [point.id for point in points]
    for point in points:
        for key, vector in point.vector.items():
            intermediate_vectors[key].append(point.vector[key])

    for key in intermediate_vectors:
        vectors[key] = np.array(intermediate_vectors[key])

    local_client.upload_collection(COLLECTION_NAME, vectors, ids=ids)
    remote_client.upload_collection(COLLECTION_NAME, vectors, ids=ids, wait=True)
    compare_collections(local_client, remote_client, UPLOAD_NUM_VECTORS)


def test_upload_wrong_vectors():
    local_client = init_local()
    remote_client = init_remote()

    vector_size = 2
    wrong_vectors_collection = "test_collection"
    vectors_config = {
        "text": models.VectorParams(size=vector_size, distance=models.Distance.COSINE)
    }
    sparse_vectors_config = {"text-sparse": models.SparseVectorParams()}

    local_client.create_collection(
        collection_name=wrong_vectors_collection,
        vectors_config=vectors_config,
        sparse_vectors_config=sparse_vectors_config,
    )
    if remote_client.collection_exists(collection_name=wrong_vectors_collection):
        remote_client.delete_collection(collection_name=wrong_vectors_collection)
    remote_client.create_collection(
        collection_name=wrong_vectors_collection,
        vectors_config=vectors_config,
        sparse_vectors_config=sparse_vectors_config,
    )

    dense_vector = {"why_am_I_so_dense": [0.1, 0.3]}
    dense_vectors = {"why_am_I_so_dense": [[0.1, 0.3]]}
    sparse_vector = {"why_am_I_so_sparse": models.SparseVector(indices=[0, 1], values=[0.5, 0.6])}
    sparse_vectors = {
        "why_am_I_so_sparse": [models.SparseVector(indices=[0, 2], values=[0.3, 0.4])]
    }

    list_points = [models.PointStruct(id=1, vector=dense_vector)]
    batch = models.Batch(ids=[2], vectors=dense_vectors)
    list_points_sparse = [models.PointStruct(id=1, vector=sparse_vector)]
    batch_sparse = models.Batch(ids=[2], vectors=sparse_vectors)

    for points in (list_points, list_points_sparse, batch, batch_sparse):
        with pytest.raises(qdrant_client.http.exceptions.UnexpectedResponse):
            remote_client.upsert(wrong_vectors_collection, points)

        with pytest.raises(ValueError):
            local_client.upsert(wrong_vectors_collection, points)

    for vector in (dense_vector, sparse_vector):
        # does not raise without wait=True
        with pytest.raises(qdrant_client.http.exceptions.UnexpectedResponse):
            remote_client.upload_collection(wrong_vectors_collection, vectors=[vector], wait=True)

        with pytest.raises(ValueError):
            local_client.upload_collection(wrong_vectors_collection, vectors=[vector])

        # does not raise without wait=True
        with pytest.raises(qdrant_client.http.exceptions.UnexpectedResponse):
            remote_client.upload_records(
                wrong_vectors_collection,
                records=[models.Record(id=3, vector=dense_vector)],
                wait=True,
            )

        with pytest.raises(ValueError):
            local_client.upload_records(
                wrong_vectors_collection, records=[models.Record(id=3, vector=dense_vector)]
            )

    unnamed_vector = [0.1, 0.3]
    with pytest.raises(qdrant_client.http.exceptions.UnexpectedResponse):
        remote_client.upsert(
            wrong_vectors_collection,
            points=[models.PointStruct(id=1, vector=unnamed_vector)],
        )
    with pytest.raises(ValueError):
        local_client.upsert(
            wrong_vectors_collection,
            points=[models.PointStruct(id=1, vector=unnamed_vector)],
        )
