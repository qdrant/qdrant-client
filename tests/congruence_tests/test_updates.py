import itertools
import math
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
from tests.fixtures.points import generate_points

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
    points = generate_fixtures(UPLOAD_NUM_VECTORS)

    vectors = []
    payload = []
    ids = []
    for point in points:
        ids.append(point.id),
        vectors.append(point.vector)
        payload.append(point.payload)

    local_client.upload_collection(COLLECTION_NAME, vectors, payload, ids=ids)
    remote_client.upload_collection(COLLECTION_NAME, vectors, payload, ids=ids)

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
    remote_client.upload_collection(COLLECTION_NAME, vectors, payload, ids=itertools.count())

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
    local_client.recreate_collection(
        COLLECTION_NAME, vectors_config=vectors_config, timeout=TIMEOUT
    )
    remote_client.recreate_collection(
        COLLECTION_NAME, vectors_config=vectors_config, timeout=TIMEOUT
    )

    ids = list(range(len(vectors)))
    local_client.upload_collection(COLLECTION_NAME, vectors, ids=ids)
    remote_client.upload_collection(COLLECTION_NAME, vectors, ids=ids)

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
    remote_client.upload_collection(COLLECTION_NAME, vectors, ids=ids)
    compare_collections(local_client, remote_client, UPLOAD_NUM_VECTORS)


def test_upload_collection_np_array_2d():
    vectors_dim = 50
    local_client = init_local()
    remote_client = init_remote()

    vectors = np.random.randn(UPLOAD_NUM_VECTORS, vectors_dim)
    ids = list(range(len(vectors)))
    vectors_config = models.VectorParams(size=vectors_dim, distance=models.Distance.EUCLID)
    local_client.recreate_collection(
        COLLECTION_NAME,
        vectors_config=vectors_config,
        timeout=TIMEOUT,
    )
    remote_client.recreate_collection(
        COLLECTION_NAME,
        vectors_config=vectors_config,
        timeout=TIMEOUT,
    )

    local_client.upload_collection(COLLECTION_NAME, vectors, ids=ids)
    remote_client.upload_collection(COLLECTION_NAME, vectors, ids=ids)

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
    local_client.recreate_collection(
        COLLECTION_NAME,
        vectors_config=vectors_config,
        timeout=TIMEOUT,
    )
    remote_client.recreate_collection(
        COLLECTION_NAME,
        vectors_config=vectors_config,
        timeout=TIMEOUT,
    )

    local_client.upload_collection(COLLECTION_NAME, vectors, ids=ids)
    remote_client.upload_collection(COLLECTION_NAME, vectors, ids=ids)

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
    remote_client.upload_collection(COLLECTION_NAME, vectors, ids=ids)
    compare_collections(local_client, remote_client, UPLOAD_NUM_VECTORS)


def test_upload_payload_contain_nan_values():
    # usual case when payload is extracted from pandas dataframe

    local_client = init_local()
    remote_client = init_remote()

    vector_size = 2
    nans_collection = "nans_collection"
    local_client.recreate_collection(
        collection_name=nans_collection,
        vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.DOT),
    )
    remote_client.recreate_collection(
        collection_name=nans_collection,
        vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.DOT),
    )
    points = generate_points(
        num_points=UPLOAD_NUM_VECTORS,
        vector_sizes=2,
        with_payload=False,
    )
    ids, vectors, payload = [], [], []
    for i in range(len(points)):
        points[i].payload = {"surprise": math.nan}

    for point in points:
        ids.append(point.id)
        vectors.append(point.vector)
        payload.append(point.payload)

    with pytest.raises(ValueError):
        local_client.upload_collection(nans_collection, vectors, payload)
    with pytest.raises(qdrant_client.http.exceptions.UnexpectedResponse):
        remote_client.upload_collection(nans_collection, vectors, payload)

    with pytest.raises(ValueError):
        local_client.upload_points(nans_collection, points)
    with pytest.raises(qdrant_client.http.exceptions.UnexpectedResponse):
        remote_client.upload_points(nans_collection, points)

    points_batch = models.Batch(
        ids=ids,
        vectors=vectors,
        payloads=payload,
    )

    with pytest.raises(ValueError):
        local_client.upsert(nans_collection, points=points_batch)
    with pytest.raises(qdrant_client.http.exceptions.UnexpectedResponse):
        remote_client.upsert(nans_collection, points=points_batch)

    local_client.delete_collection(nans_collection)
    remote_client.delete_collection(nans_collection)
