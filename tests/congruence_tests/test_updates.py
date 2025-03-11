import itertools
import uuid
from collections import defaultdict

import numpy as np
import pytest

from qdrant_client.client_base import QdrantBase
import qdrant_client.http.exceptions
from qdrant_client.http import models
from tests.congruence_tests.settings import TIMEOUT
from tests.congruence_tests.test_common import (
    COLLECTION_NAME,
    compare_collections,
    generate_fixtures,
    init_local,
    init_remote,
    initialize_fixture_collection,
)
from tests.fixtures.payload import one_random_payload_please

UPLOAD_NUM_VECTORS = 100


def test_upsert(local_client: QdrantBase, remote_client: QdrantBase, collection_name: str):
    initialize_fixture_collection(local_client, collection_name)
    initialize_fixture_collection(remote_client, collection_name)

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

    local_client.upsert(collection_name, points_batch)
    remote_client.upsert(collection_name, points_batch)

    id_ = ids[0]
    vector = {k: v[0] for k, v in vectors.items()}
    old_payload = payload[0]

    id_filter = models.Filter(must=[models.HasIdCondition(has_id=[id_])])

    local_old_point = local_client.scroll(
        collection_name,
        scroll_filter=id_filter,
        limit=1,
    )[0][0]
    remote_old_point = remote_client.scroll(collection_name, scroll_filter=id_filter, limit=1)[0][
        0
    ]

    assert local_old_point == remote_old_point
    # endregion

    # region update point
    new_payload = one_random_payload_please(id_)
    assert old_payload != new_payload

    local_client.upsert(
        collection_name,
        [models.PointStruct(id=id_, vector=vector, payload=new_payload)],
    )
    remote_client.upsert(
        collection_name,
        [models.PointStruct(id=id_, vector=vector, payload=new_payload)],
    )

    local_new_point = local_client.scroll(collection_name, scroll_filter=id_filter, limit=1)[0][0]
    remote_new_point = remote_client.scroll(collection_name, scroll_filter=id_filter, limit=1)[0][
        0
    ]

    assert local_new_point == remote_new_point
    # endregion

    compare_collections(
        local_client, remote_client, UPLOAD_NUM_VECTORS, collection_name=collection_name
    )


def test_upload_collection(
    local_client: QdrantBase, remote_client: QdrantBase, collection_name: str
):
    points = generate_fixtures(UPLOAD_NUM_VECTORS)

    initialize_fixture_collection(local_client, collection_name)
    initialize_fixture_collection(remote_client, collection_name)

    vectors = []
    payload = []
    ids = []
    for point in points:
        (ids.append(point.id),)
        vectors.append(point.vector)
        payload.append(point.payload)

    local_client.upload_collection(collection_name, vectors, payload, ids=ids)
    remote_client.upload_collection(collection_name, vectors, payload, ids=ids, wait=True)

    compare_collections(
        local_client, remote_client, UPLOAD_NUM_VECTORS, collection_name=collection_name
    )


@pytest.mark.timeout(60)  # normally takes less than a second
def test_upload_collection_generators(
    local_client: QdrantBase, remote_client: QdrantBase, collection_name: str
):
    points = generate_fixtures(UPLOAD_NUM_VECTORS)

    initialize_fixture_collection(local_client, collection_name)
    initialize_fixture_collection(remote_client, collection_name)

    vectors = []
    payload = []

    for point in points:
        vectors.append(point.vector)
        payload.append(point.payload)

    payload = itertools.cycle(payload)
    local_client.upload_collection(collection_name, vectors, payload, ids=itertools.count())
    remote_client.upload_collection(
        collection_name, vectors, payload, ids=itertools.count(), wait=True
    )

    compare_collections(
        local_client, remote_client, UPLOAD_NUM_VECTORS, collection_name=collection_name
    )


def test_upload_points(local_client: QdrantBase, remote_client: QdrantBase, collection_name: str):
    points = generate_fixtures(UPLOAD_NUM_VECTORS)

    initialize_fixture_collection(local_client, collection_name)
    initialize_fixture_collection(remote_client, collection_name)

    local_client.upload_points(collection_name, points)
    remote_client.upload_points(collection_name, points, wait=True)

    compare_collections(
        local_client, remote_client, UPLOAD_NUM_VECTORS, collection_name=collection_name
    )


def test_upload_uuid_in_batches(
    local_client: QdrantBase, remote_client: QdrantBase, collection_name: str
):
    points = generate_fixtures(UPLOAD_NUM_VECTORS)

    initialize_fixture_collection(local_client, collection_name)
    initialize_fixture_collection(remote_client, collection_name)

    vectors = defaultdict(list)

    for point in points:
        for vector_name, vector in point.vector.items():
            vectors[vector_name].append(vector)

    batch = models.Batch(
        ids=[str(uuid.uuid4()) for _ in points],
        vectors=vectors,
        payloads=[point.payload for point in points],
    )

    local_client.upsert(collection_name, batch)
    remote_client.upsert(collection_name, batch)

    compare_collections(
        local_client, remote_client, UPLOAD_NUM_VECTORS, collection_name=collection_name
    )


def test_upload_collection_float_list(collection_name: str):
    vectors_dim = 50
    local_client = init_local()
    remote_client = init_remote()

    vectors = np.random.randn(UPLOAD_NUM_VECTORS, vectors_dim).tolist()
    vectors_config = models.VectorParams(size=vectors_dim, distance=models.Distance.EUCLID)

    local_client.create_collection(collection_name, vectors_config=vectors_config, timeout=TIMEOUT)
    if remote_client.collection_exists(collection_name):
        remote_client.delete_collection(collection_name, timeout=TIMEOUT)
    remote_client.create_collection(
        collection_name, vectors_config=vectors_config, timeout=TIMEOUT
    )

    ids = list(range(len(vectors)))
    local_client.upload_collection(collection_name, vectors, ids=ids)
    remote_client.upload_collection(collection_name, vectors, ids=ids, wait=True)

    compare_collections(
        local_client, remote_client, UPLOAD_NUM_VECTORS, collection_name=collection_name
    )
    local_client.delete_collection(collection_name)
    remote_client.delete_collection(collection_name)


def test_upload_collection_named_float_list_vectors(
    local_client: QdrantBase, remote_client: QdrantBase, collection_name: str
):
    points = generate_fixtures(UPLOAD_NUM_VECTORS)

    initialize_fixture_collection(local_client, collection_name)
    initialize_fixture_collection(remote_client, collection_name)

    vectors = []  # list[dict[str, float]]
    for point in points:
        vectors.append(point.vector)
    ids = [point.id for point in points]
    local_client.upload_collection(collection_name, vectors, ids=ids)
    remote_client.upload_collection(collection_name, vectors, ids=ids, wait=True)
    compare_collections(
        local_client, remote_client, UPLOAD_NUM_VECTORS, collection_name=collection_name
    )


def test_upload_collection_np_array_2d(collection_name: str):
    vectors_dim = 50
    local_client = init_local()
    remote_client = init_remote()

    vectors = np.random.randn(UPLOAD_NUM_VECTORS, vectors_dim)
    ids = list(range(len(vectors)))
    vectors_config = models.VectorParams(size=vectors_dim, distance=models.Distance.EUCLID)

    local_client.create_collection(
        collection_name,
        vectors_config=vectors_config,
        timeout=TIMEOUT,
    )
    if remote_client.collection_exists(collection_name):
        remote_client.delete_collection(collection_name, timeout=TIMEOUT)
    remote_client.create_collection(
        collection_name,
        vectors_config=vectors_config,
        timeout=TIMEOUT,
    )

    local_client.upload_collection(collection_name, vectors, ids=ids)
    remote_client.upload_collection(collection_name, vectors, ids=ids, wait=True)

    compare_collections(
        local_client, remote_client, UPLOAD_NUM_VECTORS, collection_name=collection_name
    )
    local_client.delete_collection(collection_name)
    remote_client.delete_collection(collection_name)


def test_upload_collection_list_np_arrays(collection_name: str):
    vectors_dim = 50
    local_client = init_local()
    remote_client = init_remote()

    vectors = np.random.randn(UPLOAD_NUM_VECTORS, vectors_dim).tolist()
    vectors = [np.array(vector) for vector in vectors]
    vectors_config = models.VectorParams(size=vectors_dim, distance=models.Distance.EUCLID)
    ids = list(range(len(vectors)))

    local_client.create_collection(
        collection_name,
        vectors_config=vectors_config,
        timeout=TIMEOUT,
    )
    if remote_client.collection_exists(collection_name):
        remote_client.delete_collection(collection_name, timeout=TIMEOUT)
    remote_client.create_collection(
        collection_name,
        vectors_config=vectors_config,
        timeout=TIMEOUT,
    )

    local_client.upload_collection(collection_name, vectors, ids=ids)
    remote_client.upload_collection(collection_name, vectors, ids=ids, wait=True)

    compare_collections(
        local_client, remote_client, UPLOAD_NUM_VECTORS, collection_name=collection_name
    )
    local_client.delete_collection(collection_name)
    remote_client.delete_collection(collection_name)


def test_upload_collection_dict_np_arrays(
    local_client: QdrantBase, remote_client: QdrantBase, collection_name: str
):
    points = generate_fixtures(UPLOAD_NUM_VECTORS)

    initialize_fixture_collection(local_client, collection_name)
    initialize_fixture_collection(remote_client, collection_name)

    intermediate_vectors: dict[str, list[float]] = defaultdict(list)
    vectors: dict[str, np.ndarray] = {}
    ids = [point.id for point in points]
    for point in points:
        for key, vector in point.vector.items():
            intermediate_vectors[key].append(point.vector[key])

    for key in intermediate_vectors:
        vectors[key] = np.array(intermediate_vectors[key])

    local_client.upload_collection(collection_name, vectors, ids=ids)
    remote_client.upload_collection(collection_name, vectors, ids=ids, wait=True)
    compare_collections(
        local_client, remote_client, UPLOAD_NUM_VECTORS, collection_name=collection_name
    )


def test_upload_wrong_vectors(collection_name: str):
    local_client = init_local()
    remote_client = init_remote()

    vector_size = 2
    wrong_vectors_collection = collection_name
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


def test_upsert_without_vector_name(collection_name: str):
    local_client = init_local()
    remote_client = init_remote()

    local_client.create_collection(collection_name=collection_name, vectors_config={})
    if remote_client.collection_exists(collection_name=collection_name):
        remote_client.delete_collection(collection_name=collection_name)
    remote_client.create_collection(collection_name=collection_name, vectors_config={})

    with pytest.raises(ValueError, match="Not existing vector name error"):
        local_client.upsert(
            collection_name, points=[models.PointStruct(id=1, vector=[0.1, 0.2, 0.3])]
        )
    with pytest.raises(
        qdrant_client.http.exceptions.UnexpectedResponse, match="Not existing vector name error"
    ):
        remote_client.upsert(
            collection_name, points=[models.PointStruct(id=1, vector=[0.1, 0.2, 0.3])]
        )
