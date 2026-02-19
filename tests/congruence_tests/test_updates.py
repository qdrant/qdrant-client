import itertools
import uuid
from collections import defaultdict
import random
from copy import deepcopy

import numpy as np
import pytest

import qdrant_client.http.exceptions
from qdrant_client.client_base import QdrantBase
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
    vectors = []  # list[dict[str, float]]
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
    intermediate_vectors: dict[str, list[float]] = defaultdict(list)
    vectors: dict[str, np.ndarray] = {}
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
            remote_client.upload_points(
                wrong_vectors_collection,
                points=[models.PointStruct(id=3, vector=dense_vector)],
                wait=True,
            )

        with pytest.raises(ValueError):
            local_client.upload_points(
                wrong_vectors_collection, points=[models.PointStruct(id=3, vector=dense_vector)]
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


def test_upsert_without_vector_name():
    local_client = init_local()
    remote_client = init_remote()

    local_client.create_collection(collection_name=COLLECTION_NAME, vectors_config={})
    if remote_client.collection_exists(collection_name=COLLECTION_NAME):
        remote_client.delete_collection(collection_name=COLLECTION_NAME)
    remote_client.create_collection(collection_name=COLLECTION_NAME, vectors_config={})

    with pytest.raises(ValueError, match="Not existing vector name error"):
        local_client.upsert(
            COLLECTION_NAME, points=[models.PointStruct(id=1, vector=[0.1, 0.2, 0.3])]
        )
    with pytest.raises(
        qdrant_client.http.exceptions.UnexpectedResponse, match="Not existing vector name error"
    ):
        remote_client.upsert(
            COLLECTION_NAME, points=[models.PointStruct(id=1, vector=[0.1, 0.2, 0.3])]
        )


def test_update_vectors():
    local_client = init_local()
    remote_client = init_remote()

    # region unnamed vector in an empty collection
    vectors_config = models.VectorParams(size=2, distance=models.Distance.DOT)
    local_client.create_collection(collection_name=COLLECTION_NAME, vectors_config=vectors_config)
    if remote_client.collection_exists(collection_name=COLLECTION_NAME):
        remote_client.delete_collection(collection_name=COLLECTION_NAME)
    remote_client.create_collection(collection_name=COLLECTION_NAME, vectors_config=vectors_config)

    points = [models.PointStruct(id=1, vector={})]

    local_client.upsert(COLLECTION_NAME, points=points)
    remote_client.upsert(COLLECTION_NAME, points=points, wait=True)

    local_client.update_vectors(
        COLLECTION_NAME, points=[models.PointVectors(id=1, vector=[0.2, 0.3])]
    )
    remote_client.update_vectors(
        COLLECTION_NAME,
        points=[models.PointVectors(id=1, vector=[0.2, 0.3])],
    )

    compare_collections(
        local_client,
        remote_client,
        10,
        collection_name=COLLECTION_NAME,
    )
    local_client.delete_collection(collection_name=COLLECTION_NAME)
    remote_client.delete_collection(collection_name=COLLECTION_NAME)
    # endregion

    # region sparse vector in an empty collection
    sparse_vectors_config = {"sparse": models.SparseVectorParams()}
    local_client.create_collection(
        collection_name=COLLECTION_NAME, sparse_vectors_config=sparse_vectors_config
    )
    remote_client.create_collection(
        collection_name=COLLECTION_NAME, sparse_vectors_config=sparse_vectors_config
    )

    points = [models.PointStruct(id=1, vector={})]
    local_client.upsert(COLLECTION_NAME, points=points)
    remote_client.upsert(
        COLLECTION_NAME,
        points=points,
    )

    sparse_points = [
        models.PointVectors(
            id=1,
            vector={"sparse": models.SparseVector(indices=[0, 1], values=[0.2, 0.3])},
        )
    ]
    local_client.update_vectors(COLLECTION_NAME, points=sparse_points)
    remote_client.update_vectors(COLLECTION_NAME, points=sparse_points)

    compare_collections(
        local_client,
        remote_client,
        10,
        collection_name=COLLECTION_NAME,
    )
    local_client.delete_collection(collection_name=COLLECTION_NAME)
    remote_client.delete_collection(collection_name=COLLECTION_NAME)
    # endregion

    # region multivector in an empty collection
    local_client.delete_collection(collection_name=COLLECTION_NAME)
    remote_client.delete_collection(collection_name=COLLECTION_NAME)

    multivectors_config = models.VectorParams(
        size=2,
        distance=models.Distance.DOT,
        multivector_config=models.MultiVectorConfig(
            comparator=models.MultiVectorComparator.MAX_SIM
        ),
    )

    local_client.create_collection(
        collection_name=COLLECTION_NAME, vectors_config=multivectors_config
    )
    remote_client.create_collection(
        collection_name=COLLECTION_NAME, vectors_config=multivectors_config
    )

    points = [models.PointStruct(id=1, vector={})]
    local_client.upsert(COLLECTION_NAME, points=points)
    remote_client.upsert(
        COLLECTION_NAME,
        points=points,
    )
    multivector_points = [models.PointVectors(id=1, vector=[[0.2, 0.3], [0.4, 0.5]])]
    local_client.update_vectors(COLLECTION_NAME, points=multivector_points)
    remote_client.update_vectors(COLLECTION_NAME, points=multivector_points)
    compare_collections(
        local_client,
        remote_client,
        10,
        collection_name=COLLECTION_NAME,
    )
    local_client.delete_collection(collection_name=COLLECTION_NAME)
    remote_client.delete_collection(collection_name=COLLECTION_NAME)
    # endregion

    # region named vectors
    named_vectors_config = {"text": models.VectorParams(size=2, distance=models.Distance.DOT)}
    local_client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=named_vectors_config,
    )
    remote_client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=named_vectors_config,
    )
    points = [models.PointStruct(id=1, vector={})]

    local_client.upsert(COLLECTION_NAME, points=points)
    remote_client.upsert(
        COLLECTION_NAME,
        points=points,
    )
    named_vector_points = [
        models.PointVectors(
            id=1,
            vector={"text": [0.2, 0.3]},
        )
    ]

    local_client.update_vectors(COLLECTION_NAME, points=named_vector_points)
    remote_client.update_vectors(COLLECTION_NAME, points=named_vector_points)
    compare_collections(
        local_client,
        remote_client,
        10,
        collection_name=COLLECTION_NAME,
    )
    local_client.delete_collection(collection_name=COLLECTION_NAME)
    remote_client.delete_collection(collection_name=COLLECTION_NAME)
    # endregion


@pytest.mark.parametrize("prefer_grpc", [False, True])
def test_update_filter(prefer_grpc):
    local_client = init_local()
    remote_client = init_remote(prefer_grpc=prefer_grpc)

    vectors_config = models.VectorParams(size=2, distance=models.Distance.DOT)
    local_client.create_collection(collection_name=COLLECTION_NAME, vectors_config=vectors_config)
    if remote_client.collection_exists(collection_name=COLLECTION_NAME):
        remote_client.delete_collection(collection_name=COLLECTION_NAME)
    remote_client.create_collection(collection_name=COLLECTION_NAME, vectors_config=vectors_config)

    original_vector = [random.random(), random.random()]
    original_points = [
        models.PointStruct(id=1, vector=original_vector[:], payload={"digit": 1}),
        models.PointStruct(id=2, vector=original_vector[:], payload={"digit": 2}),
    ]

    local_client.upsert(COLLECTION_NAME, points=original_points)
    remote_client.upsert(COLLECTION_NAME, points=original_points)
    # collection points:
    # id=1, vector=original_vector, payload={digit: 1}
    # id=2, vector=original_vector, payload={digit: 2}

    new_points = [
        models.PointStruct(id=1, vector=original_vector[:], payload={"digit": 3}),
        models.PointStruct(id=2, vector=original_vector[:], payload={"digit": 4}),
        models.PointStruct(id=3, vector=original_vector[:], payload={"digit": 5}),
    ]

    update_filter = models.Filter(
        must=models.FieldCondition(key="digit", match=models.MatchValue(value=1))
    )
    local_client.upsert(COLLECTION_NAME, points=new_points, update_filter=update_filter)
    remote_client.upsert(COLLECTION_NAME, points=new_points, update_filter=update_filter)
    # collection points:
    # id=1, vector=original_vector, payload={digit: 3}
    # id=2, vector=original_vector, payload={digit: 2}
    # id=3, vector=original_vector, payload={digit: 5}
    compare_collections(local_client, remote_client, 10, collection_name=COLLECTION_NAME)

    retrieved_points = local_client.retrieve(collection_name=COLLECTION_NAME, ids=[1, 2, 3])
    assert retrieved_points[0].payload["digit"] == 3
    assert retrieved_points[1].payload["digit"] == 2
    assert len(retrieved_points) == 3

    update_filter = models.Filter(
        must=models.FieldCondition(key="digit", match=models.MatchValue(value=3))
    )
    new_vector = (-np.array(original_vector[:])).tolist()
    new_point_vectors = [
        models.PointVectors(id=1, vector=new_vector[:]),
        models.PointVectors(id=2, vector=new_vector[:]),
    ]
    local_client.update_vectors(
        COLLECTION_NAME, points=new_point_vectors, update_filter=update_filter
    )
    remote_client.update_vectors(
        COLLECTION_NAME, points=new_point_vectors, update_filter=update_filter
    )
    # collection points:
    # id=1, vector=-original_vector, payload={digit: 3}
    # id=2, vector=original_vector, payload={digit: 2}
    # id=3, vector=original_vector, payload={digit: 5}
    compare_collections(local_client, remote_client, 10, collection_name=COLLECTION_NAME)

    retrieved_points = local_client.retrieve(
        collection_name=COLLECTION_NAME, ids=[1, 2], with_vectors=True
    )
    assert np.allclose(retrieved_points[0].vector, new_vector)
    assert np.allclose(retrieved_points[1].vector, original_vector)

    new_points_2 = [
        models.PointStruct(id=1, vector=original_vector[:], payload={"digit": 1}),
        models.PointStruct(id=2, vector=new_vector, payload={"digit": 99}),
    ]

    update_filter = models.Filter(
        must=models.FieldCondition(key="digit", match=models.MatchValue(value=3))
    )

    local_client.upload_points(COLLECTION_NAME, points=new_points_2, update_filter=update_filter)
    remote_client.upload_points(COLLECTION_NAME, points=new_points_2, update_filter=update_filter)
    # collection points:
    # id=1, vector=original_vector, payload={digit: 1}
    # id=2, vector=original_vector, payload={digit: 2}
    # id=3, vector=original_vector, payload={digit: 5}
    compare_collections(local_client, remote_client, 10, collection_name=COLLECTION_NAME)

    retrieved_points = local_client.retrieve(collection_name=COLLECTION_NAME, ids=[1, 2])
    assert retrieved_points[0].payload["digit"] == 1
    assert retrieved_points[1].payload["digit"] == 2

    new_points_3 = [
        models.PointStruct(id=1, vector=original_vector[:], payload={"digit": 3}),
        models.PointStruct(id=2, vector=original_vector[:], payload={"digit": 99}),
    ]
    update_filter = models.Filter(
        must=models.FieldCondition(key="digit", match=models.MatchValue(value=1))
    )

    local_client.upload_points(
        COLLECTION_NAME, points=new_points_3, update_filter=update_filter, batch_size=1, parallel=2
    )
    remote_client.upload_points(
        COLLECTION_NAME, points=new_points_3, update_filter=update_filter, batch_size=1, parallel=2
    )
    # collection points:
    # id=1, vector=original_vector, payload={digit: 3}
    # id=2, vector=original_vector, payload={digit: 2}
    # id=3, vector=original_vector, payload={digit: 5}
    compare_collections(local_client, remote_client, 10, collection_name=COLLECTION_NAME)

    retrieved_points = local_client.retrieve(collection_name=COLLECTION_NAME, ids=[1, 2])
    assert retrieved_points[0].payload["digit"] == 3
    assert retrieved_points[1].payload["digit"] == 2

    vectors = [original_vector[:], original_vector[:]]
    ids = [1, 2]
    payload = [
        {"digit": 1},
        {"digit": 99},
    ]
    update_filter = models.Filter(
        must=models.FieldCondition(key="digit", match=models.MatchValue(value=3))
    )

    # not testing MP upload_collection, since upload_points uses _upload_collection under the hood
    local_client.upload_collection(
        COLLECTION_NAME, vectors=vectors, ids=ids, payload=payload, update_filter=update_filter
    )
    remote_client.upload_collection(
        COLLECTION_NAME, vectors=vectors, ids=ids, payload=payload, update_filter=update_filter
    )
    # collection points:
    # id=1, vector=original_vector, payload={digit: 1}
    # id=2, vector=original_vector, payload={digit: 2}
    # id=3, vector=original_vector, payload={digit: 5}
    compare_collections(local_client, remote_client, 10, collection_name=COLLECTION_NAME)

    retrieved_points = local_client.retrieve(collection_name=COLLECTION_NAME, ids=[1, 2])
    assert retrieved_points[0].payload["digit"] == 1
    assert retrieved_points[1].payload["digit"] == 2

    ids = [1, 2, 4]
    vectors = [original_vector[:], original_vector[:], original_vector[:]]
    payload = [{"digit": 3}, {"digit": 0}, {"digit": 4}]
    points_batch = models.PointsBatch(
        batch=models.Batch(ids=ids, vectors=vectors, payloads=payload),
        update_filter=models.Filter(must=models.HasIdCondition(has_id=[1])),
    )

    point_vectors = [
        models.PointVectors(
            id=3,
            vector=new_vector[:],
        ),
        models.PointVectors(id=1, vector=new_vector[:]),
    ]

    upsert_batch = models.UpsertOperation(upsert=points_batch)
    update_vectors = models.UpdateVectorsOperation(
        update_vectors=models.UpdateVectors(
            points=point_vectors,
            update_filter=models.Filter(must=models.HasIdCondition(has_id=[3])),
        )
    )

    local_client.batch_update_points(
        COLLECTION_NAME, update_operations=[upsert_batch, update_vectors]
    )
    remote_client.batch_update_points(
        COLLECTION_NAME, update_operations=[upsert_batch, update_vectors]
    )

    compare_collections(local_client, remote_client, 10, collection_name=COLLECTION_NAME)
    retrieved_points = local_client.retrieve(
        collection_name=COLLECTION_NAME, ids=[1, 2, 3, 4], with_vectors=True
    )
    assert retrieved_points[0].payload["digit"] == 3  # payload updated
    assert retrieved_points[1].payload["digit"] == 2  # payload stays unchanged
    assert np.allclose(retrieved_points[0].vector, original_vector)  # vector stays unchanged
    assert np.allclose(retrieved_points[2].vector, new_vector)  # vector updated
    assert len(retrieved_points) == 4  # not existing point inserted

    points_list = models.PointsList(
        points=[
            models.PointStruct(id=1, vector=original_vector[:], payload={"digit": 1}),
            models.PointStruct(id=2, vector=original_vector[:], payload={"digit": 99}),
            models.PointStruct(id=5, vector=original_vector[:], payload={"digit": 5}),
        ],
        update_filter=models.Filter(must=models.HasIdCondition(has_id=[2])),
    )
    upsert_points_list = models.UpsertOperation(upsert=points_list)

    local_client.batch_update_points(COLLECTION_NAME, update_operations=[upsert_points_list])
    remote_client.batch_update_points(COLLECTION_NAME, update_operations=[upsert_points_list])
    compare_collections(local_client, remote_client, 10, collection_name=COLLECTION_NAME)

    retrieved_points = local_client.retrieve(collection_name=COLLECTION_NAME, ids=[1, 2, 5])
    assert retrieved_points[0].payload["digit"] == 3
    assert retrieved_points[1].payload["digit"] == 99
    assert len(retrieved_points) == 3


@pytest.mark.parametrize("prefer_grpc", [False, True])
def test_update_mode(prefer_grpc: bool) -> None:
    def upload(
        client: QdrantBase,
        collection_name: str,
        points: list[models.PointStruct],
        update_mode: models.UpdateMode,
        method: str = "upsert",
    ) -> None:
        # method: `upsert`, `upload_points`, `upload_collection`
        print(method)
        if method == "upsert":
            client.upsert(collection_name, points, update_mode=update_mode)
        elif method == "upload_points":
            client.upload_points(collection_name, points, update_mode=update_mode)
        elif method == "upload_collection":
            ids = []
            payloads = []
            vectors = []
            for point in points:
                ids.append(point.id)
                payloads.append(point.payload)
                vectors.append(point.vector)
            client.upload_collection(
                collection_name,
                vectors=vectors,
                ids=ids,
                payload=payloads,
                update_mode=update_mode,
            )

    for method in ("upsert", "upload_points", "upload_collection"):
        local_client = init_local()
        remote_client = init_remote()
        vector_params = models.VectorParams(size=50, distance=models.Distance.DOT)
        fixture_points = generate_fixtures(UPLOAD_NUM_VECTORS, vectors_sizes=50)
        initialize_fixture_collection(local_client, COLLECTION_NAME, vectors_config=vector_params)
        initialize_fixture_collection(remote_client, COLLECTION_NAME, vectors_config=vector_params)

        first_point = fixture_points[0]
        second_point = fixture_points[1]
        new_point = fixture_points[-1]
        new_point.id += 1

        upload(
            client=local_client,
            collection_name=COLLECTION_NAME,
            points=fixture_points,
            update_mode=models.UpdateMode.UPSERT,
            method=method,
        )
        upload(
            client=remote_client,
            collection_name=COLLECTION_NAME,
            points=fixture_points,
            update_mode=models.UpdateMode.UPSERT,
            method=method,
        )
        modified_second_point = deepcopy(second_point)
        modified_second_point.vector = first_point.vector

        upload(
            client=local_client,
            collection_name=COLLECTION_NAME,
            points=[modified_second_point, new_point],
            update_mode=models.UpdateMode.INSERT_ONLY,
            method=method,
        )
        upload(
            client=remote_client,
            collection_name=COLLECTION_NAME,
            points=[modified_second_point, new_point],
            update_mode=models.UpdateMode.INSERT_ONLY,
            method=method,
        )

        local_points = local_client.retrieve(
            COLLECTION_NAME, ids=[second_point.id, new_point.id], with_vectors=True
        )
        remote_points = remote_client.retrieve(
            COLLECTION_NAME, ids=[second_point.id, new_point.id], with_vectors=True
        )

        assert np.allclose(local_points[0].vector, remote_points[0].vector)
        assert np.allclose(local_points[0].vector, second_point.vector)
        assert len(local_points) == len(remote_points) == 2

        not_existing_point = deepcopy(new_point)
        not_existing_point.id += 1

        upload(
            client=local_client,
            collection_name=COLLECTION_NAME,
            points=[modified_second_point, not_existing_point],
            update_mode=models.UpdateMode.UPDATE_ONLY,
            method=method,
        )
        upload(
            client=remote_client,
            collection_name=COLLECTION_NAME,
            points=[modified_second_point, not_existing_point],
            update_mode=models.UpdateMode.UPDATE_ONLY,
            method=method,
        )
        local_points = local_client.retrieve(
            COLLECTION_NAME, ids=[second_point.id, not_existing_point.id], with_vectors=True
        )
        remote_points = remote_client.retrieve(
            COLLECTION_NAME, ids=[second_point.id, not_existing_point.id], with_vectors=True
        )

        assert np.allclose(local_points[0].vector, remote_points[0].vector)
        assert np.allclose(local_points[0].vector, first_point.vector)
        assert len(local_points) == len(remote_points) == 1
