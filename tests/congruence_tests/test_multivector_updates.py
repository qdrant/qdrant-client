import itertools
import uuid
from collections import defaultdict

import pytest

from qdrant_client.http import models
from qdrant_client.http.exceptions import UnexpectedResponse
from tests.congruence_tests.test_common import (
    COLLECTION_NAME,
    compare_collections,
    init_client,
    init_local,
    init_remote,
    generate_multivector_fixtures,
    multi_vector_config,
)
from tests.fixtures.payload import one_random_payload_please

UPLOAD_NUM_VECTORS = 100


def test_upsert():
    # region upload data
    points = generate_multivector_fixtures(UPLOAD_NUM_VECTORS)
    local_client = init_local()
    init_client(local_client, points, vectors_config=multi_vector_config)

    remote_client = init_remote()
    init_client(remote_client, points, vectors_config=multi_vector_config)

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
    points = generate_multivector_fixtures(UPLOAD_NUM_VECTORS)

    local_client = init_local()
    init_client(local_client, points, vectors_config=multi_vector_config)

    remote_client = init_remote()
    init_client(remote_client, points, vectors_config=multi_vector_config)

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
    points = generate_multivector_fixtures(UPLOAD_NUM_VECTORS)

    local_client = init_local()
    init_client(local_client, points, vectors_config=multi_vector_config)

    remote_client = init_remote()
    init_client(remote_client, points, vectors_config=multi_vector_config)

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
    points = generate_multivector_fixtures(UPLOAD_NUM_VECTORS)

    local_client = init_local()
    init_client(local_client, points, vectors_config=multi_vector_config)

    remote_client = init_remote()
    init_client(remote_client, points, vectors_config=multi_vector_config)

    local_client.upload_points(COLLECTION_NAME, points)
    remote_client.upload_points(COLLECTION_NAME, points, wait=True)

    compare_collections(
        local_client,
        remote_client,
        UPLOAD_NUM_VECTORS,
        attrs=("points_count",),
    )


def test_upload_uuid_in_batches():
    points = generate_multivector_fixtures(UPLOAD_NUM_VECTORS)

    local_client = init_local()
    init_client(local_client, points, vectors_config=multi_vector_config)

    remote_client = init_remote()
    init_client(remote_client, points, vectors_config=multi_vector_config)

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


def test_upsert_empty_multivector():
    """Both clients must reject an empty multivector on every write path."""
    points = generate_multivector_fixtures(UPLOAD_NUM_VECTORS)

    local_client = init_local()
    init_client(local_client, points, vectors_config=multi_vector_config)

    remote_client = init_remote()
    init_client(remote_client, points, vectors_config=multi_vector_config)

    existing_id = points[0].id
    new_point_id = UPLOAD_NUM_VECTORS + 1

    # a multivector with no vectors at all, and one holding an empty vector
    cases = (
        ([], "Multivector must not be empty"),
        ([[]], "vectors of a multivector must be non-empty"),
    )
    for empty_multivector, local_error in cases:
        upsert_structs = (
            [models.PointStruct(id=new_point_id, vector={"multi-text": empty_multivector})],
            [models.PointStruct(id=existing_id, vector={"multi-text": empty_multivector})],
            models.Batch(ids=[new_point_id], vectors={"multi-text": [empty_multivector]}),
        )
        for upsert_struct in upsert_structs:
            with pytest.raises(ValueError, match=local_error):
                local_client.upsert(COLLECTION_NAME, upsert_struct)

            with pytest.raises(UnexpectedResponse):
                remote_client.upsert(COLLECTION_NAME, upsert_struct)

        point_vectors = [
            models.PointVectors(id=existing_id, vector={"multi-text": empty_multivector})
        ]
        with pytest.raises(ValueError, match=local_error):
            local_client.update_vectors(COLLECTION_NAME, points=point_vectors)

        with pytest.raises(UnexpectedResponse):
            remote_client.update_vectors(COLLECTION_NAME, points=point_vectors)

    compare_collections(
        local_client,
        remote_client,
        UPLOAD_NUM_VECTORS,
        attrs=("points_count",),
    )


def test_rejected_empty_multivector_update_leaves_point_untouched():
    """A rejected update_vectors request must not make an absent multivector visible.

    Regression: validation used to run after `deleted_per_vector` was cleared, so rejecting
    an empty multivector still exposed the placeholder to retrieval and search.
    """
    local_client = init_local()
    remote_client = init_remote()

    mvc = models.MultiVectorConfig(comparator=models.MultiVectorComparator.MAX_SIM)
    vectors_config = {
        "a": models.VectorParams(size=4, distance=models.Distance.COSINE, multivector_config=mvc),
        "b": models.VectorParams(size=4, distance=models.Distance.COSINE, multivector_config=mvc),
    }
    local_client.create_collection(COLLECTION_NAME, vectors_config=vectors_config)
    if remote_client.collection_exists(collection_name=COLLECTION_NAME):
        remote_client.delete_collection(collection_name=COLLECTION_NAME)
    remote_client.create_collection(COLLECTION_NAME, vectors_config=vectors_config)

    query = [[1.0, 0.0, 0.0, 0.0]]

    # point 1 has no "a" at all, so "a" is backed by a placeholder marked deleted
    absent_a = [models.PointStruct(id=1, vector={"b": query})]
    local_client.upsert(COLLECTION_NAME, absent_a)
    remote_client.upsert(COLLECTION_NAME, absent_a, wait=True)

    for empty_multivector, local_error in (
        ([], "Multivector must not be empty"),
        ([[]], "vectors of a multivector must be non-empty"),
    ):
        point_vectors = [models.PointVectors(id=1, vector={"a": empty_multivector})]
        with pytest.raises(ValueError, match=local_error):
            local_client.update_vectors(COLLECTION_NAME, points=point_vectors)
        with pytest.raises(UnexpectedResponse):
            remote_client.update_vectors(COLLECTION_NAME, points=point_vectors, wait=True)

        for client in (local_client, remote_client):
            record = client.retrieve(COLLECTION_NAME, ids=[1], with_vectors=True)[0]
            assert "a" not in record.vector
            hits = client.query_points(COLLECTION_NAME, query=query, using="a", limit=5).points
            assert hits == []

    compare_collections(local_client, remote_client, 1, attrs=("points_count",))
