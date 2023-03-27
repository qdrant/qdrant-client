import pytest

from qdrant_client.http import models
from tests.congruence_tests.test_common import (
    COLLECTION_NAME,
    compare_client_results,
    delete_fixture_collection,
    generate_fixtures,
    init_local,
    init_remote,
    initialize_fixture_collection,
)
from tests.fixtures.payload import one_random_payload_please

UPLOAD_NUM_VECTORS = 100


@pytest.fixture
def local_client():
    client = init_local()
    initialize_fixture_collection(client)
    yield client
    delete_fixture_collection(client)


@pytest.fixture
def remote_client():
    client = init_remote()
    initialize_fixture_collection(client)
    yield client
    delete_fixture_collection(client)


def compare_collections(
    client_1, client_2, attrs=("vectors_count", "indexed_vectors_count", "points_count")
):
    collection_1 = client_1.get_collection(COLLECTION_NAME)
    collection_2 = client_2.get_collection(COLLECTION_NAME)

    assert all(getattr(collection_1, attr) == getattr(collection_2, attr) for attr in attrs)

    # UPLOAD_NUM_VECTORS * 2 to be sure that we have no excess points uploaded
    compare_client_results(
        client_1,
        client_2,
        lambda client: client.scroll(COLLECTION_NAME, limit=UPLOAD_NUM_VECTORS * 2),
    )


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

    compare_collections(local_client, remote_client)


def test_upload_collection(local_client, remote_client):
    records = generate_fixtures(UPLOAD_NUM_VECTORS)

    vectors = []
    payload = []
    for record in records:
        vectors.append(record.vector)
        payload.append(record.payload)

    local_client.upload_collection(COLLECTION_NAME, vectors, payload)
    remote_client.upload_collection(COLLECTION_NAME, vectors, payload)

    compare_collections(local_client, remote_client)


def test_upload_records(local_client, remote_client):
    records = generate_fixtures(UPLOAD_NUM_VECTORS)

    local_client.upload_records(COLLECTION_NAME, records)
    remote_client.upload_records(COLLECTION_NAME, records)

    compare_collections(local_client, remote_client)


def test_update_payload(local_client, remote_client):
    # region upload data
    records = generate_fixtures(UPLOAD_NUM_VECTORS)

    local_client.upload_records(COLLECTION_NAME, records)
    remote_client.upload_records(COLLECTION_NAME, records)
    # endregion

    # region fetch point
    id_ = records[0].id
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

    compare_collections(local_client, remote_client)  # sanity check
