from qdrant_client.http import models
from tests.congruence_tests.test_common import (
    COLLECTION_NAME,
    compare_collections,
    generate_fixtures,
)

NUM_VECTORS = 100


def upload(client_1, client_2, num_vectors=NUM_VECTORS):
    records = generate_fixtures(num_vectors)

    client_1.upload_records(COLLECTION_NAME, records, wait=True)
    client_2.upload_records(COLLECTION_NAME, records, wait=True)
    return records


def test_delete_payload(local_client, remote_client):
    records = upload(local_client, remote_client)

    # region delete one point
    id_ = records[0].id
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
    ids = [records[1].id, records[2].id]
    local_client.delete_payload(COLLECTION_NAME, keys=keys_to_delete, points=ids)
    remote_client.delete_payload(COLLECTION_NAME, keys=keys_to_delete, points=ids, wait=True)

    compare_collections(local_client, remote_client, NUM_VECTORS)
    # endregion

    # region delete by filter
    payload = records[2].payload
    key = "text_data"
    value = payload[key]
    delete_filter = models.Filter(
        must=[models.FieldCondition(key=key, match=models.MatchValue(value=value))]
    )

    local_client.delete_payload(COLLECTION_NAME, keys=["text_data"], points=delete_filter)
    remote_client.delete_payload(COLLECTION_NAME, keys=["text_data"], points=delete_filter, wait=True)

    compare_collections(local_client, remote_client, NUM_VECTORS)
    # endregion


def test_clear_payload(local_client, remote_client):
    records = upload(local_client, remote_client)

    points_selector = [record.id for record in records[:5]]
    local_client.clear_payload(COLLECTION_NAME, points_selector)
    remote_client.clear_payload(COLLECTION_NAME, points_selector)

    compare_collections(local_client, remote_client, NUM_VECTORS)

    payload = records[42].payload
    key = "text_data"
    value = payload[key]
    points_selector = models.Filter(
        must=[models.FieldCondition(key=key, match=models.MatchValue(value=value))]
    )
    local_client.clear_payload(COLLECTION_NAME, points_selector)
    remote_client.clear_payload(COLLECTION_NAME, points_selector)

    compare_collections(local_client, remote_client, NUM_VECTORS)


def test_update_payload(local_client, remote_client):
    records = upload(local_client, remote_client)

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

    compare_collections(local_client, remote_client, NUM_VECTORS)  # sanity check
