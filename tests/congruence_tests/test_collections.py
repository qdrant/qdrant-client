import pytest

from qdrant_client.http import models
from tests.congruence_tests.test_common import (
    compare_collections,
    generate_fixtures,
    init_client,
    init_local,
    init_remote,
)

COLLECTION_NAME = "test_collection"


def test_get_collection():
    fixture_records = generate_fixtures()

    remote_client = init_remote()

    remote_collections = remote_client.get_collections()

    for collection in remote_collections.collections:
        remote_client.delete_collection(collection.name)

    local_client = init_local()
    init_client(local_client, fixture_records)

    init_client(remote_client, fixture_records)

    local_collections = local_client.get_collections()

    remote_collections = remote_client.get_collections()

    assert len(local_collections.collections) == len(remote_collections.collections)

    local_collection = local_collections.collections[0].name
    remote_collection = remote_collections.collections[0].name

    assert local_collection == remote_collection

    local_collection_info = local_client.get_collection(local_collection)

    remote_collection_info = remote_client.get_collection(remote_collection)

    assert local_collection_info.points_count == remote_collection_info.points_count

    assert (
        local_collection_info.config.params.vectors == remote_collection_info.config.params.vectors
    )


def test_init_from():
    vector_size = 2

    remote_client = init_remote()
    local_client = init_local()

    records = generate_fixtures(vectors_sizes=vector_size)
    vector_params = models.VectorParams(size=vector_size, distance=models.Distance.COSINE)

    remote_client.recreate_collection(
        collection_name=COLLECTION_NAME, vectors_config=vector_params
    )
    local_client.recreate_collection(collection_name=COLLECTION_NAME, vectors_config=vector_params)
    remote_client.upload_records(COLLECTION_NAME, records, wait=True)
    local_client.upload_records(COLLECTION_NAME, records)
    compare_collections(remote_client, local_client, len(records), collection_name=COLLECTION_NAME)

    new_collection_name = COLLECTION_NAME + "_new"
    remote_client.recreate_collection(
        new_collection_name, vectors_config=vector_params, init_from=COLLECTION_NAME
    )
    local_client.recreate_collection(
        new_collection_name, vectors_config=vector_params, init_from=COLLECTION_NAME
    )
    compare_collections(
        remote_client, local_client, len(records), collection_name=new_collection_name
    )

    remote_client.recreate_collection(
        new_collection_name,
        vectors_config=vector_params,
        init_from=models.InitFrom(collection=COLLECTION_NAME),
    )
    local_client.recreate_collection(
        new_collection_name,
        vectors_config=vector_params,
        init_from=models.InitFrom(collection=COLLECTION_NAME),
    )
    compare_collections(
        remote_client, local_client, len(records), collection_name=new_collection_name
    )
