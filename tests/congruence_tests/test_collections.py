import pytest

from qdrant_client.http import models
from tests.congruence_tests.test_common import (
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

    remote_client = init_remote()
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
