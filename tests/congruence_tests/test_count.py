import uuid

from qdrant_client.client_base import QdrantBase
from qdrant_client.http import models
from tests.congruence_tests.test_common import (
    COLLECTION_NAME,
    compare_client_results,
    generate_fixtures,
    generate_sparse_fixtures,
    init_client,
    init_local,
    init_remote,
    sparse_vectors_config,
)
from tests.fixtures.filters import one_random_filter_please


def count_all(client: QdrantBase, collection_name: str = COLLECTION_NAME) -> int:
    return client.count(
        collection_name=collection_name,
        count_filter=None,
    ).count


def filter_count(
    client: QdrantBase, count_filter: models.Filter, collection_name: str = COLLECTION_NAME
) -> int:
    return client.count(
        collection_name=collection_name,
        count_filter=count_filter,
    ).count


def test_simple_count(local_client: QdrantBase, remote_client: QdrantBase):
    fixture_points = generate_fixtures()

    collection_name = f"{COLLECTION_NAME}_{uuid.uuid4().hex}"
    init_client(local_client, fixture_points, collection_name)
    init_client(remote_client, fixture_points, collection_name)

    compare_client_results(local_client, remote_client, count_all, collection_name=collection_name)

    for _ in range(100):
        count_filter = one_random_filter_please()
        try:
            compare_client_results(
                local_client,
                remote_client,
                filter_count,
                count_filter=count_filter,
                collection_name=collection_name,
            )
        except AssertionError as e:
            print(f"\nFailed with filter {count_filter}")
            raise e


def test_simple_sparse_search(local_client: QdrantBase, remote_client: QdrantBase):
    fixture_points = generate_sparse_fixtures()

    collection_name = f"{COLLECTION_NAME}_{uuid.uuid4().hex}"
    init_client(
        local_client, fixture_points, collection_name, sparse_vectors_config=sparse_vectors_config
    )
    init_client(
        remote_client, fixture_points, collection_name, sparse_vectors_config=sparse_vectors_config
    )

    compare_client_results(local_client, remote_client, count_all, collection_name=collection_name)

    for _ in range(100):
        count_filter = one_random_filter_please()
        try:
            compare_client_results(
                local_client,
                remote_client,
                filter_count,
                count_filter=count_filter,
                collection_name=collection_name,
            )
        except AssertionError as e:
            print(f"\nFailed with filter {count_filter}")
            raise e
