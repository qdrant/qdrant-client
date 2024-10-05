from typing import List, Callable, Any

import pytest

from qdrant_client import QdrantClient, models
from qdrant_client.client_base import QdrantBase
from tests.congruence_tests.test_common import (
    COLLECTION_NAME,
    compare_client_results,
    generate_fixtures,
    init_client,
    init_local,
    init_remote,
)
from tests.fixtures.filters import one_random_filter_please

# to keep the test deterministic we sample all the points available
TEST_NUM_POINTS = 100

@pytest.fixture(scope="module")
def fixture_points() -> List[models.PointStruct]:
    return generate_fixtures(num=TEST_NUM_POINTS)


@pytest.fixture(scope="module", autouse=True)
def local_client(fixture_points) -> QdrantClient:
    client = init_local()
    init_client(client, fixture_points)
    return client


@pytest.fixture(scope="module", autouse=True)
def http_client(fixture_points) -> QdrantClient:
    client = init_remote()
    init_client(client, fixture_points)
    return client


@pytest.fixture(scope="module", autouse=True)
def grpc_client(fixture_points) -> QdrantClient:
    client = init_remote(prefer_grpc=True)
    return client


def compare_all_clients_results(
    local_client: QdrantClient,
    http_client: QdrantClient,
    grpc_client: QdrantClient,
    foo: Callable[[QdrantBase, Any], Any],
    **kwargs: Any,
):
    compare_client_results(local_client, http_client, foo, **kwargs)
    compare_client_results(http_client, grpc_client, foo, **kwargs)

def test_search_offsets_no_filter(
    local_client,
    http_client,
    grpc_client,
):
    def search_offsets_no_filter(client: QdrantBase) -> models.SearchMatrixOffsetsResponse:
        return client.search_distance_matrix_offsets(
            collection_name=COLLECTION_NAME,
            sample=TEST_NUM_POINTS,
            limit=3,
            using="text",
        )

    compare_all_clients_results(local_client, http_client, grpc_client, search_offsets_no_filter)


def test_search_pairs_no_filter(
        local_client,
        http_client,
        grpc_client,
):
    def search_pairs_no_filter(client: QdrantBase) -> models.SearchMatrixPairsResponse:
        return client.search_distance_matrix_pairs(
            collection_name=COLLECTION_NAME,
            sample=TEST_NUM_POINTS,
            limit=3,
            using="text",
        )

    compare_all_clients_results(local_client, http_client, grpc_client, search_pairs_no_filter)

def test_search_offsets_filter(
        local_client,
        http_client,
        grpc_client,
):
    def search_offsets_filter(client: QdrantBase, query_filter: models.Filter) -> models.SearchMatrixOffsetsResponse:
        return client.search_distance_matrix_offsets(
            collection_name=COLLECTION_NAME,
            sample=TEST_NUM_POINTS,
            limit=3,
            using="text",
            query_filter=query_filter,
        )

    for i in range(10):
        query_filter = one_random_filter_please()
        try:
            compare_all_clients_results(
                local_client,
                http_client,
                grpc_client,
                search_offsets_filter,
                query_filter=query_filter,
            )
        except AssertionError as e:
            print(f"\nAttempt {i} failed with filter {query_filter}")
            raise e

def test_search_pairs_filter(
        local_client,
        http_client,
        grpc_client,
):
    def search_pairs_filter(client: QdrantBase, query_filter: models.Filter) -> models.SearchMatrixPairsResponse:
        return client.search_distance_matrix_pairs(
            collection_name=COLLECTION_NAME,
            sample=TEST_NUM_POINTS,
            limit=3,
            using="text",
            query_filter=query_filter,
        )

    for i in range(10):
        query_filter = one_random_filter_please()
        try:
            compare_all_clients_results(
                local_client,
                http_client,
                grpc_client,
                search_pairs_filter,
                query_filter=query_filter,
            )
        except AssertionError as e:
            print(f"\nAttempt {i} failed with filter {query_filter}")
            raise e