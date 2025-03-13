from typing import Callable, Any

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
TEST_NUM_POINTS = 10


@pytest.fixture(scope="module")
def fixture_points() -> list[models.PointStruct]:
    return generate_fixtures(num=TEST_NUM_POINTS)


@pytest.fixture(scope="module", autouse=True)
def local_client(fixture_points: list[models.PointStruct], collection_name: str) -> QdrantClient:
    client = init_local()
    init_client(client, fixture_points, collection_name)
    return client


@pytest.fixture(scope="module", autouse=True)
def http_client(fixture_points: list[models.PointStruct], collection_name: str) -> QdrantClient:
    client = init_remote()
    init_client(client, fixture_points, collection_name)
    return client


@pytest.fixture(scope="module", autouse=True)
def grpc_client() -> QdrantClient:
    client = init_remote(prefer_grpc=True)
    return client


def compare_all_clients_results(
    local_client: QdrantBase,
    http_client: QdrantBase,
    grpc_client: QdrantBase,
    foo: Callable[[QdrantBase, Any], Any],
    **kwargs: Any,
):
    compare_client_results(local_client, http_client, foo, **kwargs)
    compare_client_results(http_client, grpc_client, foo, **kwargs)


def test_search_offsets_no_filter(
    local_client: QdrantBase,
    http_client: QdrantBase,
    grpc_client: QdrantBase,
    collection_name: str,
):
    def search_offsets_no_filter(
        client: QdrantBase, collection_name: str
    ) -> models.SearchMatrixOffsetsResponse:
        return client.search_matrix_offsets(
            collection_name=collection_name,
            sample=TEST_NUM_POINTS,
            limit=3,
            using="text",
        )

    compare_all_clients_results(
        local_client,
        http_client,
        grpc_client,
        search_offsets_no_filter,
        collection_name=collection_name,
    )


def test_search_pairs_no_filter(
    local_client: QdrantBase,
    http_client: QdrantBase,
    grpc_client: QdrantBase,
    collection_name: str,
):
    def search_pairs_no_filter(
        client: QdrantBase, collection_name: str
    ) -> models.SearchMatrixPairsResponse:
        return client.search_matrix_pairs(
            collection_name=collection_name,
            sample=TEST_NUM_POINTS,
            limit=3,
            using="text",
        )

    compare_all_clients_results(
        local_client,
        http_client,
        grpc_client,
        search_pairs_no_filter,
        collection_name=collection_name,
    )


def test_search_offsets_filter(
    local_client: QdrantBase,
    http_client: QdrantBase,
    grpc_client: QdrantBase,
    collection_name: str,
):
    def search_offsets_filter(
        client: QdrantBase, query_filter: models.Filter, collection_name: str
    ) -> models.SearchMatrixOffsetsResponse:
        return client.search_matrix_offsets(
            collection_name=collection_name,
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
                collection_name=collection_name,
            )
        except AssertionError as e:
            print(f"\nAttempt {i} failed with filter {query_filter}")
            raise e


def test_search_pairs_filter(
    local_client: QdrantBase,
    http_client: QdrantBase,
    grpc_client: QdrantBase,
    collection_name: str,
):
    def search_pairs_filter(
        client: QdrantBase, query_filter: models.Filter, collection_name: str
    ) -> models.SearchMatrixPairsResponse:
        return client.search_matrix_pairs(
            collection_name=collection_name,
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
                collection_name=collection_name,
            )
        except AssertionError as e:
            print(f"\nAttempt {i} failed with filter {query_filter}")
            raise e
