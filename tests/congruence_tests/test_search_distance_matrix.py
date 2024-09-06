from typing import List

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


def test_no_filter(
    local_client,
    http_client,
    grpc_client,
):
    def search_offsets(client: QdrantBase) -> models.SearchMatrixOffsetsResponse:
        return client.search_distance_matrix_offsets(
            collection_name=COLLECTION_NAME,
            sample=TEST_NUM_POINTS,
            limit=3,
            using="text",
        )

    def search_pairs(client: QdrantBase) -> models.SearchMatrixPairsResponse:
        return client.search_distance_matrix_pairs(
            collection_name=COLLECTION_NAME,
            sample=TEST_NUM_POINTS,
            limit=3,
            using="text",
        )

    # compare offsets output
    compare_client_results(grpc_client, http_client, search_offsets)
    compare_client_results(local_client, http_client, search_offsets)

    # compare pairs output
    compare_client_results(grpc_client, http_client, search_pairs)
    compare_client_results(local_client, http_client, search_pairs)

