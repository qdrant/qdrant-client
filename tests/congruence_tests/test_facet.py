import random
import time
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
from tests.fixtures.filters import one_random_filter_please

INT_KEY = "rand_digit"
INT_ID_KEY = "id"
UUID_KEY = "text_array"
STRING_ID_KEY = "id_str"
STRING_KEY = "city.name"


def all_facet_keys() -> List[str]:
    return [INT_KEY, INT_ID_KEY, UUID_KEY, STRING_ID_KEY, STRING_KEY]


@pytest.fixture(scope="module")
def fixture_points() -> List[models.PointStruct]:
    return generate_fixtures()


@pytest.fixture(scope="module", autouse=True)
def local_client(fixture_points) -> QdrantClient:
    client = init_local()
    init_client(client, fixture_points)
    return client


@pytest.fixture(scope="module", autouse=True)
def http_client(fixture_points) -> QdrantClient:
    client = init_remote()
    init_client(client, fixture_points)
    client.create_payload_index(
        collection_name=COLLECTION_NAME,
        field_name=INT_KEY,
        field_schema=models.PayloadSchemaType.INTEGER,
    )
    client.create_payload_index(
        collection_name=COLLECTION_NAME,
        field_name=INT_ID_KEY,
        field_schema=models.PayloadSchemaType.INTEGER,
    )
    client.create_payload_index(
        collection_name=COLLECTION_NAME,
        field_name=UUID_KEY,
        field_schema=models.PayloadSchemaType.UUID,
    )
    client.create_payload_index(
        collection_name=COLLECTION_NAME,
        field_name=STRING_KEY,
        field_schema=models.PayloadSchemaType.KEYWORD,
    )
    client.create_payload_index(
        collection_name=COLLECTION_NAME,
        field_name=STRING_ID_KEY,
        field_schema=models.PayloadSchemaType.KEYWORD,
    )
    return client


@pytest.fixture(scope="module", autouse=True)
def grpc_client(fixture_points) -> QdrantClient:
    client = init_remote(prefer_grpc=True)
    return client


def test_minimal(
    local_client,
    http_client,
    grpc_client,
):
    def f(client: QdrantBase, facet_key: str, **kwargs) -> models.FacetResponse:
        return client.facet(
            collection_name=COLLECTION_NAME,
            key=facet_key,
        )

    for key in all_facet_keys():
        compare_client_results(grpc_client, http_client, f, facet_key=key)
        compare_client_results(local_client, http_client, f, facet_key=key)


def test_limit(
    local_client,
    http_client,
    grpc_client,
):
    def f(client: QdrantBase, facet_key: str, limit: int, **kwargs) -> models.FacetResponse:
        return client.facet(
            collection_name=COLLECTION_NAME,
            key=facet_key,
            limit=limit,
        )

    for _ in range(10):
        rand_num = random.randint(1, 100)
        for key in all_facet_keys():
            compare_client_results(grpc_client, http_client, f, facet_key=key, limit=rand_num)
            compare_client_results(local_client, http_client, f, facet_key=key, limit=rand_num)


def test_exact(
    local_client,
    http_client,
    grpc_client,
):
    def f(client: QdrantBase, facet_key: str, **kwargs) -> models.FacetResponse:
        return client.facet(
            collection_name=COLLECTION_NAME,
            key=facet_key,
            limit=5000,
            exact=True,
        )

    for key in all_facet_keys():
        compare_client_results(grpc_client, http_client, f, facet_key=key)
        compare_client_results(local_client, http_client, f, facet_key=key)


def test_filtered(
    local_client,
    http_client,
    grpc_client,
):
    def f(
        client: QdrantBase, facet_key: str, facet_filter: models.Filter, **kwargs
    ) -> models.FacetResponse:
        return client.facet(
            collection_name=COLLECTION_NAME,
            key=facet_key,
            facet_filter=facet_filter,
            exact=False,
        )

    for key in all_facet_keys():
        filter_ = one_random_filter_please()
        for _ in range(10):
            compare_client_results(
                grpc_client, http_client, f, facet_key=key, facet_filter=filter_
            )
            compare_client_results(
                local_client, http_client, f, facet_key=key, facet_filter=filter_
            )


def test_exact_filtered(
    local_client,
    http_client,
    grpc_client,
):
    def f(
        client: QdrantBase, facet_key: str, facet_filter: models.Filter, **kwargs
    ) -> models.FacetResponse:
        return client.facet(
            collection_name=COLLECTION_NAME,
            key=facet_key,
            limit=5000,
            exact=True,
            facet_filter=facet_filter,
        )

    for key in all_facet_keys():
        for _ in range(10):
            filter_ = one_random_filter_please()
            compare_client_results(
                grpc_client, http_client, f, facet_key=key, facet_filter=filter_
            )
            compare_client_results(
                local_client, http_client, f, facet_key=key, facet_filter=filter_
            )
