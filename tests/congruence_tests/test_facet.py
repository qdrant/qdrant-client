import random
from typing import Any

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
BOOL_KEY = "rand_bool"


def all_facet_keys() -> list[str]:
    return [INT_KEY, INT_ID_KEY, UUID_KEY, STRING_ID_KEY, STRING_KEY, BOOL_KEY]


@pytest.fixture(scope="module")
def fixture_points() -> list[models.PointStruct]:
    return generate_fixtures()


@pytest.fixture(scope="module", autouse=True)
def local_client(fixture_points: list[models.PointStruct], collection_name: str) -> QdrantClient:
    client = init_local()
    init_client(client, fixture_points, collection_name)
    return client


@pytest.fixture(scope="module", autouse=True)
def http_client(fixture_points: list[models.PointStruct], collection_name: str) -> QdrantClient:
    client = init_remote()
    init_client(client, fixture_points, collection_name)
    client.create_payload_index(
        collection_name=collection_name,
        field_name=INT_KEY,
        field_schema=models.PayloadSchemaType.INTEGER,
    )
    client.create_payload_index(
        collection_name=collection_name,
        field_name=INT_ID_KEY,
        field_schema=models.PayloadSchemaType.INTEGER,
    )
    client.create_payload_index(
        collection_name=collection_name,
        field_name=UUID_KEY,
        field_schema=models.PayloadSchemaType.UUID,
    )
    client.create_payload_index(
        collection_name=collection_name,
        field_name=STRING_KEY,
        field_schema=models.PayloadSchemaType.KEYWORD,
    )
    client.create_payload_index(
        collection_name=collection_name,
        field_name=STRING_ID_KEY,
        field_schema=models.PayloadSchemaType.KEYWORD,
    )
    client.create_payload_index(
        collection_name=collection_name,
        field_name=BOOL_KEY,
        field_schema=models.PayloadSchemaType.BOOL,
    )
    return client


@pytest.fixture(scope="module", autouse=True)
def grpc_client(fixture_points: list[models.PointStruct]) -> QdrantClient:
    client = init_remote(prefer_grpc=True)
    return client


def test_minimal(
    local_client: QdrantBase,
    http_client: QdrantBase,
    grpc_client: QdrantBase,
    collection_name: str,
):
    def f(
        client: QdrantBase, facet_key: str, collection_name: str = COLLECTION_NAME, **kwargs: Any
    ) -> models.FacetResponse:
        return client.facet(
            collection_name=collection_name,
            key=facet_key,
        )

    for key in all_facet_keys():
        compare_client_results(
            grpc_client, http_client, f, facet_key=key, collection_name=collection_name
        )
        compare_client_results(
            local_client, http_client, f, facet_key=key, collection_name=collection_name
        )


def test_limit(
    local_client: QdrantBase,
    http_client: QdrantBase,
    grpc_client: QdrantBase,
    collection_name: str,
):
    def f(
        client: QdrantBase,
        facet_key: str,
        limit: int,
        collection_name: str = COLLECTION_NAME,
        **kwargs: Any,
    ) -> models.FacetResponse:
        return client.facet(
            collection_name=collection_name,
            key=facet_key,
            limit=limit,
        )

    for _ in range(10):
        rand_num = random.randint(1, 100)
        for key in all_facet_keys():
            compare_client_results(
                grpc_client,
                http_client,
                f,
                facet_key=key,
                limit=rand_num,
                collection_name=collection_name,
            )
            compare_client_results(
                local_client,
                http_client,
                f,
                facet_key=key,
                limit=rand_num,
                collection_name=collection_name,
            )


def test_exact(
    local_client: QdrantBase,
    http_client: QdrantBase,
    grpc_client: QdrantBase,
    collection_name: str,
):
    def f(
        client: QdrantBase, facet_key: str, collection_name: str = COLLECTION_NAME, **kwargs: Any
    ) -> models.FacetResponse:
        return client.facet(
            collection_name=collection_name,
            key=facet_key,
            limit=5000,
            exact=True,
        )

    for key in all_facet_keys():
        compare_client_results(
            grpc_client, http_client, f, facet_key=key, collection_name=collection_name
        )
        compare_client_results(
            local_client, http_client, f, facet_key=key, collection_name=collection_name
        )


def test_filtered(
    local_client: QdrantBase,
    http_client: QdrantBase,
    grpc_client: QdrantBase,
    collection_name: str,
):
    def f(
        client: QdrantBase,
        facet_key: str,
        facet_filter: models.Filter,
        collection_name: str = COLLECTION_NAME,
        **kwargs: Any,
    ) -> models.FacetResponse:
        return client.facet(
            collection_name=collection_name,
            key=facet_key,
            facet_filter=facet_filter,
            exact=False,
        )

    for key in all_facet_keys():
        filter_ = one_random_filter_please()
        for _ in range(10):
            compare_client_results(
                grpc_client,
                http_client,
                f,
                facet_key=key,
                facet_filter=filter_,
                collection_name=collection_name,
            )
            compare_client_results(
                local_client,
                http_client,
                f,
                facet_key=key,
                facet_filter=filter_,
                collection_name=collection_name,
            )


def test_exact_filtered(
    local_client: QdrantBase,
    http_client: QdrantBase,
    grpc_client: QdrantBase,
    collection_name: str,
):
    def f(
        client: QdrantBase,
        facet_key: str,
        facet_filter: models.Filter,
        collection_name: str = COLLECTION_NAME,
        **kwargs: Any,
    ) -> models.FacetResponse:
        return client.facet(
            collection_name=collection_name,
            key=facet_key,
            limit=5000,
            exact=True,
            facet_filter=facet_filter,
        )

    for key in all_facet_keys():
        for _ in range(10):
            filter_ = one_random_filter_please()
            compare_client_results(
                grpc_client,
                http_client,
                f,
                facet_key=key,
                facet_filter=filter_,
                collection_name=collection_name,
            )
            compare_client_results(
                local_client,
                http_client,
                f,
                facet_key=key,
                facet_filter=filter_,
                collection_name=collection_name,
            )


def test_other_types_in_local(collection_name: str):
    client = init_local()
    client.create_collection(collection_name=collection_name, vectors_config={})
    client.upsert(
        collection_name=collection_name,
        points=[models.PointStruct(id=1, vector={}, payload={"a": True})],
    )
    client.upsert(
        collection_name=collection_name,
        points=[models.PointStruct(id=2, vector={}, payload={"a": 12.444})],
    )
    client.upsert(
        collection_name=collection_name,
        points=[models.PointStruct(id=3, vector={}, payload={"a": {"b": 1}})],
    )

    # Assertion is that it doesn't raise an exception
    client.facet(collection_name=collection_name, key="a")
