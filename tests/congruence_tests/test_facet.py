import random

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
    client.create_payload_index(
        collection_name=COLLECTION_NAME,
        field_name=BOOL_KEY,
        field_schema=models.PayloadSchemaType.BOOL,
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


def test_other_types_in_local():
    collection_name = "test_collection"
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


# `False == 0` and `True == 1` in python, so a key holding both bools and ints is where
# local mode risks merging them into a single bucket. Every key below carries the same
# mixed values, but gets a different payload index on the server: (key, the index to
# build on it, the type of value that faceting through that index yields).
MIXED_TYPE_INDEXES = [
    ("mixed_bool", models.PayloadSchemaType.BOOL, bool),
    ("mixed_int", models.PayloadSchemaType.INTEGER, int),
    ("mixed_str", models.PayloadSchemaType.KEYWORD, str),
]
MIXED_TYPE_VALUES = [[False, 0], True, 1, "0", [True, 1], [True, True]]


def test_mixed_scalar_types():
    """Facet a key whose values span bools, ints and strings.

    A facet on the server reads a single payload index, so it only ever returns values
    of that index's type. Local mode facets the raw payload and returns every type at
    once, so its hits are compared per type against the matching indexed key.
    """
    collection_name = f"{COLLECTION_NAME}_mixed_facet"
    points = [
        models.PointStruct(
            id=idx,
            vector=[0.1, 0.2],
            payload={key: values for key, _, _ in MIXED_TYPE_INDEXES},
        )
        for idx, values in enumerate(MIXED_TYPE_VALUES)
    ]
    vectors_config = models.VectorParams(size=2, distance=models.Distance.DOT)

    local_client = init_local()
    init_client(local_client, points, collection_name, vectors_config=vectors_config)

    remote_client = init_remote()
    init_client(remote_client, points, collection_name, vectors_config=vectors_config)
    for key, schema, _ in MIXED_TYPE_INDEXES:
        remote_client.create_payload_index(collection_name, key, field_schema=schema)

    def hits(client: QdrantBase, facet_key: str) -> list[models.FacetValueHit]:
        return client.facet(
            collection_name=collection_name, key=facet_key, limit=100, exact=True
        ).hits

    for key, _, value_type in MIXED_TYPE_INDEXES:
        remote_hits = hits(remote_client, key)
        local_hits = [hit for hit in hits(local_client, key) if type(hit.value) is value_type]

        # compare the types explicitly: pydantic considers FacetValueHit(value=True)
        # and FacetValueHit(value=1) equal, which is the very thing under test here
        assert [(type(hit.value), hit.value, hit.count) for hit in local_hits] == [
            (type(hit.value), hit.value, hit.count) for hit in remote_hits
        ]
        assert remote_hits, f"no {value_type.__name__} values were faceted for {key}"
