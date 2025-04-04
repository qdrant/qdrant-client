import random
from typing import Any, Optional

import numpy as np
import pytest

from qdrant_client import QdrantClient, models
from qdrant_client.client_base import QdrantBase
from qdrant_client.http.exceptions import UnexpectedResponse
from tests.congruence_tests.test_common import (
    COLLECTION_NAME,
    compare_client_results,
    generate_multivector_fixtures,
    image_vector_size,
    init_client,
    init_local,
    init_remote,
    multi_vector_config,
)
from tests.fixtures.points import generate_random_multivector

secondary_collection_name = "congruence_secondary_collection"


NUM_MULTI_VECTORS = 100


@pytest.fixture(scope="module")
def fixture_points() -> list[models.PointStruct]:
    return generate_multivector_fixtures(NUM_MULTI_VECTORS)


@pytest.fixture(scope="module")
def secondary_collection_points() -> list[models.PointStruct]:
    return generate_multivector_fixtures(50)


@pytest.fixture(scope="module", autouse=True)
def local_client(fixture_points, secondary_collection_points) -> QdrantClient:
    client = init_local()
    init_client(client, fixture_points, vectors_config=multi_vector_config)
    init_client(
        client,
        secondary_collection_points,
        secondary_collection_name,
        vectors_config=multi_vector_config,
    )
    return client


@pytest.fixture(scope="module", autouse=True)
def http_client(fixture_points, secondary_collection_points) -> QdrantClient:
    client = init_remote()
    init_client(client, fixture_points, vectors_config=multi_vector_config)
    init_client(
        client,
        secondary_collection_points,
        secondary_collection_name,
        vectors_config=multi_vector_config,
    )
    return client


@pytest.fixture(scope="module", autouse=True)
def grpc_client(fixture_points, secondary_collection_points) -> QdrantClient:
    client = init_remote(prefer_grpc=True)
    return client


def test_context_cosine(
    local_client,
    http_client,
    grpc_client,
):
    def f(client: QdrantBase, **kwargs: dict[str, Any]) -> list[models.ScoredPoint]:
        return client.query_points(
            collection_name=COLLECTION_NAME,
            query=models.ContextQuery(context=[models.ContextPair(positive=10, negative=19)]),
            with_payload=True,
            limit=NUM_MULTI_VECTORS,
            using="multi-text",
        ).points

    compare_client_results(grpc_client, http_client, f, is_context_search=True)
    compare_client_results(local_client, http_client, f, is_context_search=True)


def test_context_dot(
    local_client,
    http_client,
    grpc_client,
):
    def f(client: QdrantBase, **kwargs: dict[str, Any]) -> list[models.ScoredPoint]:
        return client.query_points(
            collection_name=COLLECTION_NAME,
            query=models.ContextQuery(context=[models.ContextPair(positive=10, negative=19)]),
            with_payload=True,
            limit=NUM_MULTI_VECTORS,
            using="multi-image",
        ).points

    compare_client_results(grpc_client, http_client, f, is_context_search=True)
    compare_client_results(local_client, http_client, f, is_context_search=True)


def test_context_euclidean(
    local_client,
    http_client,
    grpc_client,
):
    def f(client: QdrantBase, **kwargs: dict[str, Any]) -> list[models.ScoredPoint]:
        return client.query_points(
            collection_name=COLLECTION_NAME,
            query=models.ContextQuery(context=[models.ContextPair(positive=11, negative=19)]),
            with_payload=True,
            limit=NUM_MULTI_VECTORS,
            using="multi-code",
        ).points

    compare_client_results(grpc_client, http_client, f, is_context_search=True)
    compare_client_results(local_client, http_client, f, is_context_search=True)


def test_context_many_pairs(
    local_client,
    http_client,
    grpc_client,
):
    random_image_multivector_1 = generate_random_multivector(
        image_vector_size, random.randint(2, 30)
    )
    random_image_multivector_2 = generate_random_multivector(
        image_vector_size, random.randint(2, 30)
    )

    def f(client: QdrantBase, **kwargs: dict[str, Any]) -> list[models.ScoredPoint]:
        return client.query_points(
            collection_name=COLLECTION_NAME,
            query=models.ContextQuery(
                context=[
                    models.ContextPair(positive=11, negative=19),
                    models.ContextPair(positive=42, negative=50),
                    models.ContextPair(
                        positive=random_image_multivector_1, negative=random_image_multivector_2
                    ),
                    models.ContextPair(positive=30, negative=random_image_multivector_2),
                    models.ContextPair(positive=random_image_multivector_1, negative=15),
                ]
            ),
            with_payload=True,
            limit=NUM_MULTI_VECTORS,
            using="multi-image",
        ).points

    compare_client_results(grpc_client, http_client, f, is_context_search=True)
    compare_client_results(local_client, http_client, f, is_context_search=True)


def test_discover_cosine(
    local_client,
    http_client,
    grpc_client,
):
    def f(client: QdrantBase, **kwargs: dict[str, Any]) -> list[models.ScoredPoint]:
        return client.query_points(
            collection_name=COLLECTION_NAME,
            query=models.DiscoverQuery(
                discover=models.DiscoverInput(
                    target=10, context=[models.ContextPair(positive=11, negative=19)]
                )
            ),
            with_payload=True,
            limit=10,
            using="multi-text",
        ).points

    compare_client_results(grpc_client, http_client, f)
    compare_client_results(local_client, http_client, f)


def test_discover_dot(
    local_client,
    http_client,
    grpc_client,
):
    def f(client: QdrantBase, **kwargs: dict[str, Any]) -> list[models.ScoredPoint]:
        return client.query_points(
            collection_name=COLLECTION_NAME,
            query=models.DiscoverQuery(
                discover=models.DiscoverInput(
                    target=10, context=models.ContextPair(positive=11, negative=19)
                )
            ),
            with_payload=True,
            limit=10,
            using="multi-image",
        ).points

    compare_client_results(grpc_client, http_client, f)
    compare_client_results(local_client, http_client, f)


def test_discover_euclidean(
    local_client,
    http_client,
    grpc_client,
):
    def f(client: QdrantBase, **kwargs: dict[str, Any]) -> list[models.ScoredPoint]:
        return client.query_points(
            collection_name=COLLECTION_NAME,
            query=models.DiscoverQuery(
                discover=models.DiscoverInput(
                    target=10, context=[models.ContextPair(positive=11, negative=19)]
                )
            ),
            with_payload=True,
            limit=10,
            using="multi-code",
        ).points

    compare_client_results(grpc_client, http_client, f)
    compare_client_results(local_client, http_client, f)


def test_discover_raw_target(
    local_client,
    http_client,
    grpc_client,
):
    random_image_multivector = generate_random_multivector(
        image_vector_size, random.randint(2, 30)
    )

    def f(client: QdrantBase, **kwargs: dict[str, Any]) -> list[models.ScoredPoint]:
        return client.query_points(
            collection_name=COLLECTION_NAME,
            query=models.DiscoverQuery(
                discover=models.DiscoverInput(
                    target=random_image_multivector,
                    context=models.ContextPair(positive=10, negative=19),
                )
            ),
            limit=10,
            using="multi-image",
        ).points

    compare_client_results(grpc_client, http_client, f)
    compare_client_results(local_client, http_client, f)


def test_context_raw_positive(
    local_client,
    http_client,
    grpc_client,
):
    random_image_multivector = generate_random_multivector(
        image_vector_size, random.randint(2, 30)
    )

    def f(client: QdrantBase, **kwargs: dict[str, Any]) -> list[models.ScoredPoint]:
        return client.query_points(
            collection_name=COLLECTION_NAME,
            query=models.ContextQuery(
                context=[models.ContextPair(positive=random_image_multivector, negative=19)]
            ),
            limit=NUM_MULTI_VECTORS,
            using="multi-image",
        ).points

    compare_client_results(grpc_client, http_client, f, is_context_search=True)
    compare_client_results(local_client, http_client, f, is_context_search=True)


def test_only_target(
    local_client,
    http_client,
    grpc_client,
):
    def f(client: QdrantBase, **kwargs: dict[str, Any]) -> list[models.ScoredPoint]:
        return client.query_points(
            collection_name=COLLECTION_NAME,
            query=models.DiscoverQuery(discover=models.DiscoverInput(target=10, context=[])),
            with_payload=True,
            limit=10,
            using="multi-image",
        ).points

    compare_client_results(grpc_client, http_client, f)
    compare_client_results(local_client, http_client, f)


def discover_from_another_collection(
    client: QdrantBase,
    collection_name=COLLECTION_NAME,
    lookup_collection_name=secondary_collection_name,
    positive_point_id: Optional[int] = None,
    **kwargs: dict[str, Any],
) -> list[models.ScoredPoint]:
    return client.query_points(
        collection_name=collection_name,
        query=models.DiscoverQuery(
            discover=models.DiscoverInput(
                target=5,
                context=(
                    [models.ContextPair(positive=positive_point_id, negative=6)]
                    if positive_point_id is not None
                    else []
                ),
            )
        ),
        with_payload=True,
        limit=10,
        using="multi-image",
        lookup_from=models.LookupLocation(
            collection=lookup_collection_name,
            vector="multi-image",
        ),
    ).points


def test_discover_from_another_collection(
    local_client,
    http_client,
    grpc_client,
):
    compare_client_results(grpc_client, http_client, discover_from_another_collection)
    compare_client_results(local_client, http_client, discover_from_another_collection)


def test_discover_from_another_collection_id_exclusion():
    fixture_points = generate_multivector_fixtures(10)

    secondary_collection_points = generate_multivector_fixtures(10)

    local_client = init_local()
    collection_name = COLLECTION_NAME + "_small"
    lookup_collection_name = secondary_collection_name + "_small"
    init_client(
        local_client,
        fixture_points,
        collection_name=collection_name,
        vectors_config=multi_vector_config,
    )
    init_client(
        local_client,
        secondary_collection_points,
        collection_name=lookup_collection_name,
        vectors_config=multi_vector_config,
    )

    remote_client = init_remote()
    init_client(
        remote_client,
        fixture_points,
        collection_name=collection_name,
        vectors_config=multi_vector_config,
    )
    init_client(
        remote_client,
        secondary_collection_points,
        collection_name=lookup_collection_name,
        vectors_config=multi_vector_config,
    )

    for i in range(10):
        compare_client_results(
            local_client,
            remote_client,
            discover_from_another_collection,
            positive_point_id=i,
            collection_name=collection_name,
            lookup_collection_name=lookup_collection_name,
        )


def test_discover_batch(
    local_client,
    http_client,
    grpc_client,
):
    def f(client: QdrantBase, **kwargs: dict[str, Any]) -> list[list[models.ScoredPoint]]:
        return [
            response.points
            for response in client.query_batch_points(
                collection_name=COLLECTION_NAME,
                requests=[
                    models.QueryRequest(
                        query=models.DiscoverQuery(
                            discover=models.DiscoverInput(
                                target=10, context=[models.ContextPair(positive=15, negative=7)]
                            )
                        ),
                        limit=5,
                        using="multi-image",
                    ),
                    models.QueryRequest(
                        query=models.DiscoverQuery(
                            discover=models.DiscoverInput(
                                target=11, context=[models.ContextPair(positive=15, negative=17)]
                            )
                        ),
                        limit=6,
                        using="multi-image",
                        lookup_from=models.LookupLocation(
                            collection=secondary_collection_name,
                            vector="multi-image",
                        ),
                    ),
                ],
            )
        ]

    compare_client_results(grpc_client, http_client, f)
    compare_client_results(local_client, http_client, f)


def test_query_with_nan():
    fixture_points = generate_multivector_fixtures(20)
    vector = generate_random_multivector(image_vector_size, random.randint(2, 30))
    vector[0][1] = np.nan
    using = "multi-image"

    local_client = init_local()
    remote_client = init_remote()

    init_client(local_client, fixture_points, vectors_config=multi_vector_config)
    init_client(remote_client, fixture_points, vectors_config=multi_vector_config)

    with pytest.raises(AssertionError):
        local_client.query_points(
            collection_name=COLLECTION_NAME,
            query=models.DiscoverQuery(discover=models.DiscoverInput(target=vector, context=[])),
            using=using,
        )
    with pytest.raises(UnexpectedResponse):
        remote_client.query_points(
            collection_name=COLLECTION_NAME,
            query=models.DiscoverQuery(discover=models.DiscoverInput(target=vector, context=[])),
            using=using,
        )
    with pytest.raises(AssertionError):
        local_client.query_points(
            collection_name=COLLECTION_NAME,
            query=models.ContextQuery(context=[models.ContextPair(positive=vector, negative=1)]),
            using=using,
        )
    with pytest.raises(UnexpectedResponse):
        remote_client.query_points(
            collection_name=COLLECTION_NAME,
            query=models.ContextQuery(context=[models.ContextPair(positive=1, negative=vector)]),
            using=using,
        )
    with pytest.raises(AssertionError):
        local_client.query_points(
            collection_name=COLLECTION_NAME,
            query=models.ContextQuery(context=[models.ContextPair(positive=vector, negative=1)]),
            using=using,
        )
    with pytest.raises(UnexpectedResponse):
        remote_client.query_points(
            collection_name=COLLECTION_NAME,
            query=models.ContextQuery(context=[models.ContextPair(positive=1, negative=vector)]),
            using=using,
        )
