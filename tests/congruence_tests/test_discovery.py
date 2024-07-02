from typing import Any, Dict, List, Optional

import numpy as np
import pytest

from qdrant_client import QdrantClient, models
from qdrant_client.client_base import QdrantBase
from qdrant_client.http.exceptions import UnexpectedResponse
from tests.congruence_tests.test_common import (
    COLLECTION_NAME,
    compare_client_results,
    generate_fixtures,
    image_vector_size,
    init_client,
    init_local,
    init_remote,
)
from tests.fixtures.filters import one_random_filter_please

secondary_collection_name = "congruence_secondary_collection"


def random_vector(dims: int) -> List[float]:
    return np.random.random(dims).round(3).tolist()


@pytest.fixture(scope="module")
def fixture_points() -> List[models.PointStruct]:
    return generate_fixtures()


@pytest.fixture(scope="module")
def secondary_collection_points() -> List[models.PointStruct]:
    return generate_fixtures(100)


@pytest.fixture(scope="module", autouse=True)
def local_client(fixture_points, secondary_collection_points) -> QdrantClient:
    client = init_local()
    init_client(client, fixture_points)
    init_client(client, secondary_collection_points, secondary_collection_name)
    return client


@pytest.fixture(scope="module", autouse=True)
def http_client(fixture_points, secondary_collection_points) -> QdrantClient:
    client = init_remote()
    init_client(client, fixture_points)
    init_client(client, secondary_collection_points, secondary_collection_name)
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
    def f(client: QdrantBase, **kwargs: Dict[str, Any]) -> List[models.ScoredPoint]:
        return client.discover(
            collection_name=COLLECTION_NAME,
            context=[models.ContextExamplePair(positive=10, negative=19)],
            with_payload=True,
            limit=1000,
            using="image",
        )

    compare_client_results(grpc_client, http_client, f, is_context_search=True)
    compare_client_results(local_client, http_client, f, is_context_search=True)


def test_context_dot(
    local_client,
    http_client,
    grpc_client,
):
    def f(client: QdrantBase, **kwargs: Dict[str, Any]) -> List[models.ScoredPoint]:
        return client.discover(
            collection_name=COLLECTION_NAME,
            context=[models.ContextExamplePair(positive=10, negative=19)],
            with_payload=True,
            limit=1000,
            using="text",
        )

    compare_client_results(grpc_client, http_client, f, is_context_search=True)
    compare_client_results(local_client, http_client, f, is_context_search=True)


def test_context_euclidean(
    local_client,
    http_client,
    grpc_client,
):
    def f(client: QdrantBase, **kwargs: Dict[str, Any]) -> List[models.ScoredPoint]:
        return client.discover(
            collection_name=COLLECTION_NAME,
            context=[models.ContextExamplePair(positive=11, negative=19)],
            with_payload=True,
            limit=1000,
            using="code",
        )

    compare_client_results(grpc_client, http_client, f, is_context_search=True)
    compare_client_results(local_client, http_client, f, is_context_search=True)


def test_context_many_pairs(
    local_client,
    http_client,
    grpc_client,
):
    random_image_vector_1 = random_vector(image_vector_size)
    random_image_vector_2 = random_vector(image_vector_size)

    def f(client: QdrantBase, **kwargs: Dict[str, Any]) -> List[models.ScoredPoint]:
        return client.discover(
            collection_name=COLLECTION_NAME,
            context=[
                models.ContextExamplePair(positive=11, negative=19),
                models.ContextExamplePair(positive=400, negative=200),
                models.ContextExamplePair(
                    positive=random_image_vector_1, negative=random_image_vector_2
                ),
                models.ContextExamplePair(positive=30, negative=random_image_vector_2),
                models.ContextExamplePair(positive=random_image_vector_1, negative=15),
            ],
            with_payload=True,
            limit=1000,
            using="image",
        )

    compare_client_results(grpc_client, http_client, f, is_context_search=True)
    compare_client_results(local_client, http_client, f, is_context_search=True)


def test_discover_cosine(
    local_client,
    http_client,
    grpc_client,
):
    def f(client: QdrantBase, **kwargs: Dict[str, Any]) -> List[models.ScoredPoint]:
        return client.discover(
            collection_name=COLLECTION_NAME,
            target=10,
            context=[models.ContextExamplePair(positive=11, negative=19)],
            with_payload=True,
            limit=10,
            using="image",
        )

    compare_client_results(grpc_client, http_client, f)
    compare_client_results(local_client, http_client, f)


def test_discover_dot(
    local_client,
    http_client,
    grpc_client,
):
    def f(client: QdrantBase, **kwargs: Dict[str, Any]) -> List[models.ScoredPoint]:
        return client.discover(
            collection_name=COLLECTION_NAME,
            target=10,
            context=[models.ContextExamplePair(positive=11, negative=19)],
            with_payload=True,
            limit=10,
            using="text",
        )

    compare_client_results(grpc_client, http_client, f)
    compare_client_results(local_client, http_client, f)


def test_discover_euclidean(
    local_client,
    http_client,
    grpc_client,
):
    def f(client: QdrantBase, **kwargs: Dict[str, Any]) -> List[models.ScoredPoint]:
        return client.discover(
            collection_name=COLLECTION_NAME,
            target=10,
            context=[models.ContextExamplePair(positive=11, negative=19)],
            with_payload=True,
            limit=10,
            using="code",
        )

    compare_client_results(grpc_client, http_client, f)
    compare_client_results(local_client, http_client, f)


def test_discover_raw_target(
    local_client,
    http_client,
    grpc_client,
):
    random_image_vector = random_vector(image_vector_size)

    def f(client: QdrantBase, **kwargs: Dict[str, Any]) -> List[models.ScoredPoint]:
        return client.discover(
            collection_name=COLLECTION_NAME,
            target=random_image_vector,
            context=[models.ContextExamplePair(positive=10, negative=19)],
            limit=10,
            using="image",
        )

    compare_client_results(grpc_client, http_client, f)
    compare_client_results(local_client, http_client, f)


def test_context_raw_positive(
    local_client,
    http_client,
    grpc_client,
):
    random_image_vector = random_vector(image_vector_size)

    def f(client: QdrantBase, **kwargs: Dict[str, Any]) -> List[models.ScoredPoint]:
        return client.discover(
            collection_name=COLLECTION_NAME,
            target=10,
            context=[models.ContextExamplePair(positive=random_image_vector, negative=19)],
            limit=10,
            using="image",
        )

    compare_client_results(grpc_client, http_client, f)
    compare_client_results(local_client, http_client, f)


def test_only_target(
    local_client,
    http_client,
    grpc_client,
):
    def f(client: QdrantBase, **kwargs: Dict[str, Any]) -> List[models.ScoredPoint]:
        return client.discover(
            collection_name=COLLECTION_NAME,
            target=10,
            with_payload=True,
            limit=10,
            using="image",
        )

    compare_client_results(grpc_client, http_client, f)
    compare_client_results(local_client, http_client, f)


def discover_from_another_collection(
    client: QdrantBase,
    collection_name=COLLECTION_NAME,
    lookup_collection_name=secondary_collection_name,
    positive_point_id: Optional[int] = None,
    **kwargs: Dict[str, Any],
) -> List[models.ScoredPoint]:
    return client.discover(
        collection_name=collection_name,
        target=5,
        context=[models.ContextExamplePair(positive=3, negative=6)]
        if positive_point_id is None
        else [],
        with_payload=True,
        limit=10,
        using="image",
        lookup_from=models.LookupLocation(
            collection=lookup_collection_name,
            vector="image",
        ),
    )


def test_discover_from_another_collection(
    local_client,
    http_client,
    grpc_client,
):
    compare_client_results(grpc_client, http_client, discover_from_another_collection)
    compare_client_results(local_client, http_client, discover_from_another_collection)


def test_discover_from_another_collection_id_exclusion():
    fixture_points = generate_fixtures(10)

    secondary_collection_points = generate_fixtures(10)

    local_client = init_local()
    collection_name = COLLECTION_NAME + "_small"
    lookup_collection_name = secondary_collection_name + "_small"
    init_client(local_client, fixture_points, collection_name=collection_name)
    init_client(local_client, secondary_collection_points, collection_name=lookup_collection_name)

    remote_client = init_remote()
    init_client(remote_client, fixture_points, collection_name=collection_name)
    init_client(remote_client, secondary_collection_points, collection_name=lookup_collection_name)

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
    def f(client: QdrantBase, **kwargs: Dict[str, Any]) -> List[List[models.ScoredPoint]]:
        return client.discover_batch(
            collection_name=COLLECTION_NAME,
            requests=[
                models.DiscoverRequest(
                    target=10,
                    context=[models.ContextExamplePair(positive=15, negative=7)],
                    limit=5,
                    using="image",
                ),
                models.DiscoverRequest(
                    target=11,
                    context=[models.ContextExamplePair(positive=15, negative=17)],
                    limit=6,
                    using="image",
                    lookup_from=models.LookupLocation(
                        collection=secondary_collection_name,
                        vector="image",
                    ),
                ),
            ],
        )

    compare_client_results(grpc_client, http_client, f)
    compare_client_results(local_client, http_client, f)


@pytest.mark.parametrize("filter", [one_random_filter_please() for _ in range(10)])
def test_discover_with_filters(local_client, http_client, grpc_client, filter: models.Filter):
    def f(client: QdrantBase, **kwargs: Dict[str, Any]) -> List[models.ScoredPoint]:
        return client.discover(
            collection_name=COLLECTION_NAME,
            target=10,
            context=[models.ContextExamplePair(positive=15, negative=7)],
            limit=15,
            using="image",
            query_filter=filter,
        )


@pytest.mark.parametrize("filter", [one_random_filter_please() for _ in range(10)])
def test_context_with_filters(local_client, http_client, grpc_client, filter: models.Filter):
    def f(client: QdrantBase, **kwargs: Dict[str, Any]) -> List[models.ScoredPoint]:
        return client.discover(
            collection_name=COLLECTION_NAME,
            context=[models.ContextExamplePair(positive=15, negative=7)],
            limit=1000,
            using="image",
            query_filter=filter,
        )

    compare_client_results(grpc_client, http_client, f, is_context_search=True)
    compare_client_results(local_client, http_client, f, is_context_search=True)


def test_query_with_nan():
    fixture_points = generate_fixtures()
    vector = np.random.random(image_vector_size)
    vector[0] = np.nan
    vector = vector.tolist()
    using = "image"

    local_client = init_local()
    remote_client = init_remote()

    init_client(local_client, fixture_points)
    init_client(remote_client, fixture_points)

    with pytest.raises(AssertionError):
        local_client.discover(
            collection_name=COLLECTION_NAME,
            target=vector,
            using=using,
        )
    with pytest.raises(UnexpectedResponse):
        remote_client.discover(
            collection_name=COLLECTION_NAME,
            target=vector,
            using=using,
        )
    with pytest.raises(AssertionError):
        local_client.discover(
            collection_name=COLLECTION_NAME,
            context=[models.ContextExamplePair(positive=vector, negative=1)],
            using=using,
        )
    with pytest.raises(UnexpectedResponse):
        remote_client.discover(
            collection_name=COLLECTION_NAME,
            context=[models.ContextExamplePair(positive=vector, negative=1)],
            using=using,
        )
    with pytest.raises(AssertionError):
        local_client.discover(
            collection_name=COLLECTION_NAME,
            context=[models.ContextExamplePair(positive=1, negative=vector)],
            using=using,
        )
    with pytest.raises(UnexpectedResponse):
        remote_client.discover(
            collection_name=COLLECTION_NAME,
            context=[models.ContextExamplePair(positive=1, negative=vector)],
            using=using,
        )
