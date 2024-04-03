from typing import Any, Dict, List

import numpy as np
import pytest

from qdrant_client import QdrantClient, models
from qdrant_client.client_base import QdrantBase
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.http.models import ContextExamplePair
from tests.congruence_tests.test_common import (
    COLLECTION_NAME,
    compare_client_results,
    generate_sparse_fixtures,
    init_client,
    init_local,
    init_remote,
    sparse_image_vector_size,
    sparse_vectors_config,
)
from tests.fixtures.filters import one_random_filter_please
from tests.fixtures.points import random_sparse_vectors

secondary_collection_name = "congruence_secondary_collection"


@pytest.fixture(scope="module")
def fixture_points() -> List[models.PointStruct]:
    return generate_sparse_fixtures(200)


@pytest.fixture(scope="module")
def secondary_collection_points() -> List[models.PointStruct]:
    return generate_sparse_fixtures(100)


@pytest.fixture(scope="module", autouse=True)
def local_client(fixture_points, secondary_collection_points) -> QdrantClient:
    client = init_local()
    init_client(
        client, fixture_points, vectors_config={}, sparse_vectors_config=sparse_vectors_config
    )
    init_client(
        client,
        secondary_collection_points,
        secondary_collection_name,
        sparse_vectors_config=sparse_vectors_config,
    )
    return client


@pytest.fixture(scope="module", autouse=True)
def http_client(fixture_points, secondary_collection_points) -> QdrantClient:
    client = init_remote()
    init_client(
        client, fixture_points, vectors_config={}, sparse_vectors_config=sparse_vectors_config
    )
    init_client(
        client,
        secondary_collection_points,
        secondary_collection_name,
        vectors_config={},
        sparse_vectors_config=sparse_vectors_config,
    )
    return client


@pytest.fixture(scope="module", autouse=True)
def grpc_client(fixture_points, secondary_collection_points) -> QdrantClient:
    client = init_remote(prefer_grpc=True)
    return client


def test_context(
    local_client,
    http_client,
    grpc_client,
):
    def f(client: QdrantBase, **kwargs: Dict[str, Any]) -> List[models.ScoredPoint]:
        return client.discover(
            collection_name=COLLECTION_NAME,
            context=[models.ContextExamplePair(positive=10, negative=19)],
            with_payload=True,
            limit=200,
            using="sparse-image",
        )

    compare_client_results(grpc_client, http_client, f, is_context_search=True)
    compare_client_results(local_client, http_client, f, is_context_search=True)


def test_context_many_pairs(
    local_client,
    http_client,
    grpc_client,
):
    random_sparse_image_vector_1 = random_sparse_vectors(
        {"sparse-image": sparse_image_vector_size}
    )["sparse-image"]
    random_sparse_image_vector_2 = random_sparse_vectors(
        {"sparse-image": sparse_image_vector_size}
    )["sparse-image"]

    def f(client: QdrantBase, **kwargs: Dict[str, Any]) -> List[models.ScoredPoint]:
        return client.discover(
            collection_name=COLLECTION_NAME,
            context=[
                models.ContextExamplePair(positive=11, negative=19),
                models.ContextExamplePair(positive=100, negative=199),
                models.ContextExamplePair(
                    positive=random_sparse_image_vector_1, negative=random_sparse_image_vector_2
                ),
                models.ContextExamplePair(positive=30, negative=random_sparse_image_vector_2),
                models.ContextExamplePair(positive=random_sparse_image_vector_1, negative=15),
            ],
            with_payload=True,
            limit=200,
            using="sparse-image",
        )

    compare_client_results(grpc_client, http_client, f, is_context_search=True)
    compare_client_results(local_client, http_client, f, is_context_search=True)


def test_discover(
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
            limit=100,
            using="sparse-image",
        )

    compare_client_results(grpc_client, http_client, f)
    compare_client_results(local_client, http_client, f)


def test_discover_raw_target(
    local_client,
    http_client,
    grpc_client,
):
    random_sparse_image_vector = random_sparse_vectors({"sparse-image": sparse_image_vector_size})[
        "sparse-image"
    ]

    def f(client: QdrantBase, **kwargs: Dict[str, Any]) -> List[models.ScoredPoint]:
        return client.discover(
            collection_name=COLLECTION_NAME,
            target=random_sparse_image_vector,
            context=[models.ContextExamplePair(positive=10, negative=19)],
            limit=100,
            using="sparse-image",
        )

    compare_client_results(grpc_client, http_client, f)
    compare_client_results(local_client, http_client, f)


def test_context_raw_positive(
    local_client,
    http_client,
    grpc_client,
):
    random_sparse_image_vector = random_sparse_vectors({"sparse-image": sparse_image_vector_size})[
        "sparse-image"
    ]

    def f(client: QdrantBase, **kwargs: Dict[str, Any]) -> List[models.ScoredPoint]:
        return client.discover(
            collection_name=COLLECTION_NAME,
            target=10,
            context=[models.ContextExamplePair(positive=random_sparse_image_vector, negative=19)],
            limit=10,
            using="sparse-image",
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
            using="sparse-image",
        )

    compare_client_results(grpc_client, http_client, f)
    compare_client_results(local_client, http_client, f)


def test_discover_from_another_collection(
    local_client,
    http_client,
    grpc_client,
):
    def f(client: QdrantBase, **kwargs: Dict[str, Any]) -> List[models.ScoredPoint]:
        return client.discover(
            collection_name=COLLECTION_NAME,
            target=10,
            context=[models.ContextExamplePair(positive=15, negative=7)],
            with_payload=True,
            limit=10,
            using="sparse-image",
            lookup_from=models.LookupLocation(
                collection=secondary_collection_name,
                vector="sparse-image",
            ),
        )

    compare_client_results(grpc_client, http_client, f)
    compare_client_results(local_client, http_client, f)


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
                    using="sparse-image",
                ),
                models.DiscoverRequest(
                    target=11,
                    context=[models.ContextExamplePair(positive=15, negative=17)],
                    limit=6,
                    using="sparse-image",
                    lookup_from=models.LookupLocation(
                        collection=secondary_collection_name,
                        vector="sparse-image",
                    ),
                ),
            ],
        )

    compare_client_results(grpc_client, http_client, f)
    compare_client_results(local_client, http_client, f)


@pytest.mark.parametrize("filter_", [one_random_filter_please() for _ in range(10)])
def test_discover_with_filters(local_client, http_client, grpc_client, filter_: models.Filter):
    def f(client: QdrantBase, **kwargs: Dict[str, Any]) -> List[models.ScoredPoint]:
        return client.discover(
            collection_name=COLLECTION_NAME,
            target=10,
            context=[models.ContextExamplePair(positive=15, negative=7)],
            limit=15,
            using="sparse-image",
            query_filter=filter_,
        )


@pytest.mark.parametrize("filter_", [one_random_filter_please() for _ in range(10)])
def test_context_with_filters(local_client, http_client, grpc_client, filter_: models.Filter):
    def f(client: QdrantBase, **kwargs: Dict[str, Any]) -> List[models.ScoredPoint]:
        return client.discover(
            collection_name=COLLECTION_NAME,
            context=[models.ContextExamplePair(positive=15, negative=7)],
            limit=200,
            using="sparse-image",
            query_filter=filter_,
        )

    compare_client_results(grpc_client, http_client, f, is_context_search=True)
    compare_client_results(local_client, http_client, f, is_context_search=True)


def test_query_with_nan():
    fixture_points = generate_sparse_fixtures()
    using = "sparse-image"

    local_client = init_local()
    remote_client = init_remote()

    init_client(
        local_client,
        fixture_points,
        vectors_config={},
        sparse_vectors_config=sparse_vectors_config,
    )
    init_client(
        remote_client,
        fixture_points,
        vectors_config={},
        sparse_vectors_config=sparse_vectors_config,
    )

    sparse_vector_dicts = [
        random_sparse_vectors({using: sparse_image_vector_size}) for _ in range(3)
    ]
    sparse_vectors = [sparse_vector_dict[using] for sparse_vector_dict in sparse_vector_dicts]

    sparse_vector_with_nan = sparse_vectors[0]
    sparse_vector_with_nan.values[0] = np.nan

    sparse_vector = sparse_vectors[1]
    sparse_vector_2 = sparse_vectors[2]

    for target, pos, neg in (
        [None, sparse_vector_with_nan, sparse_vector],
        [None, sparse_vector, sparse_vector_with_nan],
        [sparse_vector_with_nan, sparse_vector, sparse_vector_2],
    ):
        with pytest.raises(AssertionError):
            local_client.discover(
                collection_name=COLLECTION_NAME,
                target=target,
                context=[ContextExamplePair(positive=pos, negative=neg)],
                using=using,
            )

        with pytest.raises(UnexpectedResponse):
            remote_client.discover(
                collection_name=COLLECTION_NAME,
                target=target,
                context=[ContextExamplePair(positive=pos, negative=neg)],
                using=using,
            )
