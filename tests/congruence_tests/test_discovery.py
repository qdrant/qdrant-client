from typing import Any, List

import numpy as np

from qdrant_client.client_base import QdrantBase
from qdrant_client.http.models import models
from tests.congruence_tests.test_common import (
    COLLECTION_NAME,
    code_vector_size,
    compare_client_results,
    generate_fixtures,
    image_vector_size,
    init_client,
    init_local,
    init_remote,
)
from tests.fixtures.filters import one_random_filter_please

secondary_collection_name = "secondary_collection"


def random_code_vector() -> List[float]:
    return np.random.random(code_vector_size).round(3).tolist()


def random_image_vector() -> List[float]:
    return np.random.random(image_vector_size).round(3).tolist()


class TestDiscovery:
    __test__ = False

    def __init__(self):
        self.query_image = np.random.random(image_vector_size).tolist()

    @classmethod
    def only_target(cls, client: QdrantBase) -> List[models.ScoredPoint]:
        return client.discover(
            collection_name=COLLECTION_NAME,
            target=10,
            with_payload=True,
            limit=10,
            using="image",
        )

    @classmethod
    def only_context(
        cls, client: QdrantBase, **kwargs: dict[str, Any]
    ) -> List[models.ScoredPoint]:
        return client.discover(
            collection_name=COLLECTION_NAME,
            context_pairs=[(10, 19)],
            with_payload=True,
            limit=1000,
            using="image",
        )

    @classmethod
    def both_target_and_context(cls, client: QdrantBase) -> List[models.ScoredPoint]:
        return client.discover(
            collection_name=COLLECTION_NAME,
            target=10,
            context_pairs=[(10, 19)],
            with_payload=True,
            limit=10,
            using="image",
        )

    @classmethod
    def context_euclidean(
        cls, client: QdrantBase, **kwargs: dict[str, Any]
    ) -> List[models.ScoredPoint]:
        return client.discover(
            collection_name=COLLECTION_NAME,
            context_pairs=[(random_code_vector(), 19)],
            with_payload=True,
            limit=1000,
            using="code",
        )

    @classmethod
    def discover_euclidean(cls, client: QdrantBase) -> List[models.ScoredPoint]:
        return client.discover(
            collection_name=COLLECTION_NAME,
            target=10,
            context_pairs=[(random_code_vector(), 19)],
            with_payload=True,
            limit=10,
            using="code",
        )

    @classmethod
    def discover_from_another_collection(cls, client: QdrantBase) -> List[models.ScoredPoint]:
        return client.discover(
            collection_name=COLLECTION_NAME,
            target=10,
            context_pairs=[(15, 7)],
            with_payload=True,
            limit=10,
            using="image",
            lookup_from=models.LookupLocation(
                collection=secondary_collection_name,
                vector="image",
            ),
        )

    @classmethod
    def filter_discover_code(
        cls, client: QdrantBase, query_filter: models.Filter
    ) -> List[models.ScoredPoint]:
        return client.discover(
            collection_name=COLLECTION_NAME,
            target=10,
            context_pairs=[(15, random_image_vector())],
            query_filter=query_filter,
            with_payload=True,
            limit=10,
            using="image",
        )

    @staticmethod
    def discover_batch(client: QdrantBase) -> List[List[models.ScoredPoint]]:
        return client.discover_batch(
            collection_name=COLLECTION_NAME,
            requests=[
                models.DiscoverRequest(
                    target=10,
                    context_pairs=[[15, 7]],
                    limit=1,
                    using="image",
                ),
                models.DiscoverRequest(
                    target=11,
                    context_pairs=[[16, 17]],
                    limit=2,
                    using="image",
                    lookup_from=models.LookupLocation(
                        collection=secondary_collection_name,
                        vector="image",
                    ),
                ),
            ],
        )


def test_discovery() -> None:
    fixture_records = generate_fixtures()

    secondary_collection_records = generate_fixtures(100)

    searcher = TestDiscovery()

    local_client = init_local()
    init_client(local_client, fixture_records)
    init_client(local_client, secondary_collection_records, secondary_collection_name)

    remote_client = init_remote()
    init_client(remote_client, fixture_records)
    init_client(remote_client, secondary_collection_records, secondary_collection_name)

    compare_client_results(local_client, remote_client, searcher.only_target)
    compare_client_results(
        local_client, remote_client, searcher.only_context, is_context_search=True
    )
    compare_client_results(local_client, remote_client, searcher.both_target_and_context)
    compare_client_results(
        local_client, remote_client, searcher.context_euclidean, is_context_search=True
    )
    compare_client_results(local_client, remote_client, searcher.discover_euclidean)
    compare_client_results(local_client, remote_client, searcher.discover_from_another_collection)
    compare_client_results(local_client, remote_client, searcher.discover_batch)

    for _ in range(10):
        query_filter = one_random_filter_please()
        try:
            compare_client_results(
                local_client,
                remote_client,
                searcher.filter_discover_code,
                query_filter=query_filter,
            )
        except AssertionError as e:
            print(f"\nFailed with filter {query_filter}")
            raise e
