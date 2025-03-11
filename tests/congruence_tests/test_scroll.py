import random

from qdrant_client.client_base import QdrantBase
from qdrant_client.http.models import models
from tests.congruence_tests.test_common import (
    COLLECTION_NAME,
    compare_client_results,
    generate_fixtures,
    generate_sparse_fixtures,
    init_client,
    init_local,
    init_remote,
    sparse_vectors_config,
)


class TestSimpleScroller:
    @classmethod
    def scroll_all(
        cls, client: QdrantBase, collection_name: str = COLLECTION_NAME
    ) -> list[models.Record]:
        all_records = []

        records, next_page = client.scroll(
            collection_name=collection_name,
            limit=10,
            with_payload=True,
        )
        all_records.extend(records)

        while next_page:
            records, next_page = client.scroll(
                collection_name=collection_name,
                limit=20,
                offset=next_page,
                with_payload=True,
            )
            all_records.extend(records)

        return all_records


def test_simple_search(collection_name: str) -> None:
    fixture_points = generate_fixtures(200)

    scroller = TestSimpleScroller()

    local_client = init_local()
    init_client(local_client, fixture_points, collection_name)

    remote_client = init_remote()
    init_client(remote_client, fixture_points, collection_name)

    compare_client_results(
        local_client, remote_client, scroller.scroll_all, collection_name=collection_name
    )


def test_simple_sparse_scroll(collection_name: str) -> None:
    fixture_points = generate_sparse_fixtures(200)

    local_client = init_local()
    init_client(
        local_client, fixture_points, collection_name, sparse_vectors_config=sparse_vectors_config
    )

    remote_client = init_remote()
    init_client(
        remote_client, fixture_points, collection_name, sparse_vectors_config=sparse_vectors_config
    )

    scroller = TestSimpleScroller()

    compare_client_results(
        local_client, remote_client, scroller.scroll_all, collection_name=collection_name
    )


def test_mixed_ids(collection_name: str) -> None:
    fixture_points = generate_fixtures(100, random_ids=True) + generate_fixtures(
        100, random_ids=False
    )

    random.shuffle(fixture_points)

    scroller = TestSimpleScroller()

    local_client = init_local()
    init_client(local_client, fixture_points, collection_name)

    remote_client = init_remote()
    init_client(remote_client, fixture_points, collection_name)

    compare_client_results(
        local_client, remote_client, scroller.scroll_all, collection_name=collection_name
    )


def test_sparse_mixed_ids(collection_name: str) -> None:
    fixture_points = generate_sparse_fixtures(100, random_ids=True) + generate_sparse_fixtures(
        100, random_ids=False
    )

    random.shuffle(fixture_points)

    scroller = TestSimpleScroller()

    local_client = init_local()
    init_client(
        local_client, fixture_points, collection_name, sparse_vectors_config=sparse_vectors_config
    )

    remote_client = init_remote()
    init_client(
        remote_client, fixture_points, collection_name, sparse_vectors_config=sparse_vectors_config
    )

    compare_client_results(
        local_client, remote_client, scroller.scroll_all, collection_name=collection_name
    )
