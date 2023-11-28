import random
from typing import List

from qdrant_client.client_base import QdrantBase
from qdrant_client.http.models import models
from tests.congruence_tests.test_common import (
    COLLECTION_NAME,
    compare_client_results,
    generate_fixtures,
    generate_sparse_fixtures,
    init_client,
    init_local,
    init_remote
)


class TestSimpleScroller:
    @classmethod
    def scroll_all(cls, client: QdrantBase) -> List[models.Record]:
        all_records = []

        records, next_page = client.scroll(
            collection_name=COLLECTION_NAME,
            limit=10,
            with_payload=True,
        )
        all_records.extend(records)

        while next_page:
            records, next_page = client.scroll(
                collection_name=COLLECTION_NAME,
                limit=20,
                offset=next_page,
                with_payload=True,
            )
            all_records.extend(records)

        return all_records


def test_simple_search() -> None:
    fixture_records = generate_fixtures(200)

    scroller = TestSimpleScroller()

    local_client = init_local()
    init_client(local_client, fixture_records)

    remote_client = init_remote()
    init_client(remote_client, fixture_records)

    compare_client_results(local_client, remote_client, scroller.scroll_all)


def test_simple_sparse_search() -> None:
    fixture_records = generate_sparse_fixtures(200)

    scroller = TestSimpleScroller()

    local_client = init_local()
    init_client(local_client, fixture_records)

    remote_client = init_remote()
    init_client(remote_client, fixture_records)

    compare_client_results(local_client, remote_client, scroller.scroll_all)


def test_mixed_ids() -> None:
    fixture_records = generate_fixtures(100, random_ids=True) + generate_fixtures(
        100, random_ids=False
    )

    random.shuffle(fixture_records)

    scroller = TestSimpleScroller()

    local_client = init_local()
    init_client(local_client, fixture_records)

    remote_client = init_remote()
    init_client(remote_client, fixture_records)

    compare_client_results(local_client, remote_client, scroller.scroll_all)


def test_sparse_mixed_ids() -> None:
    fixture_records = generate_sparse_fixtures(100, random_ids=True) + generate_sparse_fixtures(
        100, random_ids=False
    )

    random.shuffle(fixture_records)

    scroller = TestSimpleScroller()

    local_client = init_local()
    init_client(local_client, fixture_records)

    remote_client = init_remote()
    init_client(remote_client, fixture_records)

    compare_client_results(local_client, remote_client, scroller.scroll_all)
