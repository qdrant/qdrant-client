from typing import List

from qdrant_client import models
from qdrant_client.client_base import QdrantBase
from tests.congruence_tests.test_common import (
    COLLECTION_NAME,
    compare_client_results,
    generate_fixtures,
    init_client,
    init_local,
    init_remote,
)


def scroll_all_with_key(client: QdrantBase, key: str) -> List[models.Record]:
    all_records = []

    last_seen_value = None

    last_value_ids = []

    while True:
        records, _next_page = client.scroll(
            collection_name=COLLECTION_NAME,
            limit=20,
            order_by=models.OrderBy(**{"key": key, "start_from": last_seen_value}),
            scroll_filter=models.Filter(**{"must_not": [{"has_id": last_value_ids}]}),
            with_payload=True,
        )

        if len(records) == 0:
            break

        last_value = records[-1].payload[key]
        if last_seen_value != last_value:
            last_seen_value = last_value
            last_value_ids = []

        last_value_ids.extend(
            [record.id for record in records if record.payload[key] == last_seen_value]
        )

        all_records.extend(records)

    return all_records


def scroll_all_floats(client: QdrantBase) -> List[models.Record]:
    return scroll_all_with_key(client, "rand_number")


def scroll_all_datetimes(client: QdrantBase) -> List[models.Record]:
    return scroll_all_with_key(client, "rand_datetime")


def test_simple_scroll() -> None:
    fixture_points = generate_fixtures(200)

    local_client = init_local()
    init_client(local_client, fixture_points)

    http_client = init_remote()
    init_client(http_client, fixture_points)
    http_client.create_payload_index(
        COLLECTION_NAME, "rand_number", models.PayloadSchemaType.FLOAT, wait=True
    )
    http_client.create_payload_index(
        COLLECTION_NAME, "rand_datetime", models.PayloadSchemaType.DATETIME, wait=True
    )

    grpc_client = init_remote(prefer_grpc=True)

    compare_client_results(grpc_client, http_client, scroll_all_floats)
    compare_client_results(local_client, http_client, scroll_all_floats)

    compare_client_results(grpc_client, http_client, scroll_all_datetimes)
    compare_client_results(local_client, http_client, scroll_all_datetimes)
