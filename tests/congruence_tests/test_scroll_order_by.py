from datetime import date, datetime

from qdrant_client import models
from qdrant_client.client_base import QdrantBase
from qdrant_client.local import datetime_utils
from qdrant_client.local.order_by import to_order_value
from tests.congruence_tests.test_common import (
    COLLECTION_NAME,
    compare_client_results,
    generate_fixtures,
    init_client,
    init_local,
    init_remote,
)


def scroll_all_with_key(client: QdrantBase, key: str) -> list[models.Record]:
    all_records = []

    last_seen_value = None

    last_value_ids = []

    while True:
        if isinstance(last_seen_value, str):
            start_from = datetime_utils.parse(last_seen_value)
        else:
            start_from = last_seen_value

        records, next_page = client.scroll(
            collection_name=COLLECTION_NAME,
            limit=20,
            order_by=models.OrderBy(**{"key": key, "start_from": start_from}),
            scroll_filter=models.Filter(**{"must_not": [{"has_id": last_value_ids}]}),
            with_payload=True,
        )

        assert next_page is None

        if len(records) == 0:
            break

        last_value = records[-1].order_value
        if last_seen_value != last_value:
            last_seen_value = last_value
            last_value_ids = []

        for record in records:
            if isinstance(record.payload[key], list):
                order_value_payload = [to_order_value(item) for item in record.payload[key]]
                if last_seen_value in order_value_payload:
                    last_value_ids.append(record.id)
            else:
                order_value_payload = to_order_value(record.payload[key])
                if last_seen_value == order_value_payload:
                    last_value_ids.append(record.id)

        all_records.extend(records)

    # order_by does not guarantee secondary ordering by id,
    # so let's sort all the consecutive same-value records by ID to be able to compare results
    return subsorted_by_id(all_records, key)


def subsorted_by_id(all_records: list[models.Record], key: str) -> list[models.Record]:
    resorted_records = []
    same_value_batch = []
    same_value = None

    for record in all_records:
        if same_value is None:
            same_value = record.payload[key]
        if record.payload[key] != same_value:
            resorted_records.extend(sorted(same_value_batch, key=lambda r: r.id))
            same_value_batch = []
            same_value = record.payload[key]
        same_value_batch.append(record)

    resorted_records.extend(sorted(same_value_batch, key=lambda r: r.id))

    assert len(resorted_records) == len(all_records)

    return resorted_records


def scroll_all_integers(client: QdrantBase) -> list[models.Record]:
    return scroll_all_with_key(client, "rand_digit")


def scroll_all_floats(client: QdrantBase) -> list[models.Record]:
    return scroll_all_with_key(client, "rand_number")


def scroll_all_datetimes(client: QdrantBase) -> list[models.Record]:
    return scroll_all_with_key(client, "rand_datetime")


def scroll_all_integer_arrays(client: QdrantBase) -> list[models.Record]:
    return scroll_all_with_key(client, "integer_array")


def test_simple_scroll() -> None:
    fixture_points = generate_fixtures(200)

    local_client = init_local()
    init_client(local_client, fixture_points)

    http_client = init_remote()
    init_client(http_client, fixture_points)
    http_client.create_payload_index(
        COLLECTION_NAME, "rand_digit", models.PayloadSchemaType.INTEGER, wait=True
    )
    http_client.create_payload_index(
        COLLECTION_NAME, "rand_number", models.PayloadSchemaType.FLOAT, wait=True
    )
    http_client.create_payload_index(
        COLLECTION_NAME, "rand_datetime", models.PayloadSchemaType.DATETIME, wait=True
    )

    grpc_client = init_remote(prefer_grpc=True)

    # integers test the case of same-value records, since we generate only 10 different values
    compare_client_results(grpc_client, http_client, scroll_all_integers)
    compare_client_results(local_client, http_client, scroll_all_integers)

    compare_client_results(grpc_client, http_client, scroll_all_floats)
    compare_client_results(local_client, http_client, scroll_all_floats)

    compare_client_results(grpc_client, http_client, scroll_all_datetimes)
    compare_client_results(local_client, http_client, scroll_all_datetimes)


def test_scroll_duplicated_values():
    local_client = init_local()
    http_client = init_remote()
    grpc_client = init_remote(prefer_grpc=True)

    fixture_points = [
        models.PointStruct(id=1, vector=[], payload={"integer_array": [1, 2, 3, 4]}),
        models.PointStruct(id=2, vector=[], payload={"integer_array": [2, 3, 4, 5]}),
    ]
    init_client(http_client, fixture_points, vectors_config={})
    init_client(local_client, fixture_points, vectors_config={})

    http_client.create_payload_index(
        COLLECTION_NAME, "integer_array", models.PayloadSchemaType.INTEGER, wait=True
    )
    compare_client_results(grpc_client, http_client, scroll_all_integer_arrays)
    compare_client_results(local_client, http_client, scroll_all_integer_arrays)


def scroll_all_bools_and_ints(client: QdrantBase) -> list[models.Record]:
    return scroll_all_with_key(client, "bool_and_int")


def test_scroll_bool_values_are_skipped():
    local_client = init_local()
    http_client = init_remote()
    grpc_client = init_remote(prefer_grpc=True)

    fixture_points = [
        models.PointStruct(id=1, vector=[], payload={"bool_and_int": True}),
        models.PointStruct(id=2, vector=[], payload={"bool_and_int": 1}),
        models.PointStruct(id=3, vector=[], payload={"bool_and_int": False}),
        models.PointStruct(id=4, vector=[], payload={"bool_and_int": 0}),
        models.PointStruct(id=5, vector=[], payload={"bool_and_int": [True, 7]}),
    ]
    init_client(http_client, fixture_points, vectors_config={})
    init_client(local_client, fixture_points, vectors_config={})

    http_client.create_payload_index(
        COLLECTION_NAME, "bool_and_int", models.PayloadSchemaType.INTEGER, wait=True
    )

    # all points are stored, only their ordering values differ
    assert len(http_client.scroll(COLLECTION_NAME, limit=10)[0]) == len(fixture_points)
    assert len(local_client.scroll(COLLECTION_NAME, limit=10)[0]) == len(fixture_points)

    compare_client_results(grpc_client, http_client, scroll_all_bools_and_ints)
    compare_client_results(local_client, http_client, scroll_all_bools_and_ints)


def scroll_from_naive_datetime(client: QdrantBase) -> list[models.Record]:
    records, next_page = client.scroll(
        collection_name=COLLECTION_NAME,
        limit=24,
        # no offset, so this means noon UTC, both to the server and to local mode
        order_by=models.OrderBy(key="datetime", start_from=datetime(2024, 6, 15, 12, 0, 0)),
        with_payload=True,
    )

    assert next_page is None

    return subsorted_by_id(records, "datetime")


def test_scroll_from_naive_datetime() -> None:
    """A naive `start_from` means the same instant to local mode as it does to the server.

    Local mode used to read it as the client machine's local time, so a regression here is
    only visible on a client outside UTC.
    """
    fixture_points = [
        models.PointStruct(
            id=hour, vector=[], payload={"datetime": f"2024-06-15T{hour:02d}:00:00Z"}
        )
        for hour in range(24)
    ]

    local_client = init_local()
    init_client(local_client, fixture_points, vectors_config={})

    http_client = init_remote()
    init_client(http_client, fixture_points, vectors_config={})
    http_client.create_payload_index(
        COLLECTION_NAME, "datetime", models.PayloadSchemaType.DATETIME, wait=True
    )

    grpc_client = init_remote(prefer_grpc=True)

    compare_client_results(grpc_client, http_client, scroll_from_naive_datetime)
    compare_client_results(local_client, http_client, scroll_from_naive_datetime)


DATE_START_FROM = date(2024, 6, 15)

DATED_POINTS = [
    models.PointStruct(id=1, vector=[], payload={"date_field": "2024-06-13T12:00:00+00:00"}),
    models.PointStruct(id=2, vector=[], payload={"date_field": "2024-06-14T23:59:59+00:00"}),
    # exactly on the boundary: start_from is inclusive
    models.PointStruct(id=3, vector=[], payload={"date_field": "2024-06-15T00:00:00+00:00"}),
    models.PointStruct(id=4, vector=[], payload={"date_field": "2024-06-15T08:30:00+00:00"}),
    models.PointStruct(id=5, vector=[], payload={"date_field": "2024-06-17T00:00:00+00:00"}),
]


def scroll_from_date(client: QdrantBase, direction: models.Direction) -> list[models.Record]:
    records, next_page = client.scroll(
        collection_name=COLLECTION_NAME,
        limit=10,
        order_by=models.OrderBy(key="date_field", direction=direction, start_from=DATE_START_FROM),
        with_payload=True,
    )

    assert next_page is None

    return records


def test_scroll_order_by_date_start_from() -> None:
    """A `date` start_from means midnight UTC on that day to local, REST and gRPC alike.

    Local mode used to drop it and scroll the whole collection, and gRPC used to raise.
    """
    local_client = init_local()
    init_client(local_client, DATED_POINTS, vectors_config={})

    http_client = init_remote()
    init_client(http_client, DATED_POINTS, vectors_config={})
    http_client.create_payload_index(
        COLLECTION_NAME, "date_field", models.PayloadSchemaType.DATETIME, wait=True
    )

    grpc_client = init_remote(prefer_grpc=True)

    for direction, expected_ids in [
        (models.Direction.ASC, [3, 4, 5]),
        (models.Direction.DESC, [3, 2, 1]),
    ]:
        # a dropped start_from returns all five points, which congruence alone would miss
        assert [record.id for record in scroll_from_date(http_client, direction)] == expected_ids

        compare_client_results(grpc_client, http_client, scroll_from_date, direction=direction)
        compare_client_results(local_client, http_client, scroll_from_date, direction=direction)
