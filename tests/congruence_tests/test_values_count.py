import pytest

from qdrant_client.http.models import models
from tests.congruence_tests.test_common import (
    COLLECTION_NAME,
    generate_fixtures,
    init_client,
    init_local,
    init_remote,
)

PAYLOAD_KEY = "city"


@pytest.mark.parametrize(
    "payloads,filter_params",
    [
        ([{"city": "Berlin"}], {"lte": 1}),  # case 1
        # ([{}], {"lte": 1}),  # case 2, fails
        ([{"city": None}], {"lte": 1}),  # case 3
        ([{"city": []}], {"lte": 1}),  # case 4
        ([{"city": ["Berlin", "London"]}], {"lte": 1}),  # case 5
        # ([None], {"lte": 1}),  # case 6 fails
        ([{"city": ""}], {"lte": 1}),  # case 7
        ([{"country": "Germany"}], {"lte": 1}),  # case 8, fails - similar to case 2
    ],
)
def test_values_count_query(payloads, filter_params):
    fixture_records = generate_fixtures(num=len(payloads))
    for i, record in enumerate(fixture_records):
        record.payload = payloads[i]

    local_client = init_local()
    init_client(local_client, fixture_records)

    remote_client = init_remote()
    init_client(remote_client, fixture_records)

    filter_ = models.Filter(
        must=[
            models.FieldCondition(
                key=PAYLOAD_KEY, values_count=models.ValuesCount(**filter_params)
            )
        ]
    )

    local_result, _next_page = local_client.scroll(
        collection_name=COLLECTION_NAME,
        scroll_filter=filter_,
        limit=10,
        with_payload=True,
    )

    remote_result, _next_page = remote_client.scroll(
        collection_name=COLLECTION_NAME,
        scroll_filter=filter_,
        limit=10,
        with_payload=True,
    )

    assert [record.payload for record in local_result] == [
        record.payload for record in remote_result
    ]
