from typing import Any, Dict, List

import pytest

from qdrant_client.http.models import models
from tests.congruence_tests.test_common import (
    COLLECTION_NAME,
    generate_fixtures,
    init_client,
    init_local,
    init_remote,
)


@pytest.mark.parametrize(
    "payloads,filter_params,payload_key",
    [
        # key is absent
        ([{}], {"lte": 1}, "city"),
        ([{}], {"lt": 1}, "city"),
        ([None], {"lte": 1}, "city"),
        # key is present, but value is None
        ([{"city": None}], {"lt": 1}, "city"),
        ([{"city": None}], {"gte": 1}, "city"),
        # primitives
        ([{"nested": [{"empty": 1}, {"empty": 2}, {"empty": 3}]}], {"lte": 1}, "nested[].empty"),
        # mixed types
        (
            [{"nested": [{"empty": 1}, {"empty": [1]}, {"empty": "z"}, {"empty": None}]}],
            {"lte": 1},
            "nested[].empty",
        ),
        # Several None
        (
            [{"nested": [{"empty": None}, {"empty": None}, {"empty": None}]}],
            {"lt": 1},
            "nested[].empty",
        ),
        # Single None
        (
            [
                {
                    "nested": [
                        {"empty": None},
                    ]
                }
            ],
            {"lt": 1},
            "nested[].empty",
        ),
        # Various length
        (
            [{"nested": [{"empty": [1]}, {"empty": [1, 2]}, {"empty": [1, 2, 3]}]}],
            {"lte": 1},
            "nested[].empty",
        ),
        # Lists and None
        (
            [{"nested": [{"empty": []}, {"empty": []}, {"empty": None}]}],
            {"gte": 1},
            "nested[].empty",
        ),  # local
        # Nested item with None
        ([{"nested": [{"empty": [{"inner": None}]}]}], {"lt": 1}, "nested[].empty"),  # local
        # Check that local mode does not sum up values
        (
            [
                {
                    "nested": [
                        {"empty": [None, None]},
                        {"empty": [None, None]},
                        {"empty": [None, None]},
                    ]
                }
            ],
            {"lt": 3},
            "nested[].empty",
        ),
        (
            [{"nested": [{"empty": [None]}, {"empty": [None]}, {"empty": [None]}]}],
            {"lt": 2},
            "nested[].empty",
        ),
        (
            [{"nested": [{"empty": [234, 33, 22]}, {"empty": [11, 22]}]}],
            {"lt": 4},
            "nested[].empty",
        ),
    ],
)
def test_values_count_query(payloads, filter_params, payload_key):
    fixture_points = generate_fixtures(num=len(payloads))
    for i, point in enumerate(fixture_points):
        point.payload = payloads[i]

    local_client = init_local()
    init_client(local_client, fixture_points)

    remote_client = init_remote()
    init_client(remote_client, fixture_points)

    filter_ = models.Filter(
        must=[
            models.FieldCondition(
                key=payload_key, values_count=models.ValuesCount(**filter_params)
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


@pytest.mark.parametrize(
    "payloads,payload_key",
    [
        ([{"nested": [{"empty": None}, {"empty": None}]}], "nested[].empty"),
        (
            [
                {
                    "exist": 123,
                }
            ],
            "not_exist",
        ),
        # List of None
        (
            [{"nested": [{"empty": [None]}, {"empty": [None]}, {"empty": [None]}]}],
            "nested[].empty",
        ),
        # List of Nones
        ([{"nones": [None, None]}], "nones"),
        # Empty list
        ([{"nested": [{"empty": []}]}], "nested[].empty"),
        # Empty lists and Nones
        ([{"nested": [{"empty": []}, {"empty": None}]}], "nested[].empty"),
        ([{}], "city"),
    ],
)
def test_is_empty(payloads: List[Dict[str, Any]], payload_key: str) -> None:
    fixture_points = generate_fixtures(num=len(payloads))
    for i, point in enumerate(fixture_points):
        point.payload = payloads[i]

    local_client = init_local()
    init_client(local_client, fixture_points)

    remote_client = init_remote()
    init_client(remote_client, fixture_points)

    filter_ = models.Filter(
        must=[models.IsEmptyCondition(is_empty=models.PayloadField(key=payload_key))]
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


@pytest.mark.parametrize(
    "payloads,payload_key",
    [
        # List of None
        (
            [{"nested": [{"empty": [None]}, {"empty": [None]}, {"empty": [None]}]}],
            "nested[].empty",
        ),
        # List of Nones
        ([{"nones": [None, None]}], "nones"),
        (
            [
                {
                    "exist": 123,
                }
            ],
            "not_exist",
        ),
        ([{}], "city"),
    ],
)
def test_is_null(payloads, payload_key):
    fixture_points = generate_fixtures(num=len(payloads))
    for i, point in enumerate(fixture_points):
        point.payload = payloads[i]

    local_client = init_local()
    init_client(local_client, fixture_points)

    remote_client = init_remote()
    init_client(remote_client, fixture_points)

    filter_ = models.Filter(
        must=[models.IsNullCondition(is_null=models.PayloadField(key=payload_key))]
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
