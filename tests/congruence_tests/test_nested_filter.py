import json

from qdrant_client.client_base import QdrantBase
from qdrant_client.http.models import models
from tests.congruence_tests.test_common import (
    COLLECTION_NAME,
    compare_client_results,
    generate_fixtures,
    init_client,
    init_local,
    init_remote,
)


def test_nested_query():
    fixture_points = generate_fixtures(num=20)

    local_client = init_local()
    init_client(local_client, fixture_points)

    remote_client = init_remote()
    init_client(remote_client, fixture_points)

    filter_ = models.Filter(
        **{
            "must": [
                {
                    "nested": {
                        "key": "nested.array",
                        "filter": {
                            "must": [
                                {
                                    "key": "word",
                                    "match": {"value": "cat"},
                                }
                            ],
                            "must_not": [
                                {
                                    "key": "number",
                                    "range": {
                                        "lt": 3.0,
                                    },
                                }
                            ],
                        },
                    }
                }
            ]
        }
    )

    local_result, _next_page = local_client.scroll(
        collection_name=COLLECTION_NAME,
        scroll_filter=filter_,
        limit=100,
        with_payload=True,
    )

    remote_result, _next_page = remote_client.scroll(
        collection_name=COLLECTION_NAME,
        scroll_filter=filter_,
        limit=100,
        with_payload=True,
    )

    # assert len(local_result) == len(remote_result)

    for local, remote in zip(local_result, remote_result):
        if local.id != remote.id:
            print(f"Local: {local.id}, Remote: {remote.id}")

            print(f"Local:", json.dumps(local.payload["nested"]["array"], indent=2))
            print(f"Remote:", json.dumps(remote.payload["nested"]["array"], indent=2))

            assert False


def scroll_with_filter(client: QdrantBase, scroll_filter: models.Filter):
    return client.scroll(
        collection_name=COLLECTION_NAME,
        scroll_filter=scroll_filter,
        limit=100,
        with_payload=True,
    )


def test_match_text_on_non_string_field():
    """MatchText and MatchTextAny must not crash when applied to a non-string
    payload field. The "rand_digit" field is an int, so they should simply not
    match instead of raising in local mode.
    """
    fixture_points = generate_fixtures()

    local_client = init_local()
    init_client(local_client, fixture_points)

    remote_client = init_remote()
    init_client(remote_client, fixture_points)

    match_text_filter = models.Filter(
        must=[models.FieldCondition(key="rand_digit", match=models.MatchText(text="5"))]
    )
    compare_client_results(
        local_client,
        remote_client,
        scroll_with_filter,
        scroll_filter=match_text_filter,
    )

    match_text_any_filter = models.Filter(
        must=[models.FieldCondition(key="rand_digit", match=models.MatchTextAny(text_any="5"))]
    )
    compare_client_results(
        local_client,
        remote_client,
        scroll_with_filter,
        scroll_filter=match_text_any_filter,
    )
