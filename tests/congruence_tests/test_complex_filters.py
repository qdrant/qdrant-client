import json

import pytest

from qdrant_client.client_base import QdrantBase
from qdrant_client.http.exceptions import UnexpectedResponse
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


def slice_filter(total: int, index: int) -> models.Filter:
    return models.Filter(
        must=[models.SliceCondition(slice=models.Slice(total=total, index=index))]
    )


@pytest.mark.parametrize("random_ids", [False, True], ids=["int_ids", "uuid_ids"])
def test_slices_cover_all_points(random_ids: bool):
    """Slices of the id space must be disjoint and together cover every point.

    Membership comes from a hash of the raw id bytes, so local mode has to reproduce the
    engine's hash exactly - comparing the slices themselves, not just their sizes.
    """
    fixture_points = generate_fixtures(num=200, random_ids=random_ids)

    local_client = init_local()
    init_client(local_client, fixture_points)

    remote_client = init_remote()
    init_client(remote_client, fixture_points)

    all_ids = {point.id for point in fixture_points}

    for total in (1, 2, 3, 4, 7, 16, 64):
        slices = {}
        for client in (local_client, remote_client):
            seen: set = set()
            for index in range(total):
                ids = {
                    point.id
                    for point in client.scroll(
                        COLLECTION_NAME,
                        scroll_filter=slice_filter(total, index),
                        limit=len(all_ids),
                    )[0]
                }
                assert not (ids & seen), f"slices overlap for total={total}"
                seen |= ids
                slices.setdefault(index, []).append(ids)

            assert seen == all_ids, f"slices did not cover every point for total={total}"

        for index, (local_ids, remote_ids) in slices.items():
            assert local_ids == remote_ids, f"slice {index} of {total} differs between clients"


@pytest.mark.parametrize(
    "total,index,reported",
    [
        (0, 0, "total"),
        (-1, 0, "total"),
        (1, 1, "index"),
        (4, 4, "index"),
        (4, 9, "index"),
        (2, -1, "index"),
    ],
)
def test_invalid_slice_is_rejected(total: int, index: int, reported: str):
    """`index` must be within `0..total`, and `total` at least 1, on both clients."""
    fixture_points = generate_fixtures(num=10)

    local_client = init_local()
    init_client(local_client, fixture_points)

    remote_client = init_remote()
    init_client(remote_client, fixture_points)

    # local mode names the offending parameter, the server reports its own validation error;
    # the point is that neither silently accepts it
    with pytest.raises(ValueError, match=reported):
        local_client.scroll(COLLECTION_NAME, scroll_filter=slice_filter(total, index), limit=10)

    with pytest.raises(UnexpectedResponse):
        remote_client.scroll(COLLECTION_NAME, scroll_filter=slice_filter(total, index), limit=10)


def test_slice_condition_inside_nested_filter():
    """Local mode evaluates a nested filter against a sentinel point id, which must not reach
    the id hash as if it were a real id.

    `total=1, index=0` matches every point at the top level, so if the sentinel were treated as
    a real id the two clients would disagree.
    """
    fixture_points = generate_fixtures(num=50)

    local_client = init_local()
    init_client(local_client, fixture_points)

    remote_client = init_remote()
    init_client(remote_client, fixture_points)

    nested = models.NestedCondition(
        nested=models.Nested(
            key="nested.array",
            filter=models.Filter(
                must=[models.SliceCondition(slice=models.Slice(total=1, index=0))]
            ),
        )
    )

    compare_client_results(
        local_client,
        remote_client,
        lambda client: client.scroll(
            COLLECTION_NAME, scroll_filter=models.Filter(must=[nested]), limit=100
        )[0],
    )


@pytest.mark.parametrize("key", ["company", "company[]", "company[0]", "company[1]"])
def test_nested_filter_payload_shapes(key: str):
    """A nested filter is applied to the elements of an array of objects, so a value which is not
    an array has no elements to match against. `company` and `company[]` are equivalent.
    """
    shapes = [
        {"name": "qdrant"},  # plain object, not an array
        [{"name": "qdrant"}],  # array of objects
        {"a": {"name": "qdrant"}},  # map of objects, still not an array
        [{"name": "other"}, {"name": "qdrant"}],  # array where only the second element matches
        ["qdrant", 42, True, None],  # array of scalars, nothing to resolve `name` against
        [[{"name": "qdrant"}]],  # array whose single element is itself an array of objects
    ]

    fixture_points = generate_fixtures(num=len(shapes))
    for point, shape in zip(fixture_points, shapes):
        point.payload = {"company": shape}

    local_client = init_local()
    init_client(local_client, fixture_points)

    remote_client = init_remote()
    init_client(remote_client, fixture_points)

    nested_filter = models.Filter(
        must=[
            models.NestedCondition(
                nested=models.Nested(
                    key=key,
                    filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="name", match=models.MatchValue(value="qdrant")
                            )
                        ]
                    ),
                )
            )
        ]
    )

    compare_client_results(
        local_client,
        remote_client,
        scroll_with_filter,
        scroll_filter=nested_filter,
    )
