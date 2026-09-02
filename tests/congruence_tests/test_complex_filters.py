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


@pytest.mark.parametrize("num_points", [10, 0], ids=["populated", "empty"])
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
def test_invalid_slice_is_rejected(total: int, index: int, reported: str, num_points: int):
    """`index` must be within `0..total`, and `total` at least 1, on both clients.

    Empty collections included: the values are invalid on their own, so validating them
    while scanning points means never validating them when there are none.
    """
    fixture_points = generate_fixtures(num=num_points) if num_points else []

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


def min_should_filter(min_count: int, conditions: list | None = None) -> models.Filter:
    if conditions is None:
        conditions = [models.FieldCondition(key="rand_digit", match=models.MatchValue(value=1))]
    return models.Filter(min_should=models.MinShould(conditions=conditions, min_count=min_count))


def nested_condition(inner: models.Filter) -> models.NestedCondition:
    return models.NestedCondition(nested=models.Nested(key="nested.array", filter=inner))


@pytest.mark.parametrize(
    "scroll_filter",
    [
        min_should_filter(0),
        min_should_filter(-1),
        models.Filter(must=[min_should_filter(0)]),
        models.Filter(must_not=[min_should_filter(0)]),
        min_should_filter(1, conditions=[min_should_filter(0)]),
        models.Filter(must=[nested_condition(min_should_filter(0))]),
        models.Filter(should=[models.Filter(must=[min_should_filter(0)])]),
    ],
    ids=[
        "zero",
        "negative",
        "in_must",
        "in_must_not",
        "in_min_should_conditions",
        "through_nested_condition",
        "two_levels_deep",
    ],
)
def test_invalid_min_count_is_rejected(scroll_filter: models.Filter):
    """`min_count` must be at least 1, at any depth, on both clients.

    Local mode evaluates `min_should` as `matches >= min_count`, so anything at or below
    zero is trivially true for every point and the whole collection comes back instead of
    being rejected - the worst direction for a filter to be wrong in.
    """
    fixture_points = generate_fixtures(num=10)

    local_client = init_local()
    init_client(local_client, fixture_points)

    remote_client = init_remote()
    init_client(remote_client, fixture_points)

    # local mode names the offending parameter, the server reports its own validation error;
    # the point is that neither silently accepts it
    with pytest.raises(ValueError, match="min_count"):
        local_client.scroll(COLLECTION_NAME, scroll_filter=scroll_filter, limit=10)

    with pytest.raises(UnexpectedResponse):
        remote_client.scroll(COLLECTION_NAME, scroll_filter=scroll_filter, limit=10)


@pytest.mark.parametrize("num_points", [10, 0], ids=["populated", "empty"])
def test_invalid_min_count_is_rejected_on_every_filter_path(num_points: int):
    """Reads and `update_filter` writes alike, and on an empty collection too.

    The `update_filter` paths check points directly instead of building a payload mask, and
    `scroll` returns early when the collection is empty, so each needs the filter validated
    where it enters the collection rather than deep in the scan.
    """
    fixture_points = generate_fixtures(num=10)
    point = fixture_points[0]

    local_client = init_local()
    init_client(local_client, fixture_points[:num_points])

    remote_client = init_remote()
    init_client(remote_client, fixture_points[:num_points])

    flt = min_should_filter(0)
    # `delete` and `clear_payload` are left out on purpose: the server does not validate the
    # filter on those two requests and happily wipes everything, so local mode rejecting them
    # is a deliberate divergence rather than a congruence failure.
    operations = [
        lambda client: client.scroll(COLLECTION_NAME, scroll_filter=flt, limit=10),
        lambda client: client.count(COLLECTION_NAME, count_filter=flt),
        lambda client: client.upsert(
            COLLECTION_NAME, points=[point], update_filter=flt, wait=True
        ),
        lambda client: client.update_vectors(
            COLLECTION_NAME,
            points=[models.PointVectors(id=point.id, vector=point.vector)],
            update_filter=flt,
            wait=True,
        ),
    ]

    for operation in operations:
        with pytest.raises(ValueError, match="min_count"):
            operation(local_client)

        with pytest.raises(UnexpectedResponse):
            operation(remote_client)


@pytest.mark.parametrize(
    "except_",
    [[1], [], ["x"]],
    ids=["excluded_value", "nothing_excluded", "other_type"],
)
def test_match_except_does_not_match_null(except_: list):
    """A null payload value is absence, not a value which happens to differ from every entry
    in the `except` list. The server drops nulls before matching, so a field whose only value
    is null never satisfies an `except` condition - and neither does `[1, None]` when 1 is
    excluded.
    """
    values = [
        1,  # a value the `except` list excludes
        5,  # a value it keeps
        "x",  # a value of another type
        None,  # an explicit null
        [1, None],  # an array holding a null beside an excluded value
        [None],  # an array holding nothing but a null
    ]

    fixture_points = generate_fixtures(num=len(values) + 1)
    for point, value in zip(fixture_points, values):
        point.payload = {"n": value}
    fixture_points[-1].payload = {"other": 1}  # the key absent altogether

    local_client = init_local()
    init_client(local_client, fixture_points)

    remote_client = init_remote()
    init_client(remote_client, fixture_points)

    compare_client_results(
        local_client,
        remote_client,
        scroll_with_filter,
        scroll_filter=models.Filter(
            must=[models.FieldCondition(key="n", match=models.MatchExcept(**{"except": except_}))]
        ),
    )


@pytest.mark.parametrize(
    "filter_",
    [
        models.Filter(should=[]),
        models.Filter(must=[], must_not=[], should=[]),
    ],
    ids=["should", "all_clauses"],
)
def test_empty_should_is_not_a_constraint(filter_: models.Filter):
    """An empty `should` states no alternatives to satisfy, not an unsatisfiable one. The
    server reads it as no constraint and returns every point, while `any([])` is False, so a
    literal reading returns none - the exact inverse. `must` and `must_not` are `all(...)`,
    vacuously true when empty, and already agree.
    """
    fixture_points = generate_fixtures(num=10)

    local_client = init_local()
    init_client(local_client, fixture_points)

    remote_client = init_remote()
    init_client(remote_client, fixture_points)

    compare_client_results(
        local_client,
        remote_client,
        scroll_with_filter,
        scroll_filter=filter_,
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


@pytest.mark.parametrize(
    "match",
    [
        models.MatchText(text="fly"),
        models.MatchText(text="FLY"),
        models.MatchText(text="good cheap"),
        models.MatchPhrase(phrase="alpha beta"),
        models.MatchPhrase(phrase="Alpha, Beta!"),
        models.MatchPhrase(phrase="good"),
        models.MatchTextAny(text_any="good fly"),
    ],
    ids=[
        "text_word",
        "text_uppercase",
        "text_two_words",
        "phrase_two_words",
        "phrase_punctuated",
        "phrase_one_word",
        "text_any",
    ],
)
def test_text_match_on_unindexed_field(match: models.Match):
    """On a field without a text index the server tokenizes both sides with the default word
    tokenizer - split on non-alphanumeric, lowercased - and matches whole tokens rather than
    substrings, so "fly" does not match "butterfly". `MatchText` accepts the query tokens in
    any order, `MatchPhrase` only consecutively, and `MatchTextAny` is the exception which
    still scans for substrings.
    """
    values = [
        "goodness only",  # substring of the query, not a token
        "good cheap stuff",
        "cheap hardware good",  # query tokens present, reversed
        "cheap hardware",  # only one of two query tokens
        "fly agaric",
        "come fly, with me",  # token followed by punctuation
        "butterfly dragonfly",  # substrings only
        "foo alpha beta bar",
        "beta alpha",
        "alpha x beta",  # in order but not consecutive
        "alphabeta",  # a single token, not two
        "goodness only good",
        7,  # a non-string value has no tokens
    ]

    fixture_points = generate_fixtures(num=len(values) + 1)
    for point, value in zip(fixture_points, values):
        point.payload = {"words": value}
    fixture_points[-1].payload = {"other": 1}  # the key absent altogether

    local_client = init_local()
    init_client(local_client, fixture_points)

    remote_client = init_remote()
    init_client(remote_client, fixture_points)

    compare_client_results(
        local_client,
        remote_client,
        scroll_with_filter,
        scroll_filter=models.Filter(must=[models.FieldCondition(key="words", match=match)]),
    )


@pytest.mark.parametrize("key", ["a", "nested[].empty"])
def test_is_null_matches_null_inside_arrays(key: str):
    """A value counts as null when it is null itself or is an array holding a null element,
    one level deep - so `[null, 1]` matches while `[[null]]` does not.
    """
    payloads = [
        {"a": [None, 1]},
        {"a": [1, None]},
        {"a": [1, 2]},
        {"a": None},
        {"a": [[None]]},  # the null is one level too deep
        {"a": []},
        {"nested": [{"empty": [None]}, {"empty": [None]}]},
        {"nested": [{"empty": 1}]},
        {"other": 1},  # the key absent altogether
    ]

    fixture_points = generate_fixtures(num=len(payloads))
    for point, payload in zip(fixture_points, payloads):
        point.payload = payload

    local_client = init_local()
    init_client(local_client, fixture_points)

    remote_client = init_remote()
    init_client(remote_client, fixture_points)

    compare_client_results(
        local_client,
        remote_client,
        scroll_with_filter,
        scroll_filter=models.Filter(
            must=[models.IsNullCondition(is_null=models.PayloadField(key=key))]
        ),
    )
