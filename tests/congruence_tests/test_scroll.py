import random

import pytest

from qdrant_client import QdrantClient
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
    def scroll_all(cls, client: QdrantBase) -> list[models.Record]:
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
    fixture_points = generate_fixtures(200)

    scroller = TestSimpleScroller()

    local_client = init_local()
    init_client(local_client, fixture_points)

    remote_client = init_remote()
    init_client(remote_client, fixture_points)

    compare_client_results(local_client, remote_client, scroller.scroll_all)


def test_simple_sparse_scroll() -> None:
    fixture_points = generate_sparse_fixtures(200)

    local_client = init_local()
    init_client(local_client, fixture_points, sparse_vectors_config=sparse_vectors_config)

    remote_client = init_remote()
    init_client(remote_client, fixture_points, sparse_vectors_config=sparse_vectors_config)

    scroller = TestSimpleScroller()

    compare_client_results(local_client, remote_client, scroller.scroll_all)


def test_mixed_ids() -> None:
    fixture_points = generate_fixtures(100, random_ids=True) + generate_fixtures(
        100, random_ids=False
    )

    random.shuffle(fixture_points)

    scroller = TestSimpleScroller()

    local_client = init_local()
    init_client(local_client, fixture_points)

    remote_client = init_remote()
    init_client(remote_client, fixture_points)

    compare_client_results(local_client, remote_client, scroller.scroll_all)


def test_sparse_mixed_ids() -> None:
    fixture_points = generate_sparse_fixtures(100, random_ids=True) + generate_sparse_fixtures(
        100, random_ids=False
    )

    random.shuffle(fixture_points)

    scroller = TestSimpleScroller()

    local_client = init_local()
    init_client(local_client, fixture_points, sparse_vectors_config=sparse_vectors_config)

    remote_client = init_remote()
    init_client(remote_client, fixture_points, sparse_vectors_config=sparse_vectors_config)

    compare_client_results(local_client, remote_client, scroller.scroll_all)


def _init_value_type_collection(client: QdrantBase) -> None:
    if client.collection_exists(COLLECTION_NAME):
        client.delete_collection(COLLECTION_NAME)
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=models.VectorParams(size=2, distance=models.Distance.COSINE),
    )
    client.upsert(
        collection_name=COLLECTION_NAME,
        points=[
            models.PointStruct(id=1, vector=[0.1, 0.2], payload={"n": 1}),
            models.PointStruct(id=2, vector=[0.1, 0.2], payload={"n": True}),
            models.PointStruct(id=3, vector=[0.1, 0.2], payload={"n": 0}),
            models.PointStruct(id=4, vector=[0.1, 0.2], payload={"n": False}),
            models.PointStruct(id=5, vector=[0.1, 0.2], payload={"n": 1.0}),
            models.PointStruct(id=6, vector=[0.1, 0.2], payload={"n": 0.0}),
        ],
        wait=True,
    )


def _scroll_value_type(client: QdrantBase, scroll_filter: models.Filter) -> list[models.Record]:
    records, _next_page = client.scroll(
        collection_name=COLLECTION_NAME,
        scroll_filter=scroll_filter,
        limit=10,
        with_payload=True,
    )
    return records


@pytest.mark.parametrize("prefer_grpc", [True, False])
def test_bool_int_float_filters_do_not_cross_match(prefer_grpc) -> None:
    """Python cross-matches bool / int / float (bool is a subclass of int, and
    1 == 1.0), but Qdrant keeps them as distinct payload value types for exact
    match. Local mode must not cross-match them, matching the remote server
    behaviour. Match condition operands can only be bool / int / str (never float),
    so a float payload value is never matched by an exact-match condition.
    """
    local_client: QdrantClient = init_local()
    _init_value_type_collection(local_client)

    remote_client: QdrantClient = init_remote(prefer_grpc=prefer_grpc)
    _init_value_type_collection(remote_client)

    filters = [
        models.Filter(must=[models.FieldCondition(key="n", match=models.MatchValue(value=1))]),
        models.Filter(must=[models.FieldCondition(key="n", match=models.MatchValue(value=True))]),
        models.Filter(must=[models.FieldCondition(key="n", match=models.MatchValue(value=0))]),
        models.Filter(must=[models.FieldCondition(key="n", match=models.MatchValue(value=False))]),
        models.Filter(must=[models.FieldCondition(key="n", match=models.MatchAny(any=[1, 0]))]),
        models.Filter(
            must=[models.FieldCondition(key="n", match=models.MatchExcept(**{"except": [1, 0]}))]
        ),
        models.Filter(must=[models.FieldCondition(key="n", range=models.Range(gte=1, lte=1))]),
        models.Filter(must=[models.FieldCondition(key="n", range=models.Range(gte=0, lte=1))]),
    ]

    for scroll_filter in filters:
        compare_client_results(
            local_client,
            remote_client,
            _scroll_value_type,
            scroll_filter=scroll_filter,
        )


def _init_null_payload_collection(client: QdrantBase) -> None:
    if client.collection_exists(COLLECTION_NAME):
        client.delete_collection(COLLECTION_NAME)
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=models.VectorParams(size=2, distance=models.Distance.COSINE),
    )
    client.upsert(
        collection_name=COLLECTION_NAME,
        points=[
            # A real value that the `except` list excludes, and one it keeps.
            models.PointStruct(id=1, vector=[0.1, 0.2], payload={"n": 1}),
            models.PointStruct(id=2, vector=[0.1, 0.2], payload={"n": 5}),
            # Explicit nulls, in each shape a payload can carry them.
            models.PointStruct(id=3, vector=[0.1, 0.2], payload={"n": None}),
            models.PointStruct(id=4, vector=[0.1, 0.2], payload={"s": None}),
            models.PointStruct(id=5, vector=[0.1, 0.2], payload={"arr": None}),
            # An array holding a null beside an excluded value.
            models.PointStruct(id=6, vector=[0.1, 0.2], payload={"n": [1, None]}),
            # The key absent altogether.
            models.PointStruct(id=7, vector=[0.1, 0.2], payload={"other": 1}),
        ],
        wait=True,
    )


@pytest.mark.parametrize("prefer_grpc", [True, False])
def test_match_except_does_not_match_null_payload_values(prefer_grpc) -> None:
    """A null payload value is absence, not a value that differs from everything.

    `MatchExcept` is the only negated match condition, so it is the only one where
    treating null as an ordinary value flips the result: `not any(...)` over a null
    returns True and the point matches, while the server drops nulls before matching
    and does not match it. A field whose only value is null must not satisfy an
    "except" condition, and neither must `[1, None]` when 1 is excluded.
    """
    local_client: QdrantClient = init_local()
    _init_null_payload_collection(local_client)

    remote_client: QdrantClient = init_remote(prefer_grpc=prefer_grpc)
    _init_null_payload_collection(remote_client)

    filters = [
        models.Filter(
            must=[models.FieldCondition(key="n", match=models.MatchExcept(**{"except": [1]}))]
        ),
        models.Filter(
            must=[models.FieldCondition(key="n", match=models.MatchExcept(**{"except": []}))]
        ),
        models.Filter(
            must=[models.FieldCondition(key="s", match=models.MatchExcept(**{"except": ["x"]}))]
        ),
        models.Filter(
            must=[models.FieldCondition(key="arr", match=models.MatchExcept(**{"except": [1]}))]
        ),
    ]

    for scroll_filter in filters:
        compare_client_results(
            local_client,
            remote_client,
            _scroll_value_type,
            scroll_filter=scroll_filter,
        )


@pytest.mark.parametrize("prefer_grpc", [True, False])
def test_empty_should_is_not_a_constraint(prefer_grpc) -> None:
    """An empty `should` states no alternatives, not an unsatisfiable one.

    The server treats `should: []` as no constraint and returns every point.
    `any([])` is False, so a literal reading returns no points instead — the exact
    inverse. `must: []` and `must_not: []` already agree on the vacuous reading;
    this pins `should` to the same rule.
    """
    local_client: QdrantClient = init_local()
    _init_null_payload_collection(local_client)

    remote_client: QdrantClient = init_remote(prefer_grpc=prefer_grpc)
    _init_null_payload_collection(remote_client)

    filters = [
        models.Filter(should=[]),
        models.Filter(must=[], must_not=[], should=[]),
        # A non-empty `should` must still constrain, so the fix cannot be "ignore should".
        models.Filter(should=[models.FieldCondition(key="n", match=models.MatchValue(value=1))]),
    ]

    for scroll_filter in filters:
        compare_client_results(
            local_client,
            remote_client,
            _scroll_value_type,
            scroll_filter=scroll_filter,
        )
