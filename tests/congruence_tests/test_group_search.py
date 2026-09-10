from typing import Sequence

import numpy as np

from qdrant_client.client_base import QdrantBase
from qdrant_client.conversions import common_types as types
from qdrant_client.http.models import models
from tests.congruence_tests.test_common import (
    COLLECTION_NAME,
    code_vector_size,
    compare_client_results,
    delete_fixture_collection,
    generate_fixtures,
    image_vector_size,
    init_client,
    init_local,
    init_remote,
    text_vector_size,
)
from tests.fixtures.filters import one_random_filter_please

LOOKUP_COLLECTION_NAME = "lookup_collection"


class TestGroupSearcher:
    __test__ = False

    def __init__(self):
        self.query_text = np.random.random(text_vector_size).tolist()
        self.query_image = np.random.random(image_vector_size).tolist()
        self.query_code = np.random.random(code_vector_size).tolist()
        self.group_by = "rand_digit"
        self.group_size = 1
        self.limit = 10

    def group_search(
        self,
        client: QdrantBase,
        query_vector: types.NumpyArray | Sequence[float] | tuple[str, list[float]],
    ) -> models.GroupsResult:
        using = None
        if isinstance(query_vector, tuple):
            using, query_vector = query_vector
        return client.query_points_groups(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            using=using,
            with_payload=models.PayloadSelectorExclude(exclude=["city.geo", "rand_number"]),
            group_by=self.group_by,
            limit=self.limit,
            group_size=self.group_size,
        )

    def group_search_text(self, client: QdrantBase) -> models.GroupsResult:
        return client.query_points_groups(
            collection_name=COLLECTION_NAME,
            using="text",
            query=self.query_text,
            with_payload=models.PayloadSelectorExclude(exclude=["city.geo", "rand_number"]),
            group_by=self.group_by,
            limit=self.limit,
            group_size=self.group_size,
        )

    def group_search_text_single(self, client: QdrantBase) -> models.GroupsResult:
        return client.query_points_groups(
            collection_name=COLLECTION_NAME,
            query=self.query_text,
            with_payload=models.PayloadSelectorExclude(exclude=["city.geo", "rand_number"]),
            group_by=self.group_by,
            limit=self.limit,
            group_size=self.group_size,
        )

    def group_search_image(self, client: QdrantBase) -> models.GroupsResult:
        return client.query_points_groups(
            collection_name=COLLECTION_NAME,
            using="image",
            query=self.query_image,
            with_payload=models.PayloadSelectorExclude(exclude=["city.geo", "rand_number"]),
            group_by=self.group_by,
            limit=self.limit,
            group_size=self.group_size,
        )

    def group_search_image_with_lookup(self, client: QdrantBase) -> models.GroupsResult:
        return client.query_points_groups(
            collection_name=COLLECTION_NAME,
            query=self.query_image,
            using="image",
            with_payload=models.PayloadSelectorExclude(exclude=["city.geo", "rand_number"]),
            group_by=self.group_by,
            limit=self.limit,
            group_size=self.group_size,
            with_lookup=LOOKUP_COLLECTION_NAME,
        )

    def group_search_image_with_lookup_2(self, client: QdrantBase) -> models.GroupsResult:
        return client.query_points_groups(
            collection_name=COLLECTION_NAME,
            using="image",
            query=self.query_image,
            with_payload=models.PayloadSelectorExclude(exclude=["city.geo", "rand_number"]),
            group_by=self.group_by,
            limit=self.limit,
            group_size=self.group_size,
            with_lookup=models.WithLookup(
                collection=LOOKUP_COLLECTION_NAME,
                with_payload=models.PayloadSelectorExclude(exclude=["city.geo", "rand_number"]),
                with_vectors=["image"],
            ),
        )

    def group_search_code(self, client: QdrantBase) -> models.GroupsResult:
        return client.query_points_groups(
            collection_name=COLLECTION_NAME,
            using="code",
            query=self.query_code,
            with_payload=models.PayloadSelectorExclude(exclude=["city.geo", "rand_number"]),
            group_by=self.group_by,
            limit=self.limit,
            group_size=self.group_size,
        )

    def group_search_score_threshold(self, client: QdrantBase) -> models.GroupsResult:
        res1 = client.query_points_groups(
            collection_name=COLLECTION_NAME,
            using="text",
            query=self.query_text,
            with_payload=models.PayloadSelectorExclude(exclude=["city.geo", "rand_number"]),
            limit=self.limit,
            group_by=self.group_by,
            score_threshold=0.9,
            group_size=self.group_size,
        )

        res2 = client.query_points_groups(
            collection_name=COLLECTION_NAME,
            using="text",
            query=self.query_text,
            with_payload=models.PayloadSelectorExclude(exclude=["city.geo", "rand_number"]),
            limit=self.limit,
            group_by=self.group_by,
            score_threshold=0.95,
            group_size=self.group_size,
        )

        res3 = client.query_points_groups(
            collection_name=COLLECTION_NAME,
            using="text",
            query=self.query_text,
            with_payload=models.PayloadSelectorExclude(exclude=["city.geo", "rand_number"]),
            limit=self.limit,
            group_by=self.group_by,
            score_threshold=0.1,
            group_size=self.group_size,
        )

        return models.GroupsResult(groups=res1.groups + res2.groups + res3.groups)

    def group_search_text_select_payload(self, client: QdrantBase) -> models.GroupsResult:
        return client.query_points_groups(
            collection_name=COLLECTION_NAME,
            using="text",
            query=self.query_text,
            with_payload=["text_array", "nested.id"],
            limit=self.limit,
            group_by=self.group_by,
            group_size=self.group_size,
        )

    def group_search_payload_exclude(self, client: QdrantBase) -> models.GroupsResult:
        return client.query_points_groups(
            collection_name=COLLECTION_NAME,
            using="text",
            query=self.query_text,
            with_payload=models.PayloadSelectorExclude(
                exclude=["text_array", "nested.id", "city.geo", "rand_number"]
            ),
            limit=self.limit,
            group_by=self.group_by,
            group_size=self.group_size,
        )

    def group_search_image_select_vector(self, client: QdrantBase) -> models.GroupsResult:
        return client.query_points_groups(
            collection_name=COLLECTION_NAME,
            using="image",
            query=self.query_image,
            with_payload=False,
            with_vectors=["image", "code"],
            limit=self.limit,
            group_by=self.group_by,
            group_size=self.group_size,
        )

    def filter_group_search_text(
        self, client: QdrantBase, query_filter: models.Filter
    ) -> models.GroupsResult:
        return client.query_points_groups(
            collection_name=COLLECTION_NAME,
            using="text",
            query=self.query_text,
            query_filter=query_filter,
            with_payload=models.PayloadSelectorExclude(exclude=["city.geo", "rand_number"]),
            limit=self.limit,
            group_by=self.group_by,
            group_size=self.group_size,
        )

    def filter_group_search_text_single(
        self, client: QdrantBase, query_filter: models.Filter
    ) -> models.GroupsResult:
        return client.query_points_groups(
            collection_name=COLLECTION_NAME,
            query=self.query_text,
            query_filter=query_filter,
            with_payload=models.PayloadSelectorExclude(exclude=["city.geo", "rand_number"]),
            with_vectors=True,
            limit=self.limit,
            group_by=self.group_by,
            group_size=self.group_size,
        )


def group_by_keys():
    return ["id", "rand_digit", "two_words", "city.name", "maybe", "maybe_null"]


def test_group_search_types():
    fixture_points = generate_fixtures(vectors_sizes=50)
    vectors_config = models.VectorParams(size=50, distance=models.Distance.EUCLID)

    searcher = TestGroupSearcher()

    local_client = init_local()
    init_client(local_client, fixture_points, vectors_config=vectors_config)

    remote_client = init_remote()
    init_client(remote_client, fixture_points, vectors_config=vectors_config)

    query_vector_np = np.random.random(text_vector_size)
    compare_client_results(
        local_client,
        remote_client,
        searcher.group_search,
        query_vector=query_vector_np,
    )

    query_vector_list = query_vector_np.tolist()
    compare_client_results(
        local_client, remote_client, searcher.group_search, query_vector=query_vector_list
    )

    delete_fixture_collection(local_client)
    delete_fixture_collection(remote_client)


def test_simple_group_search():
    fixture_points = generate_fixtures()

    lookup_points = generate_fixtures(
        num=7,
        random_ids=False,  # Less that group ids to test the empty lookups
    )

    searcher = TestGroupSearcher()

    local_client = init_local()
    init_client(local_client, fixture_points)
    init_client(local_client, lookup_points, collection_name=LOOKUP_COLLECTION_NAME)

    remote_client = init_remote()
    init_client(remote_client, fixture_points)
    init_client(remote_client, lookup_points, collection_name=LOOKUP_COLLECTION_NAME)

    searcher.group_size = 1
    searcher.limit = 2
    for key in group_by_keys():
        searcher.group_by = key
        compare_client_results(local_client, remote_client, searcher.group_search_text)

    searcher.group_size = 3
    compare_client_results(local_client, remote_client, searcher.group_search_text)
    compare_client_results(local_client, remote_client, searcher.group_search_image)
    compare_client_results(local_client, remote_client, searcher.group_search_code)
    compare_client_results(local_client, remote_client, searcher.group_search_image_with_lookup)
    compare_client_results(local_client, remote_client, searcher.group_search_image_with_lookup_2)
    compare_client_results(local_client, remote_client, searcher.group_search_score_threshold)
    compare_client_results(local_client, remote_client, searcher.group_search_text_select_payload)
    compare_client_results(local_client, remote_client, searcher.group_search_image_select_vector)
    compare_client_results(local_client, remote_client, searcher.group_search_payload_exclude)

    for i in range(100):
        query_filter = one_random_filter_please()
        try:
            compare_client_results(
                local_client,
                remote_client,
                searcher.filter_group_search_text,
                query_filter=query_filter,
            )
        except AssertionError as e:
            print(f"\nFailed with filter {query_filter}")
            raise e


def test_single_vector():
    fixture_points = generate_fixtures(num=200, vectors_sizes=text_vector_size)

    searcher = TestGroupSearcher()

    vectors_config = models.VectorParams(
        size=text_vector_size,
        distance=models.Distance.DOT,
    )

    local_client = init_local()
    init_client(local_client, fixture_points, vectors_config=vectors_config)

    remote_client = init_remote()
    init_client(remote_client, fixture_points, vectors_config=vectors_config)

    for group_size in (1, 5):
        searcher.group_size = group_size

        for i in range(100):
            query_filter = one_random_filter_please()

            try:
                compare_client_results(
                    local_client,
                    remote_client,
                    searcher.filter_group_search_text_single,
                    query_filter=query_filter,
                )
            except AssertionError as e:
                print(f"\nFailed with filter {query_filter}")
                raise e


def test_search_with_persistence():
    import tempfile

    fixture_points = generate_fixtures()
    searcher = TestGroupSearcher()
    with tempfile.TemporaryDirectory() as tmpdir:
        local_client = init_local(tmpdir)
        init_client(local_client, fixture_points)

        payload_update_filter = one_random_filter_please()
        local_client.set_payload(COLLECTION_NAME, {"test": f"test"}, payload_update_filter)

        local_client.close()
        local_client_2 = init_local(tmpdir)

        remote_client = init_remote()
        init_client(remote_client, fixture_points)

        remote_client.set_payload(COLLECTION_NAME, {"test": f"test"}, payload_update_filter)

        payload_update_filter = one_random_filter_please()
        local_client_2.set_payload(COLLECTION_NAME, {"test": "test2"}, payload_update_filter)
        remote_client.set_payload(COLLECTION_NAME, {"test": "test2"}, payload_update_filter)

        for i in range(10):
            query_filter = one_random_filter_please()
            try:
                compare_client_results(
                    local_client_2,
                    remote_client,
                    searcher.filter_group_search_text,
                    query_filter=query_filter,
                )
            except AssertionError as e:
                print(f"\nFailed with filter {query_filter}")
                raise e


def test_group_search_value_types():
    """Only strings and integers can become a group id, and a bad value drops the whole point.

    `GroupId::try_from` accepts nothing else, and the aggregator ignores a point as soon as one
    of its `group_by` values fails to convert. `search_groups` goes through the same code.
    """
    points = [
        models.PointStruct(id=1, vector=[1.0, 0.0], payload={"a": True}),
        models.PointStruct(id=2, vector=[2.0, 0.0], payload={"a": 1}),
        models.PointStruct(id=3, vector=[3.0, 0.0], payload={"a": False}),
        models.PointStruct(id=4, vector=[4.0, 0.0], payload={"a": 0}),
        models.PointStruct(id=5, vector=[5.0, 0.0], payload={"a": "1"}),
        models.PointStruct(id=6, vector=[6.0, 0.0], payload={"a": 1.5}),
        models.PointStruct(id=7, vector=[7.0, 0.0], payload={"a": None}),
        models.PointStruct(id=8, vector=[8.0, 0.0], payload={"b": "no such key"}),
        models.PointStruct(id=9, vector=[9.0, 0.0], payload={"a": [2, True]}),
        models.PointStruct(id=10, vector=[10.0, 0.0], payload={"a": [3, None]}),
        models.PointStruct(id=11, vector=[11.0, 0.0], payload={"a": [4, 5]}),
    ]
    # bools (1, 3), a float (6), a null (7) and a missing key (8) form no group. Points 9 and
    # 10 are dropped as a whole, values 2 and 3 included, because one unsupported value in the
    # array is enough; only 11 spreads a point over several groups. The expected type of every
    # group id is spelled out, since `False`/`0` and `True`/`1` are equal in python and a
    # leaked bool would otherwise hide behind the integer group
    expected_groups = {
        (int, 0, (4,)),
        (int, 1, (2,)),
        (int, 4, (11,)),
        (int, 5, (11,)),
        (str, "1", (5,)),
    }

    vectors_config = models.VectorParams(size=2, distance=models.Distance.DOT)

    local_client = init_local()
    init_client(local_client, points, vectors_config=vectors_config)

    remote_client = init_remote()
    init_client(remote_client, points, vectors_config=vectors_config)

    for client in (local_client, remote_client):
        result = client.query_points_groups(
            COLLECTION_NAME, group_by="a", query=[1.0, 0.0], limit=10, group_size=10
        )
        assert {
            (type(group.id), group.id, tuple(hit.id for hit in group.hits))
            for group in result.groups
        } == expected_groups
