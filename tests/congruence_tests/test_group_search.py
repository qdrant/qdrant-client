import numpy as np

from qdrant_client.client_base import QdrantBase
from qdrant_client.http.models import models
from tests.congruence_tests.test_common import (
    COLLECTION_NAME,
    code_vector_size,
    compare_client_results,
    generate_fixtures,
    image_vector_size,
    init_client,
    init_local,
    init_remote,
    text_vector_size,
)
from tests.fixtures.filters import one_random_filter_please
from tests.fixtures.payload import one_random_payload_please


class TestGroupSearcher:
    __test__ = False

    def __init__(self):
        self.query_text = np.random.random(text_vector_size).tolist()
        self.query_image = np.random.random(image_vector_size).tolist()
        self.query_code = np.random.random(code_vector_size).tolist()
        self.group_by = "rand_digit"
        self.group_size = 1
        self.limit = 10

    def group_search_text(self, client: QdrantBase) -> models.GroupsResult:
        return client.search_groups(
            collection_name=COLLECTION_NAME,
            query_vector=("text", self.query_text),
            with_payload=models.PayloadSelectorExclude(exclude=["city.geo", "rand_number"]),
            group_by=self.group_by,
            limit=self.limit,
            group_size=self.group_size,
        )

    def group_search_text_single(self, client: QdrantBase) -> models.GroupsResult:
        return client.search_groups(
            collection_name=COLLECTION_NAME,
            query_vector=self.query_text,
            with_payload=models.PayloadSelectorExclude(exclude=["city.geo", "rand_number"]),
            group_by=self.group_by,
            limit=self.limit,
            group_size=self.group_size,
        )

    def group_search_image(self, client: QdrantBase) -> models.GroupsResult:
        return client.search_groups(
            collection_name=COLLECTION_NAME,
            query_vector=("image", self.query_image),
            with_payload=models.PayloadSelectorExclude(exclude=["city.geo", "rand_number"]),
            group_by=self.group_by,
            limit=self.limit,
            group_size=self.group_size,
        )

    def group_search_code(self, client: QdrantBase) -> models.GroupsResult:
        return client.search_groups(
            collection_name=COLLECTION_NAME,
            query_vector=("code", self.query_code),
            with_payload=models.PayloadSelectorExclude(exclude=["city.geo", "rand_number"]),
            group_by=self.group_by,
            limit=self.limit,
            group_size=self.group_size,
        )

    def group_search_score_threshold(self, client: QdrantBase) -> models.GroupsResult:
        res1 = client.search_groups(
            collection_name=COLLECTION_NAME,
            query_vector=("text", self.query_text),
            with_payload=models.PayloadSelectorExclude(exclude=["city.geo", "rand_number"]),
            limit=self.limit,
            group_by=self.group_by,
            score_threshold=0.9,
            group_size=self.group_size,
        )

        res2 = client.search_groups(
            collection_name=COLLECTION_NAME,
            query_vector=("text", self.query_text),
            with_payload=models.PayloadSelectorExclude(exclude=["city.geo", "rand_number"]),
            limit=self.limit,
            group_by=self.group_by,
            score_threshold=0.95,
            group_size=self.group_size,
        )

        res3 = client.search_groups(
            collection_name=COLLECTION_NAME,
            query_vector=("text", self.query_text),
            with_payload=models.PayloadSelectorExclude(exclude=["city.geo", "rand_number"]),
            limit=self.limit,
            group_by=self.group_by,
            score_threshold=0.1,
            group_size=self.group_size,
        )

        return models.GroupsResult(groups=res1.groups + res2.groups + res3.groups)

    def group_search_text_select_payload(self, client: QdrantBase) -> models.GroupsResult:
        return client.search_groups(
            collection_name=COLLECTION_NAME,
            query_vector=("text", self.query_text),
            with_payload=["text_array", "nested.id"],
            limit=self.limit,
            group_by=self.group_by,
            group_size=self.group_size,
        )

    def group_search_payload_exclude(self, client: QdrantBase) -> models.GroupsResult:
        return client.search_groups(
            collection_name=COLLECTION_NAME,
            query_vector=("text", self.query_text),
            with_payload=models.PayloadSelectorExclude(
                exclude=["text_array", "nested.id", "city.geo", "rand_number"]
            ),
            limit=self.limit,
            group_by=self.group_by,
            group_size=self.group_size,
        )

    def group_search_image_select_vector(self, client: QdrantBase) -> models.ScoredPoint:
        return client.search_groups(
            collection_name=COLLECTION_NAME,
            query_vector=("image", self.query_image),
            with_payload=False,
            with_vectors=["image", "code"],
            limit=self.limit,
            group_by=self.group_by,
            group_size=self.group_size,
        )

    def filter_group_search_text(
        self, client: QdrantBase, query_filter: models.Filter
    ) -> models.GroupsResult:
        return client.search_groups(
            collection_name=COLLECTION_NAME,
            query_vector=("text", self.query_text),
            query_filter=query_filter,
            with_payload=models.PayloadSelectorExclude(exclude=["city.geo", "rand_number"]),
            limit=self.limit,
            group_by=self.group_by,
            group_size=self.group_size,
        )

    def filter_group_search_text_single(
        self, client: QdrantBase, query_filter: models.Filter
    ) -> models.GroupsResult:
        return client.search_groups(
            collection_name=COLLECTION_NAME,
            query_vector=self.query_text,
            query_filter=query_filter,
            with_payload=models.PayloadSelectorExclude(exclude=["city.geo", "rand_number"]),
            with_vectors=True,
            limit=self.limit,
            group_by=self.group_by,
            group_size=self.group_size,
        )


def group_by_keys():
    random_payload = one_random_payload_please(0)
    keys = set(random_payload.keys())
    keys.add("maybe")
    keys.add("maybe_null")

    if "nested" in keys:
        keys.remove("nested")
    if "city" in keys:
        keys.remove("city")
        for city_key in random_payload["city"].keys():
            keys.add(f"city.{city_key}")

    # tmp until problem with absent keys in payload is solved
    if "maybe" in keys:
        keys.remove("maybe")
    if "maybe_null" in keys:
        keys.remove("maybe_null")
    return keys


def test_simple_group_search():
    fixture_records = generate_fixtures()

    searcher = TestGroupSearcher()

    local_client = init_local()
    init_client(local_client, fixture_records)

    remote_client = init_remote()
    init_client(remote_client, fixture_records)

    searcher.group_size = 1
    searcher.limit = 2
    for key in group_by_keys():
        searcher.group_by = key
        compare_client_results(local_client, remote_client, searcher.group_search_text)

    for group_size in (1, 5):
        searcher.group_size = group_size
        compare_client_results(local_client, remote_client, searcher.group_search_text)
        compare_client_results(local_client, remote_client, searcher.group_search_image)
        compare_client_results(local_client, remote_client, searcher.group_search_code)
        compare_client_results(local_client, remote_client, searcher.group_search_score_threshold)
        compare_client_results(
            local_client, remote_client, searcher.group_search_text_select_payload
        )
        compare_client_results(
            local_client, remote_client, searcher.group_search_image_select_vector
        )
        compare_client_results(local_client, remote_client, searcher.group_search_payload_exclude)

    for group_size in (1, 5):
        searcher.group_size = group_size
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
    fixture_records = generate_fixtures(num=200, vectors_sizes=text_vector_size)

    searcher = TestGroupSearcher()

    vectors_config = models.VectorParams(
        size=text_vector_size,
        distance=models.Distance.DOT,
    )

    local_client = init_local()
    init_client(local_client, fixture_records, vectors_config=vectors_config)

    remote_client = init_remote()
    init_client(remote_client, fixture_records, vectors_config=vectors_config)

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

    fixture_records = generate_fixtures()
    searcher = TestGroupSearcher()
    with tempfile.TemporaryDirectory() as tmpdir:
        local_client = init_local(tmpdir)
        init_client(local_client, fixture_records)

        payload_update_filter = one_random_filter_please()
        local_client.set_payload(COLLECTION_NAME, {"test": f"test"}, payload_update_filter)

        del local_client
        local_client_2 = init_local(tmpdir)

        remote_client = init_remote()
        init_client(remote_client, fixture_records)

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
