from pathlib import Path
from typing import Sequence, Union

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
        query_vector: Union[
            types.NumpyArray,
            Sequence[float],
            tuple[str, list[float]],
            types.NamedVector,
        ],
        collection_name: str = COLLECTION_NAME,
    ) -> models.GroupsResult:
        return client.search_groups(
            collection_name=collection_name,
            query_vector=query_vector,
            with_payload=models.PayloadSelectorExclude(exclude=["city.geo", "rand_number"]),
            group_by=self.group_by,
            limit=self.limit,
            group_size=self.group_size,
        )

    def group_search_text(
        self, client: QdrantBase, collection_name: str = COLLECTION_NAME
    ) -> models.GroupsResult:
        return client.search_groups(
            collection_name=collection_name,
            query_vector=("text", self.query_text),
            with_payload=models.PayloadSelectorExclude(exclude=["city.geo", "rand_number"]),
            group_by=self.group_by,
            limit=self.limit,
            group_size=self.group_size,
        )

    def group_search_text_single(
        self, client: QdrantBase, collection_name: str = COLLECTION_NAME
    ) -> models.GroupsResult:
        return client.search_groups(
            collection_name=collection_name,
            query_vector=self.query_text,
            with_payload=models.PayloadSelectorExclude(exclude=["city.geo", "rand_number"]),
            group_by=self.group_by,
            limit=self.limit,
            group_size=self.group_size,
        )

    def group_search_image(
        self, client: QdrantBase, collection_name: str = COLLECTION_NAME
    ) -> models.GroupsResult:
        return client.search_groups(
            collection_name=collection_name,
            query_vector=("image", self.query_image),
            with_payload=models.PayloadSelectorExclude(exclude=["city.geo", "rand_number"]),
            group_by=self.group_by,
            limit=self.limit,
            group_size=self.group_size,
        )

    def group_search_image_with_lookup(
        self,
        client: QdrantBase,
        collection_name: str = COLLECTION_NAME,
        lookup_collection_name: str = LOOKUP_COLLECTION_NAME,
    ) -> models.GroupsResult:
        return client.search_groups(
            collection_name=collection_name,
            query_vector=("image", self.query_image),
            with_payload=models.PayloadSelectorExclude(exclude=["city.geo", "rand_number"]),
            group_by=self.group_by,
            limit=self.limit,
            group_size=self.group_size,
            with_lookup=lookup_collection_name,
        )

    def group_search_image_with_lookup_2(
        self,
        client: QdrantBase,
        collection_name: str = COLLECTION_NAME,
        lookup_collection_name: str = LOOKUP_COLLECTION_NAME,
    ) -> models.GroupsResult:
        return client.search_groups(
            collection_name=collection_name,
            query_vector=("image", self.query_image),
            with_payload=models.PayloadSelectorExclude(exclude=["city.geo", "rand_number"]),
            group_by=self.group_by,
            limit=self.limit,
            group_size=self.group_size,
            with_lookup=models.WithLookup(
                collection=lookup_collection_name,
                with_payload=models.PayloadSelectorExclude(exclude=["city.geo", "rand_number"]),
                with_vectors=["image"],
            ),
        )

    def group_search_code(
        self, client: QdrantBase, collection_name: str = COLLECTION_NAME
    ) -> models.GroupsResult:
        return client.search_groups(
            collection_name=collection_name,
            query_vector=("code", self.query_code),
            with_payload=models.PayloadSelectorExclude(exclude=["city.geo", "rand_number"]),
            group_by=self.group_by,
            limit=self.limit,
            group_size=self.group_size,
        )

    def group_search_score_threshold(
        self, client: QdrantBase, collection_name: str = COLLECTION_NAME
    ) -> models.GroupsResult:
        res1 = client.search_groups(
            collection_name=collection_name,
            query_vector=("text", self.query_text),
            with_payload=models.PayloadSelectorExclude(exclude=["city.geo", "rand_number"]),
            limit=self.limit,
            group_by=self.group_by,
            score_threshold=0.9,
            group_size=self.group_size,
        )

        res2 = client.search_groups(
            collection_name=collection_name,
            query_vector=("text", self.query_text),
            with_payload=models.PayloadSelectorExclude(exclude=["city.geo", "rand_number"]),
            limit=self.limit,
            group_by=self.group_by,
            score_threshold=0.95,
            group_size=self.group_size,
        )

        res3 = client.search_groups(
            collection_name=collection_name,
            query_vector=("text", self.query_text),
            with_payload=models.PayloadSelectorExclude(exclude=["city.geo", "rand_number"]),
            limit=self.limit,
            group_by=self.group_by,
            score_threshold=0.1,
            group_size=self.group_size,
        )

        return models.GroupsResult(groups=res1.groups + res2.groups + res3.groups)

    def group_search_text_select_payload(
        self, client: QdrantBase, collection_name: str = COLLECTION_NAME
    ) -> models.GroupsResult:
        return client.search_groups(
            collection_name=collection_name,
            query_vector=("text", self.query_text),
            with_payload=["text_array", "nested.id"],
            limit=self.limit,
            group_by=self.group_by,
            group_size=self.group_size,
        )

    def group_search_payload_exclude(
        self, client: QdrantBase, collection_name: str = COLLECTION_NAME
    ) -> models.GroupsResult:
        return client.search_groups(
            collection_name=collection_name,
            query_vector=("text", self.query_text),
            with_payload=models.PayloadSelectorExclude(
                exclude=["text_array", "nested.id", "city.geo", "rand_number"]
            ),
            limit=self.limit,
            group_by=self.group_by,
            group_size=self.group_size,
        )

    def group_search_image_select_vector(
        self, client: QdrantBase, collection_name: str = COLLECTION_NAME
    ) -> models.GroupsResult:
        return client.search_groups(
            collection_name=collection_name,
            query_vector=("image", self.query_image),
            with_payload=False,
            with_vectors=["image", "code"],
            limit=self.limit,
            group_by=self.group_by,
            group_size=self.group_size,
        )

    def filter_group_search_text(
        self,
        client: QdrantBase,
        query_filter: models.Filter,
        collection_name: str = COLLECTION_NAME,
    ) -> models.GroupsResult:
        return client.search_groups(
            collection_name=collection_name,
            query_vector=("text", self.query_text),
            query_filter=query_filter,
            with_payload=models.PayloadSelectorExclude(exclude=["city.geo", "rand_number"]),
            limit=self.limit,
            group_by=self.group_by,
            group_size=self.group_size,
        )

    def filter_group_search_text_single(
        self,
        client: QdrantBase,
        query_filter: models.Filter,
        collection_name: str = COLLECTION_NAME,
    ) -> models.GroupsResult:
        return client.search_groups(
            collection_name=collection_name,
            query_vector=self.query_text,
            query_filter=query_filter,
            with_payload=models.PayloadSelectorExclude(exclude=["city.geo", "rand_number"]),
            with_vectors=True,
            limit=self.limit,
            group_by=self.group_by,
            group_size=self.group_size,
        )


def group_by_keys():
    return ["id", "rand_digit", "two_words", "city.name", "maybe", "maybe_null"]


def test_group_search_types(
    local_client: QdrantBase, remote_client: QdrantBase, collection_name: str
):
    fixture_points = generate_fixtures(vectors_sizes=50)
    vectors_config = models.VectorParams(size=50, distance=models.Distance.EUCLID)

    searcher = TestGroupSearcher()

    init_client(
        local_client,
        fixture_points,
        collection_name=collection_name,
        vectors_config=vectors_config,
    )
    init_client(
        remote_client,
        fixture_points,
        collection_name=collection_name,
        vectors_config=vectors_config,
    )

    query_vector_np = np.random.random(text_vector_size)
    compare_client_results(
        local_client,
        remote_client,
        searcher.group_search,
        query_vector=query_vector_np,
        collection_name=collection_name,
    )

    query_vector_list = query_vector_np.tolist()
    compare_client_results(
        local_client,
        remote_client,
        searcher.group_search,
        query_vector=query_vector_list,
        collection_name=collection_name,
    )

    delete_fixture_collection(local_client, collection_name=collection_name)
    delete_fixture_collection(remote_client, collection_name=collection_name)

    fixture_points = generate_fixtures()
    init_client(local_client, fixture_points, collection_name=collection_name)
    init_client(remote_client, fixture_points, collection_name=collection_name)

    query_vector_tuple = ("text", query_vector_list)
    compare_client_results(
        local_client,
        remote_client,
        searcher.group_search,
        query_vector=query_vector_tuple,
        collection_name=collection_name,
    )

    query_named_vector = types.NamedVector(name="text", vector=query_vector_list)
    compare_client_results(
        local_client,
        remote_client,
        searcher.group_search,
        query_vector=query_named_vector,
        collection_name=collection_name,
    )

    delete_fixture_collection(local_client, collection_name=collection_name)
    delete_fixture_collection(remote_client, collection_name=collection_name)


def test_simple_group_search(
    local_client: QdrantBase,
    remote_client: QdrantBase,
    collection_name: str,
    secondary_collection_name: str,
):
    fixture_points = generate_fixtures()

    lookup_points = generate_fixtures(
        num=7,
        random_ids=False,  # Less that group ids to test the empty lookups
    )

    searcher = TestGroupSearcher()

    init_client(local_client, fixture_points, collection_name=collection_name)
    init_client(local_client, lookup_points, collection_name=secondary_collection_name)

    init_client(remote_client, fixture_points, collection_name=collection_name)
    init_client(remote_client, lookup_points, collection_name=secondary_collection_name)

    searcher.group_size = 1
    searcher.limit = 2
    for key in group_by_keys():
        searcher.group_by = key
        compare_client_results(
            local_client,
            remote_client,
            searcher.group_search_text,
            collection_name=collection_name,
        )

    searcher.group_size = 3
    compare_client_results(
        local_client,
        remote_client,
        searcher.group_search_text,
        collection_name=collection_name,
    )
    compare_client_results(
        local_client,
        remote_client,
        searcher.group_search_image,
        collection_name=collection_name,
    )
    compare_client_results(
        local_client,
        remote_client,
        searcher.group_search_code,
        collection_name=collection_name,
    )
    compare_client_results(
        local_client,
        remote_client,
        searcher.group_search_image_with_lookup,
        collection_name=collection_name,
        lookup_collection_name=secondary_collection_name,
    )
    compare_client_results(
        local_client,
        remote_client,
        searcher.group_search_image_with_lookup_2,
        collection_name=collection_name,
        lookup_collection_name=secondary_collection_name,
    )
    compare_client_results(
        local_client,
        remote_client,
        searcher.group_search_score_threshold,
        collection_name=collection_name,
    )
    compare_client_results(
        local_client,
        remote_client,
        searcher.group_search_text_select_payload,
        collection_name=collection_name,
    )
    compare_client_results(
        local_client,
        remote_client,
        searcher.group_search_image_select_vector,
        collection_name=collection_name,
    )
    compare_client_results(
        local_client,
        remote_client,
        searcher.group_search_payload_exclude,
        collection_name=collection_name,
    )

    for _ in range(100):
        query_filter = one_random_filter_please()
        try:
            compare_client_results(
                local_client,
                remote_client,
                searcher.filter_group_search_text,
                query_filter=query_filter,
                collection_name=collection_name,
            )
        except AssertionError as e:
            print(f"\nFailed with filter {query_filter}")
            raise e


def test_single_vector(local_client: QdrantBase, remote_client: QdrantBase, collection_name: str):
    fixture_points = generate_fixtures(num=200, vectors_sizes=text_vector_size)

    searcher = TestGroupSearcher()

    vectors_config = models.VectorParams(
        size=text_vector_size,
        distance=models.Distance.DOT,
    )

    init_client(
        local_client,
        fixture_points,
        collection_name=collection_name,
        vectors_config=vectors_config,
    )
    init_client(
        remote_client,
        fixture_points,
        collection_name=collection_name,
        vectors_config=vectors_config,
    )

    for group_size in (1, 5):
        searcher.group_size = group_size

        for _ in range(100):
            query_filter = one_random_filter_please()

            try:
                compare_client_results(
                    local_client,
                    remote_client,
                    searcher.filter_group_search_text_single,
                    query_filter=query_filter,
                    collection_name=collection_name,
                )
            except AssertionError as e:
                print(f"\nFailed with filter {query_filter}")
                raise e


def test_search_with_persistence(
    local_client: QdrantBase, remote_client: QdrantBase, tmp_path: Path, collection_name: str
):
    fixture_points = generate_fixtures()
    searcher = TestGroupSearcher()

    tmpdir: str = str(tmp_path)

    local_client = init_local(tmpdir)

    init_client(local_client, fixture_points, collection_name=collection_name)

    payload_update_filter = one_random_filter_please()
    local_client.set_payload(collection_name, {"test": f"test"}, payload_update_filter)

    del local_client
    local_client_2 = init_local(tmpdir)

    init_client(remote_client, fixture_points, collection_name=collection_name)

    remote_client.set_payload(collection_name, {"test": f"test"}, payload_update_filter)

    payload_update_filter = one_random_filter_please()
    local_client_2.set_payload(collection_name, {"test": "test2"}, payload_update_filter)
    remote_client.set_payload(collection_name, {"test": "test2"}, payload_update_filter)

    for _ in range(10):
        query_filter = one_random_filter_please()
        try:
            compare_client_results(
                local_client_2,
                remote_client,
                searcher.filter_group_search_text,
                query_filter=query_filter,
                collection_name=collection_name,
            )
        except AssertionError as e:
            print(f"\nFailed with filter {query_filter}")
            raise e
