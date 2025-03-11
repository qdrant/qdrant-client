from pathlib import Path
import pytest

import numpy as np

from qdrant_client.client_base import QdrantBase
from qdrant_client.http.exceptions import UnexpectedResponse
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


class TestSimpleSearcher:
    __test__ = False

    def __init__(self):
        self.query_text = np.random.random(text_vector_size).tolist()
        self.query_image = np.random.random(image_vector_size).tolist()
        self.query_code = np.random.random(code_vector_size).tolist()

    def simple_search_text(
        self, client: QdrantBase, collection_name: str = COLLECTION_NAME
    ) -> list[models.ScoredPoint]:
        return client.search(
            collection_name=collection_name,
            query_vector=("text", self.query_text),
            with_payload=True,
            limit=10,
        )

    def simple_search_image(
        self, client: QdrantBase, collection_name: str = COLLECTION_NAME
    ) -> list[models.ScoredPoint]:
        return client.search(
            collection_name=collection_name,
            query_vector=("image", self.query_image),
            with_payload=True,
            limit=10,
        )

    def simple_search_code(
        self, client: QdrantBase, collection_name: str = COLLECTION_NAME
    ) -> list[models.ScoredPoint]:
        return client.search(
            collection_name=collection_name,
            query_vector=("code", self.query_code),
            with_payload=True,
            limit=10,
        )

    def simple_search_text_offset(
        self, client: QdrantBase, collection_name: str = COLLECTION_NAME
    ) -> list[models.ScoredPoint]:
        return client.search(
            collection_name=collection_name,
            query_vector=("text", self.query_text),
            with_payload=True,
            limit=10,
            offset=10,
        )

    def simple_search_text_with_vector(
        self, client: QdrantBase, collection_name: str = COLLECTION_NAME
    ) -> list[models.ScoredPoint]:
        return client.search(
            collection_name=collection_name,
            query_vector=("text", self.query_text),
            with_payload=True,
            with_vectors=True,
            limit=10,
            offset=10,
        )

    def search_score_threshold(
        self, client: QdrantBase, collection_name: str = COLLECTION_NAME
    ) -> list[models.ScoredPoint]:
        res1 = client.search(
            collection_name=collection_name,
            query_vector=("text", self.query_text),
            with_payload=True,
            limit=10,
            score_threshold=0.9,
        )

        res2 = client.search(
            collection_name=collection_name,
            query_vector=("text", self.query_text),
            with_payload=True,
            limit=10,
            score_threshold=0.95,
        )

        res3 = client.search(
            collection_name=collection_name,
            query_vector=("text", self.query_text),
            with_payload=True,
            limit=10,
            score_threshold=0.1,
        )

        return res1 + res2 + res3

    def simple_search_text_select_payload(
        self, client: QdrantBase, collection_name: str = COLLECTION_NAME
    ) -> list[models.ScoredPoint]:
        return client.search(
            collection_name=collection_name,
            query_vector=("text", self.query_text),
            with_payload=["text_array", "nested.id"],
            limit=10,
        )

    def search_payload_exclude(
        self, client: QdrantBase, collection_name: str = COLLECTION_NAME
    ) -> list[models.ScoredPoint]:
        return client.search(
            collection_name=collection_name,
            query_vector=("text", self.query_text),
            with_payload=models.PayloadSelectorExclude(exclude=["text_array", "nested.id"]),
            limit=10,
        )

    def simple_search_image_select_vector(
        self, client: QdrantBase, collection_name: str = COLLECTION_NAME
    ) -> list[models.ScoredPoint]:
        return client.search(
            collection_name=collection_name,
            query_vector=("image", self.query_image),
            with_payload=False,
            with_vectors=["image", "code"],
            limit=10,
        )

    def filter_search_text(
        self,
        client: QdrantBase,
        query_filter: models.Filter,
        collection_name: str = COLLECTION_NAME,
    ) -> list[models.ScoredPoint]:
        return client.search(
            collection_name=collection_name,
            query_vector=("text", self.query_text),
            query_filter=query_filter,
            with_payload=True,
            limit=10,
        )

    def filter_search_text_single(
        self,
        client: QdrantBase,
        query_filter: models.Filter,
        collection_name: str = COLLECTION_NAME,
    ) -> list[models.ScoredPoint]:
        return client.search(
            collection_name=collection_name,
            query_vector=self.query_text,
            query_filter=query_filter,
            with_payload=True,
            with_vectors=True,
            limit=10,
        )


def test_simple_search(collection_name: str):
    fixture_points = generate_fixtures()

    searcher = TestSimpleSearcher()

    local_client = init_local()
    init_client(local_client, fixture_points, collection_name)

    remote_client = init_remote()
    init_client(remote_client, fixture_points, collection_name)

    compare_client_results(
        local_client, remote_client, searcher.simple_search_text, collection_name=collection_name
    )
    compare_client_results(
        local_client, remote_client, searcher.simple_search_image, collection_name=collection_name
    )
    compare_client_results(
        local_client, remote_client, searcher.simple_search_code, collection_name=collection_name
    )
    compare_client_results(
        local_client,
        remote_client,
        searcher.simple_search_text_offset,
        collection_name=collection_name,
    )
    compare_client_results(
        local_client,
        remote_client,
        searcher.simple_search_text_with_vector,
        collection_name=collection_name,
    )
    compare_client_results(
        local_client,
        remote_client,
        searcher.search_score_threshold,
        collection_name=collection_name,
    )
    compare_client_results(
        local_client,
        remote_client,
        searcher.simple_search_text_select_payload,
        collection_name=collection_name,
    )
    compare_client_results(
        local_client,
        remote_client,
        searcher.simple_search_image_select_vector,
        collection_name=collection_name,
    )
    compare_client_results(
        local_client,
        remote_client,
        searcher.search_payload_exclude,
        collection_name=collection_name,
    )

    for i in range(100):
        query_filter = one_random_filter_please()
        try:
            compare_client_results(
                local_client,
                remote_client,
                searcher.filter_search_text,
                query_filter=query_filter,
                collection_name=collection_name,
            )
        except AssertionError as e:
            print(f"\nFailed with filter {query_filter}")
            raise e


def test_simple_opt_vectors_search(collection_name: str):
    fixture_points = generate_fixtures(skip_vectors=True)

    searcher = TestSimpleSearcher()

    local_client = init_local()
    init_client(local_client, fixture_points, collection_name)

    remote_client = init_remote()
    init_client(remote_client, fixture_points, collection_name)

    compare_client_results(
        local_client, remote_client, searcher.simple_search_text, collection_name=collection_name
    )
    compare_client_results(
        local_client, remote_client, searcher.simple_search_image, collection_name=collection_name
    )
    compare_client_results(
        local_client, remote_client, searcher.simple_search_code, collection_name=collection_name
    )
    compare_client_results(
        local_client,
        remote_client,
        searcher.simple_search_text_offset,
        collection_name=collection_name,
    )
    compare_client_results(
        local_client,
        remote_client,
        searcher.simple_search_text_with_vector,
        collection_name=collection_name,
    )
    compare_client_results(
        local_client,
        remote_client,
        searcher.search_score_threshold,
        collection_name=collection_name,
    )
    compare_client_results(
        local_client,
        remote_client,
        searcher.simple_search_text_select_payload,
        collection_name=collection_name,
    )
    compare_client_results(
        local_client,
        remote_client,
        searcher.simple_search_image_select_vector,
        collection_name=collection_name,
    )
    compare_client_results(
        local_client,
        remote_client,
        searcher.search_payload_exclude,
        collection_name=collection_name,
    )

    for i in range(100):
        query_filter = one_random_filter_please()
        try:
            compare_client_results(
                local_client,
                remote_client,
                searcher.filter_search_text,
                query_filter=query_filter,
                collection_name=collection_name,
            )
        except AssertionError as e:
            print(f"\nFailed with filter {query_filter}")
            raise e


def test_single_vector(collection_name: str):
    fixture_points = generate_fixtures(num=200, vectors_sizes=text_vector_size)

    searcher = TestSimpleSearcher()

    vectors_config = models.VectorParams(
        size=text_vector_size,
        distance=models.Distance.DOT,
    )

    local_client = init_local()
    init_client(local_client, fixture_points, collection_name, vectors_config=vectors_config)

    remote_client = init_remote()
    init_client(remote_client, fixture_points, collection_name, vectors_config=vectors_config)

    for _ in range(100):
        query_filter = one_random_filter_please()
        try:
            compare_client_results(
                local_client,
                remote_client,
                searcher.filter_search_text_single,
                query_filter=query_filter,
                collection_name=collection_name,
            )
        except AssertionError as e:
            print(f"\nFailed with filter {query_filter}")
            raise e


def test_search_with_persistence(tmp_path: Path, collection_name: str):
    fixture_points = generate_fixtures()
    searcher = TestSimpleSearcher()

    tmpdir = str(tmp_path)
    local_client = init_local(tmpdir)
    init_client(local_client, fixture_points, collection_name)

    payload_update_filter = one_random_filter_please()
    local_client.set_payload(collection_name, {"test": f"test"}, payload_update_filter)

    del local_client
    local_client_2 = init_local(tmpdir)

    remote_client = init_remote()
    init_client(remote_client, fixture_points, collection_name)

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
                searcher.filter_search_text,
                query_filter=query_filter,
                collection_name=collection_name,
            )
        except AssertionError as e:
            print(f"\nFailed with filter {query_filter}")
            raise e


def test_search_with_persistence_and_skipped_vectors(tmp_path: Path, collection_name: str):
    fixture_points = generate_fixtures(skip_vectors=True)
    searcher = TestSimpleSearcher()

    tmpdir = str(tmp_path)

    local_client = init_local(tmpdir)
    init_client(local_client, fixture_points, collection_name)

    payload_update_filter = one_random_filter_please()
    local_client.set_payload(collection_name, {"test": f"test"}, payload_update_filter)

    count_before_load = local_client.count(collection_name)
    del local_client
    local_client_2 = init_local(tmpdir)

    count_after_load = local_client_2.count(collection_name)

    assert count_after_load == count_before_load

    remote_client = init_remote()
    init_client(remote_client, fixture_points, collection_name)

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
                searcher.filter_search_text,
                query_filter=query_filter,
                collection_name=collection_name,
            )
        except AssertionError as e:
            print(f"\nFailed with filter {query_filter}")
            raise e


def test_search_invalid_vector_type(collection_name: str):
    fixture_points = generate_fixtures()

    local_client = init_local()
    init_client(local_client, fixture_points, collection_name)

    remote_client = init_remote()
    init_client(remote_client, fixture_points, collection_name)

    vector_invalid_type = {"text": [1, 2, 3, 4]}
    with pytest.raises(ValueError):
        local_client.search(collection_name=collection_name, query_vector=vector_invalid_type)

    with pytest.raises(ValueError):
        remote_client.search(collection_name=collection_name, query_vector=vector_invalid_type)


def test_query_with_nan(collection_name: str):
    fixture_points = generate_fixtures()

    local_client = init_local()
    init_client(local_client, fixture_points, collection_name)

    remote_client = init_remote()
    init_client(remote_client, fixture_points, collection_name)

    vector = np.random.random(text_vector_size)
    vector[4] = np.nan
    query_vector = ("text", vector.tolist())
    with pytest.raises(AssertionError):
        local_client.search(collection_name, query_vector)
    with pytest.raises(UnexpectedResponse):
        remote_client.search(collection_name, query_vector)

    single_vector_config = models.VectorParams(
        size=text_vector_size, distance=models.Distance.COSINE
    )

    local_client.delete_collection(collection_name)
    local_client.create_collection(collection_name, vectors_config=single_vector_config)

    remote_client.delete_collection(collection_name)
    remote_client.create_collection(collection_name, vectors_config=single_vector_config)

    fixture_points = generate_fixtures(vectors_sizes=text_vector_size)
    init_client(local_client, fixture_points, collection_name, vectors_config=single_vector_config)
    init_client(
        remote_client, fixture_points, collection_name, vectors_config=single_vector_config
    )

    with pytest.raises(AssertionError):
        local_client.search(collection_name, vector.tolist())
    with pytest.raises(UnexpectedResponse):
        remote_client.search(collection_name, vector.tolist())
