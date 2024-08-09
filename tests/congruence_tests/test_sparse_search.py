from typing import List

import numpy as np
import pytest

from qdrant_client.client_base import QdrantBase
from qdrant_client.conversions.common_types import NamedSparseVector
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.http.models import models
from tests.congruence_tests.test_common import (
    COLLECTION_NAME,
    compare_client_results,
    generate_sparse_fixtures,
    init_client,
    init_local,
    init_remote,
    sparse_code_vector_size,
    sparse_image_vector_size,
    sparse_text_vector_size,
    sparse_vectors_config,
)
from tests.fixtures.filters import one_random_filter_please
from tests.fixtures.points import generate_random_sparse_vector, random_sparse_vectors


class TestSimpleSparseSearcher:
    __test__ = False

    def __init__(self):
        self.query_text = generate_random_sparse_vector(sparse_text_vector_size, density=0.3)
        self.query_image = generate_random_sparse_vector(sparse_image_vector_size, density=0.2)
        self.query_code = generate_random_sparse_vector(sparse_code_vector_size, density=0.1)

    def simple_search_text(self, client: QdrantBase) -> List[models.ScoredPoint]:
        return client.search(
            collection_name=COLLECTION_NAME,
            query_vector=NamedSparseVector(name="sparse-text", vector=self.query_text),
            with_payload=True,
            with_vectors=["sparse-text"],
            limit=10,
        )

    def simple_search_image(self, client: QdrantBase) -> List[models.ScoredPoint]:
        return client.search(
            collection_name=COLLECTION_NAME,
            query_vector=NamedSparseVector(name="sparse-image", vector=self.query_image),
            with_payload=True,
            with_vectors=["sparse-image"],
            limit=10,
        )

    def simple_search_code(self, client: QdrantBase) -> List[models.ScoredPoint]:
        return client.search(
            collection_name=COLLECTION_NAME,
            query_vector=NamedSparseVector(name="sparse-code", vector=self.query_code),
            with_payload=True,
            with_vectors=True,
            limit=10,
        )

    def simple_search_text_offset(self, client: QdrantBase) -> List[models.ScoredPoint]:
        return client.search(
            collection_name=COLLECTION_NAME,
            query_vector=NamedSparseVector(name="sparse-text", vector=self.query_text),
            with_payload=True,
            limit=10,
            offset=10,
        )

    def search_score_threshold(self, client: QdrantBase) -> List[models.ScoredPoint]:
        res1 = client.search(
            collection_name=COLLECTION_NAME,
            query_vector=NamedSparseVector(name="sparse-text", vector=self.query_text),
            with_payload=True,
            limit=10,
            score_threshold=0.9,
        )

        res2 = client.search(
            collection_name=COLLECTION_NAME,
            query_vector=NamedSparseVector(name="sparse-text", vector=self.query_text),
            with_payload=True,
            limit=10,
            score_threshold=0.95,
        )

        res3 = client.search(
            collection_name=COLLECTION_NAME,
            query_vector=NamedSparseVector(name="sparse-text", vector=self.query_text),
            with_payload=True,
            limit=10,
            score_threshold=0.1,
        )

        return res1 + res2 + res3

    def simple_search_text_select_payload(self, client: QdrantBase) -> List[models.ScoredPoint]:
        return client.search(
            collection_name=COLLECTION_NAME,
            query_vector=NamedSparseVector(name="sparse-text", vector=self.query_text),
            with_payload=["text_array", "nested.id"],
            limit=10,
        )

    def search_payload_exclude(self, client: QdrantBase) -> List[models.ScoredPoint]:
        return client.search(
            collection_name=COLLECTION_NAME,
            query_vector=NamedSparseVector(name="sparse-text", vector=self.query_text),
            with_payload=models.PayloadSelectorExclude(exclude=["text_array", "nested.id"]),
            limit=10,
        )

    def simple_search_image_select_vector(self, client: QdrantBase) -> List[models.ScoredPoint]:
        return client.search(
            collection_name=COLLECTION_NAME,
            query_vector=NamedSparseVector(name="sparse-image", vector=self.query_image),
            with_payload=False,
            with_vectors=["sparse-image", "sparse-code"],
            limit=10,
        )

    def filter_search_text(
        self, client: QdrantBase, query_filter: models.Filter
    ) -> List[models.ScoredPoint]:
        return client.search(
            collection_name=COLLECTION_NAME,
            query_vector=NamedSparseVector(name="sparse-text", vector=self.query_text),
            query_filter=query_filter,
            with_payload=True,
            limit=10,
        )

    def filter_search_text_single(
        self, client: QdrantBase, query_filter: models.Filter
    ) -> List[models.ScoredPoint]:
        return client.search(
            collection_name=COLLECTION_NAME,
            query_vector=self.query_text,  # why it is not a NamedSparseVector?
            query_filter=query_filter,
            with_payload=True,
            with_vectors=True,
            limit=10,
        )


def test_simple_search():
    fixture_points = generate_sparse_fixtures()

    searcher = TestSimpleSparseSearcher()

    local_client = init_local()
    init_client(local_client, fixture_points, sparse_vectors_config=sparse_vectors_config)

    remote_client = init_remote()
    init_client(remote_client, fixture_points, sparse_vectors_config=sparse_vectors_config)

    compare_client_results(local_client, remote_client, searcher.simple_search_text)
    compare_client_results(local_client, remote_client, searcher.simple_search_image)
    compare_client_results(local_client, remote_client, searcher.simple_search_code)
    compare_client_results(local_client, remote_client, searcher.simple_search_text_offset)
    compare_client_results(local_client, remote_client, searcher.search_score_threshold)
    compare_client_results(local_client, remote_client, searcher.simple_search_text_select_payload)
    compare_client_results(local_client, remote_client, searcher.simple_search_image_select_vector)
    compare_client_results(local_client, remote_client, searcher.search_payload_exclude)

    for i in range(100):
        query_filter = one_random_filter_please()
        try:
            compare_client_results(
                local_client, remote_client, searcher.filter_search_text, query_filter=query_filter
            )
        except AssertionError as e:
            print(f"\nFailed with filter {query_filter}")
            raise e


def test_simple_opt_vectors_search():
    fixture_points = generate_sparse_fixtures(skip_vectors=True)

    searcher = TestSimpleSparseSearcher()

    local_client = init_local()
    init_client(local_client, fixture_points, sparse_vectors_config=sparse_vectors_config)

    remote_client = init_remote()
    init_client(remote_client, fixture_points, sparse_vectors_config=sparse_vectors_config)

    compare_client_results(local_client, remote_client, searcher.simple_search_text)
    compare_client_results(local_client, remote_client, searcher.simple_search_image)
    compare_client_results(local_client, remote_client, searcher.simple_search_code)
    compare_client_results(local_client, remote_client, searcher.simple_search_text_offset)
    compare_client_results(local_client, remote_client, searcher.search_score_threshold)
    compare_client_results(local_client, remote_client, searcher.simple_search_text_select_payload)
    compare_client_results(local_client, remote_client, searcher.simple_search_image_select_vector)
    compare_client_results(local_client, remote_client, searcher.search_payload_exclude)

    for i in range(100):
        query_filter = one_random_filter_please()
        try:
            compare_client_results(
                local_client, remote_client, searcher.filter_search_text, query_filter=query_filter
            )
        except AssertionError as e:
            print(f"\nFailed with filter {query_filter}")
            raise e


def test_search_with_persistence():
    import tempfile

    fixture_points = generate_sparse_fixtures()
    searcher = TestSimpleSparseSearcher()
    with tempfile.TemporaryDirectory() as tmpdir:
        local_client = init_local(tmpdir)
        init_client(local_client, fixture_points, sparse_vectors_config=sparse_vectors_config)

        payload_update_filter = one_random_filter_please()
        local_client.set_payload(COLLECTION_NAME, {"test": f"test"}, payload_update_filter)

        del local_client
        local_client_2 = init_local(tmpdir)

        remote_client = init_remote()
        init_client(remote_client, fixture_points, sparse_vectors_config=sparse_vectors_config)

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
                    searcher.filter_search_text,
                    query_filter=query_filter,
                )
            except AssertionError as e:
                print(f"\nFailed with filter {query_filter}")
                raise e


def test_search_with_persistence_and_skipped_vectors():
    import tempfile

    fixture_points = generate_sparse_fixtures(skip_vectors=True)
    searcher = TestSimpleSparseSearcher()
    with tempfile.TemporaryDirectory() as tmpdir:
        local_client = init_local(tmpdir)
        init_client(local_client, fixture_points, sparse_vectors_config=sparse_vectors_config)

        payload_update_filter = one_random_filter_please()
        local_client.set_payload(COLLECTION_NAME, {"test": f"test"}, payload_update_filter)

        count_before_load = local_client.count(COLLECTION_NAME)
        del local_client
        local_client_2 = init_local(tmpdir)

        count_after_load = local_client_2.count(COLLECTION_NAME)

        assert count_after_load == count_before_load

        remote_client = init_remote()
        init_client(remote_client, fixture_points, sparse_vectors_config=sparse_vectors_config)

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
                    searcher.filter_search_text,
                    query_filter=query_filter,
                )
            except AssertionError as e:
                print(f"\nFailed with filter {query_filter}")
                raise e


def test_query_with_nan():
    local_client = init_local()
    remote_client = init_remote()

    fixture_points = generate_sparse_fixtures()
    sparse_vector = random_sparse_vectors({"sparse-text": sparse_text_vector_size})
    named_sparse_vector = models.NamedSparseVector(
        name="sparse-text", vector=sparse_vector["sparse-text"]
    )
    named_sparse_vector.vector.values[0] = np.nan

    local_client.create_collection(
        COLLECTION_NAME, vectors_config={}, sparse_vectors_config=sparse_vectors_config
    )
    if remote_client.collection_exists(COLLECTION_NAME):
        remote_client.delete_collection(COLLECTION_NAME)
    remote_client.create_collection(
        COLLECTION_NAME, vectors_config={}, sparse_vectors_config=sparse_vectors_config
    )
    init_client(
        local_client,
        fixture_points,
        vectors_config={},
        sparse_vectors_config=sparse_vectors_config,
    )
    init_client(
        remote_client,
        fixture_points,
        vectors_config={},
        sparse_vectors_config=sparse_vectors_config,
    )

    with pytest.raises(AssertionError):
        local_client.search(COLLECTION_NAME, named_sparse_vector)
    with pytest.raises(UnexpectedResponse):
        remote_client.search(COLLECTION_NAME, named_sparse_vector)
