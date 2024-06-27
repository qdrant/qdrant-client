from typing import List

import numpy as np
import pytest

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

    def simple_query_text(self, client: QdrantBase) -> List[models.ScoredPoint]:
        return client.query_points(
            collection_name=COLLECTION_NAME,
            query=self.query_text,
            using="text",
            with_payload=True,
            limit=10,
        )

    def simple_query_image(self, client: QdrantBase) -> List[models.ScoredPoint]:
        return client.query_points(
            collection_name=COLLECTION_NAME,
            query=self.query_image,
            using="image",
            with_payload=True,
            limit=10,
        )

    def simple_query_code(self, client: QdrantBase) -> List[models.ScoredPoint]:
        return client.query_points(
            collection_name=COLLECTION_NAME,
            query=self.query_code,
            using="code",
            with_payload=True,
            limit=10,
        )

    def simple_query_text_offset(self, client: QdrantBase) -> List[models.ScoredPoint]:
        return client.query_points(
            collection_name=COLLECTION_NAME,
            query=self.query_text,
            using="text",
            with_payload=True,
            limit=10,
            offset=10,
        )

    def simple_query_text_with_vector(self, client: QdrantBase) -> List[models.ScoredPoint]:
        return client.query_points(
            collection_name=COLLECTION_NAME,
            query=self.query_text,
            using="text",
            with_payload=True,
            with_vectors=True,
            limit=10,
            offset=10,
        )

    def query_score_threshold(self, client: QdrantBase) -> List[models.ScoredPoint]:
        res1 = client.query_points(
            collection_name=COLLECTION_NAME,
            query=self.query_text,
            using="text",
            with_payload=True,
            limit=10,
            score_threshold=0.9,
        ).points

        res2 = client.query_points(
            collection_name=COLLECTION_NAME,
            query=self.query_text,
            using="text",
            with_payload=True,
            limit=10,
            score_threshold=0.95,
        ).points

        res3 = client.query_points(
            collection_name=COLLECTION_NAME,
            query=self.query_text,
            using="text",
            with_payload=True,
            limit=10,
            score_threshold=0.1,
        ).points

        return res1 + res2 + res3

    def simple_query_text_select_payload(self, client: QdrantBase) -> List[models.ScoredPoint]:
        return client.query_points(
            collection_name=COLLECTION_NAME,
            query=self.query_text,
            using="text",
            with_payload=["text_array", "nested.id"],
            limit=10,
        )

    def query_payload_exclude(self, client: QdrantBase) -> List[models.ScoredPoint]:
        return client.query_points(
            collection_name=COLLECTION_NAME,
            query=self.query_text,
            using="text",
            with_payload=models.PayloadSelectorExclude(exclude=["text_array", "nested.id"]),
            limit=10,
        )

    def simple_query_image_select_vector(self, client: QdrantBase) -> List[models.ScoredPoint]:
        return client.query_points(
            collection_name=COLLECTION_NAME,
            query=self.query_image,
            using="image",
            with_payload=False,
            with_vectors=["image", "code"],
            limit=10,
        )

    def filter_query_text(
            self, client: QdrantBase, query_filter: models.Filter
    ) -> List[models.ScoredPoint]:
        return client.query_points(
            collection_name=COLLECTION_NAME,
            query=self.query_text,
            using="text",
            query_filter=query_filter,
            with_payload=True,
            limit=10,
        )

    def filter_query_text_single(
            self, client: QdrantBase, query_filter: models.Filter
    ) -> List[models.ScoredPoint]:
        return client.query_points(
            collection_name=COLLECTION_NAME,
            query=self.query_text,
            query_filter=query_filter,
            with_payload=True,
            with_vectors=True,
            limit=10,
        )

    @classmethod
    def simple_query_text_scroll(
            cls, client: QdrantBase, query_filter: models.Filter
    ) -> List[models.ScoredPoint]:
        return client.query_points(
            collection_name=COLLECTION_NAME,
            using="text",
            query_filter=query_filter,
            with_payload=True,
            with_vectors=True,
            limit=10,
        )

    def simple_query_fusion(self, client: QdrantBase) -> List[models.ScoredPoint]:
        return client.query_points(
            collection_name=COLLECTION_NAME,
            prefetch=[
                models.Prefetch(
                    query=self.query_text,
                    using="text",
                )
            ],
            query=models.FusionQuery(
                fusion=models.Fusion.RRF
            ),
            with_payload=True,
            limit=10,
        )

    def simple_queries_rescore(self, client: QdrantBase) -> List[models.ScoredPoint]:
        return client.query_points(
            collection_name=COLLECTION_NAME,
            prefetch=[
                models.Prefetch(
                    query=self.query_text,
                    using="text",
                ),
                models.Prefetch(
                    query=self.query_code,
                    using="code",
                )
            ],
            query=self.query_image,
            using="image",
            with_payload=True,
            limit=10,
        )

    def simple_queries_orderby(self, client: QdrantBase) -> List[models.ScoredPoint]:
        return client.query_points(
            collection_name=COLLECTION_NAME,
            prefetch=[
                models.Prefetch(
                    query=self.query_text,
                    using="text",
                ),
                models.Prefetch(
                    query=self.query_code,
                    using="code",
                )
            ],
            query=models.OrderByQuery(
                order_by="rand_digit",
            ),
            with_payload=True,
            limit=10,
        )

    @classmethod
    def simple_recommend_image(cls, client: QdrantBase) -> List[models.ScoredPoint]:
        return client.query_points(
            collection_name=COLLECTION_NAME,
            query=models.RecommendQuery(
                recommend=models.RecommendInput(
                    positive=[10],
                )
            ),
            with_payload=True,
            limit=10,
            using="image",
        )

    @classmethod
    def many_recommend(cls, client: QdrantBase) -> List[models.ScoredPoint]:
        return client.query_points(
            collection_name=COLLECTION_NAME,
            query=models.RecommendQuery(
                recommend=models.RecommendInput(
                    positive=[10, 19],
                )
            ),
            with_payload=True,
            limit=10,
            using="image",
        )

    @classmethod
    def simple_discovery_image(cls, client: QdrantBase) -> List[models.ScoredPoint]:
        return client.query_points(
            collection_name=COLLECTION_NAME,
            query=models.DiscoverQuery(
                discover=models.DiscoverInput(
                    target=10,
                    context=models.ContextPair(positive=11, negative=19),
                )
            ),
            with_payload=True,
            limit=10,
            using="image",
        )

    @classmethod
    def many_discover(cls, client: QdrantBase) -> List[models.ScoredPoint]:
        return client.query_points(
            collection_name=COLLECTION_NAME,
            query=models.DiscoverQuery(
                discover=models.DiscoverInput(
                    target=10,
                    context=[
                        models.ContextPair(positive=11, negative=19),
                        models.ContextPair(positive=12, negative=20),
                    ]
                )
            ),
            with_payload=True,
            limit=10,
            using="image",
        )

    @classmethod
    def simple_context_image(cls, client: QdrantBase) -> List[models.ScoredPoint]:
        return client.query_points(
            collection_name=COLLECTION_NAME,
            query=models.ContextQuery(
                context=models.ContextPair(positive=11, negative=19)
            ),
            with_payload=True,
            limit=10,
            using="image",
        )

# ---- TESTS  ---- #


def test_simple_query():
    fixture_points = generate_fixtures()

    searcher = TestSimpleSearcher()

    local_client = init_local()
    init_client(local_client, fixture_points)

    remote_client = init_remote()
    init_client(remote_client, fixture_points)

    compare_client_results(local_client, remote_client, searcher.simple_query_text)
    compare_client_results(local_client, remote_client, searcher.simple_query_image)
    compare_client_results(local_client, remote_client, searcher.simple_query_code)
    compare_client_results(local_client, remote_client, searcher.simple_query_text_offset)
    compare_client_results(local_client, remote_client, searcher.simple_query_text_with_vector)
    compare_client_results(local_client, remote_client, searcher.query_score_threshold)
    compare_client_results(local_client, remote_client, searcher.simple_query_text_select_payload)
    compare_client_results(local_client, remote_client, searcher.simple_query_image_select_vector)
    compare_client_results(local_client, remote_client, searcher.query_payload_exclude)

    for i in range(100):
        query_filter = one_random_filter_please()
        try:
            compare_client_results(
                local_client, remote_client, searcher.filter_query_text, query_filter=query_filter
            )
            compare_client_results(
                local_client, remote_client, searcher.simple_query_text_scroll, query_filter=query_filter,
            )
        except AssertionError as e:
            print(f"\nFailed with filter {query_filter}")
            raise e


def test_simple_query_orderby():
    fixture_points = generate_fixtures(200)

    searcher = TestSimpleSearcher()

    local_client = init_local()
    init_client(local_client, fixture_points)

    remote_client = init_remote()
    init_client(remote_client, fixture_points)

    remote_client.create_payload_index(
        COLLECTION_NAME, "rand_digit", models.PayloadSchemaType.INTEGER, wait=True
    )

    compare_client_results(local_client, remote_client, searcher.simple_queries_orderby)


def test_simple_query_recommend():
    fixture_points = generate_fixtures()

    searcher = TestSimpleSearcher()

    local_client = init_local()
    init_client(local_client, fixture_points)

    remote_client = init_remote()
    init_client(remote_client, fixture_points)

    compare_client_results(local_client, remote_client, searcher.simple_recommend_image)
    compare_client_results(local_client, remote_client, searcher.many_recommend)


def test_simple_query_rescore():
    fixture_points = generate_fixtures()

    searcher = TestSimpleSearcher()

    local_client = init_local()
    init_client(local_client, fixture_points)

    remote_client = init_remote()
    init_client(remote_client, fixture_points)

    compare_client_results(local_client, remote_client, searcher.simple_queries_rescore)


def test_simple_query_fusion():
    fixture_points = generate_fixtures()

    searcher = TestSimpleSearcher()

    local_client = init_local()
    init_client(local_client, fixture_points)

    remote_client = init_remote()
    init_client(remote_client, fixture_points)

    compare_client_results(local_client, remote_client, searcher.simple_query_fusion)


def test_simple_query_discovery_context():
    fixture_points = generate_fixtures()

    searcher = TestSimpleSearcher()

    local_client = init_local()
    init_client(local_client, fixture_points)

    remote_client = init_remote()
    init_client(remote_client, fixture_points)

    compare_client_results(local_client, remote_client, searcher.simple_discovery_image)
    compare_client_results(local_client, remote_client, searcher.many_discover)
    compare_client_results(local_client, remote_client, searcher.simple_context_image, is_context_search=True)


def test_simple_opt_vectors_query():
    fixture_points = generate_fixtures(skip_vectors=True)

    searcher = TestSimpleSearcher()

    local_client = init_local()
    init_client(local_client, fixture_points)

    remote_client = init_remote()
    init_client(remote_client, fixture_points)

    compare_client_results(local_client, remote_client, searcher.simple_query_text)
    compare_client_results(local_client, remote_client, searcher.simple_query_image)
    compare_client_results(local_client, remote_client, searcher.simple_query_code)
    compare_client_results(local_client, remote_client, searcher.simple_query_text_offset)
    compare_client_results(local_client, remote_client, searcher.simple_query_text_with_vector)
    compare_client_results(local_client, remote_client, searcher.query_score_threshold)
    compare_client_results(local_client, remote_client, searcher.simple_query_text_select_payload)
    compare_client_results(local_client, remote_client, searcher.simple_query_image_select_vector)
    compare_client_results(local_client, remote_client, searcher.query_payload_exclude)

    for i in range(100):
        query_filter = one_random_filter_please()
        try:
            compare_client_results(
                local_client, remote_client, searcher.filter_query_text, query_filter=query_filter
            )
            compare_client_results(
                local_client, remote_client, searcher.simple_query_text_scroll, query_filter=query_filter,
            )
        except AssertionError as e:
            print(f"\nFailed with filter {query_filter}")
            raise e


def test_single_vector():
    fixture_points = generate_fixtures(num=200, vectors_sizes=text_vector_size)

    searcher = TestSimpleSearcher()

    vectors_config = models.VectorParams(
        size=text_vector_size,
        distance=models.Distance.DOT,
    )

    local_client = init_local()
    init_client(local_client, fixture_points, vectors_config=vectors_config)

    remote_client = init_remote()
    init_client(remote_client, fixture_points, vectors_config=vectors_config)

    for i in range(100):
        query_filter = one_random_filter_please()
        try:
            compare_client_results(
                local_client,
                remote_client,
                searcher.filter_query_text_single,
                query_filter=query_filter,
            )
        except AssertionError as e:
            print(f"\nFailed with filter {query_filter}")
            raise e


def test_search_with_persistence():
    import tempfile

    fixture_points = generate_fixtures()
    searcher = TestSimpleSearcher()
    with tempfile.TemporaryDirectory() as tmpdir:
        local_client = init_local(tmpdir)
        init_client(local_client, fixture_points)

        payload_update_filter = one_random_filter_please()
        local_client.set_payload(COLLECTION_NAME, {"test": f"test"}, payload_update_filter)

        del local_client
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
                    searcher.filter_query_text,
                    query_filter=query_filter,
                )
            except AssertionError as e:
                print(f"\nFailed with filter {query_filter}")
                raise e


def test_search_with_persistence_and_skipped_vectors():
    import tempfile

    fixture_points = generate_fixtures(skip_vectors=True)
    searcher = TestSimpleSearcher()
    with tempfile.TemporaryDirectory() as tmpdir:
        local_client = init_local(tmpdir)
        init_client(local_client, fixture_points)

        payload_update_filter = one_random_filter_please()
        local_client.set_payload(COLLECTION_NAME, {"test": f"test"}, payload_update_filter)

        count_before_load = local_client.count(COLLECTION_NAME)
        del local_client
        local_client_2 = init_local(tmpdir)

        count_after_load = local_client_2.count(COLLECTION_NAME)

        assert count_after_load == count_before_load

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
                    searcher.filter_query_text,
                    query_filter=query_filter,
                )
            except AssertionError as e:
                print(f"\nFailed with filter {query_filter}")
                raise e


def test_query_invalid_vector_type():
    fixture_points = generate_fixtures()

    local_client = init_local()
    init_client(local_client, fixture_points)

    remote_client = init_remote()
    init_client(remote_client, fixture_points)

    vector_invalid_type = [1, 2, 3, 4]
    with pytest.raises(ValueError):
        print(local_client.query_points(collection_name=COLLECTION_NAME, query=vector_invalid_type, using="text"))

    with pytest.raises(UnexpectedResponse):
        remote_client.query_points(collection_name=COLLECTION_NAME, query=vector_invalid_type, using="text")


def test_query_with_nan():
    fixture_points = generate_fixtures()

    local_client = init_local()
    init_client(local_client, fixture_points)

    remote_client = init_remote()
    init_client(remote_client, fixture_points)

    vector = np.random.random(text_vector_size)
    vector[4] = np.nan
    query = vector.tolist()
    with pytest.raises(AssertionError):
        local_client.query_points(COLLECTION_NAME, query=query, using="text")
    with pytest.raises(UnexpectedResponse):
        remote_client.query_points(COLLECTION_NAME, query=query, using="text")

    single_vector_config = models.VectorParams(
        size=text_vector_size, distance=models.Distance.COSINE
    )
    local_client.recreate_collection(COLLECTION_NAME, vectors_config=single_vector_config)
    remote_client.recreate_collection(COLLECTION_NAME, vectors_config=single_vector_config)
    fixture_points = generate_fixtures(vectors_sizes=text_vector_size)
    init_client(local_client, fixture_points, vectors_config=single_vector_config)
    init_client(remote_client, fixture_points, vectors_config=single_vector_config)

    with pytest.raises(AssertionError):
        print(local_client.query_points(COLLECTION_NAME, query=query))
    with pytest.raises(UnexpectedResponse):
        remote_client.query_points(COLLECTION_NAME, query=query)
