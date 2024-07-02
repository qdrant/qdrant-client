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
    sparse_text_vector_size,
    sparse_image_vector_size,
    sparse_code_vector_size,
    generate_sparse_fixtures,
    sparse_vectors_config,
    generate_multivector_fixtures,
    multi_vector_config,
)
from tests.fixtures.filters import one_random_filter_please
from tests.fixtures.points import generate_random_sparse_vector, generate_random_multivector


SECONDARY_COLLECTION_NAME = "congruence_secondary_collection"


class TestSimpleSearcher:
    __test__ = False

    def __init__(self):
        # dense query vectors
        self.dense_vector_query_text = np.random.random(text_vector_size).tolist()
        self.dense_vector_query_image = np.random.random(image_vector_size).tolist()
        self.dense_vector_query_code = np.random.random(code_vector_size).tolist()

        # sparse query vectors
        self.sparse_vector_query_text = generate_random_sparse_vector(
            sparse_text_vector_size, density=0.3
        )
        self.sparse_vector_query_image = generate_random_sparse_vector(
            sparse_image_vector_size, density=0.2
        )
        self.sparse_vector_query_code = generate_random_sparse_vector(
            sparse_code_vector_size, density=0.1
        )

        # multivector query vectors
        self.multivector_query_text = generate_random_multivector(text_vector_size, 3)
        self.multivector_query_image = generate_random_multivector(image_vector_size, 3)
        self.multivector_query_code = generate_random_multivector(code_vector_size, 3)

    def sparse_query_text(self, client: QdrantBase) -> models.QueryResponse:
        return client.query_points(
            collection_name=COLLECTION_NAME,
            query=self.sparse_vector_query_text,
            using="sparse-text",
            with_payload=True,
            limit=10,
        )

    def multivec_query_text(self, client: QdrantBase) -> models.QueryResponse:
        return client.query_points(
            collection_name=COLLECTION_NAME,
            query=self.multivector_query_text,
            using="multi-text",
            with_payload=True,
            limit=10,
        )

    def dense_query_text(self, client: QdrantBase) -> models.QueryResponse:
        return client.query_points(
            collection_name=COLLECTION_NAME,
            query=self.dense_vector_query_text,
            using="text",
            with_payload=True,
            limit=10,
        )

    def dense_query_image(self, client: QdrantBase) -> models.QueryResponse:
        return client.query_points(
            collection_name=COLLECTION_NAME,
            query=self.dense_vector_query_image,
            using="image",
            with_payload=True,
            limit=10,
        )

    def dense_query_code(self, client: QdrantBase) -> models.QueryResponse:
        return client.query_points(
            collection_name=COLLECTION_NAME,
            query=self.dense_vector_query_code,
            using="code",
            with_payload=True,
            limit=10,
        )

    def dense_query_text_offset(self, client: QdrantBase) -> models.QueryResponse:
        return client.query_points(
            collection_name=COLLECTION_NAME,
            query=self.dense_vector_query_text,
            using="text",
            with_payload=True,
            limit=10,
            offset=10,
        )

    def dense_query_text_with_vector(self, client: QdrantBase) -> models.QueryResponse:
        return client.query_points(
            collection_name=COLLECTION_NAME,
            query=self.dense_vector_query_text,
            using="text",
            with_payload=True,
            with_vectors=True,
            limit=10,
            offset=10,
        )

    def dense_query_score_threshold(self, client: QdrantBase) -> List[models.ScoredPoint]:
        res1 = client.query_points(
            collection_name=COLLECTION_NAME,
            query=self.dense_vector_query_text,
            using="text",
            with_payload=True,
            limit=10,
            score_threshold=0.9,
        ).points

        res2 = client.query_points(
            collection_name=COLLECTION_NAME,
            query=self.dense_vector_query_text,
            using="text",
            with_payload=True,
            limit=10,
            score_threshold=0.95,
        ).points

        res3 = client.query_points(
            collection_name=COLLECTION_NAME,
            query=self.dense_vector_query_text,
            using="text",
            with_payload=True,
            limit=10,
            score_threshold=0.1,
        ).points

        return res1 + res2 + res3

    def dense_query_text_select_payload(self, client: QdrantBase) -> models.QueryResponse:
        return client.query_points(
            collection_name=COLLECTION_NAME,
            query=self.dense_vector_query_text,
            using="text",
            with_payload=["text_array", "nested.id"],
            limit=10,
        )

    def dense_payload_exclude(self, client: QdrantBase) -> models.QueryResponse:
        return client.query_points(
            collection_name=COLLECTION_NAME,
            query=self.dense_vector_query_text,
            using="text",
            with_payload=models.PayloadSelectorExclude(exclude=["text_array", "nested.id"]),
            limit=10,
        )

    def dense_query_image_select_vector(self, client: QdrantBase) -> models.QueryResponse:
        return client.query_points(
            collection_name=COLLECTION_NAME,
            query=self.dense_vector_query_image,
            using="image",
            with_payload=False,
            with_vectors=["image", "code"],
            limit=10,
        )

    def filter_dense_query_text(
        self, client: QdrantBase, query_filter: models.Filter
    ) -> models.QueryResponse:
        return client.query_points(
            collection_name=COLLECTION_NAME,
            query=self.dense_vector_query_text,
            using="text",
            query_filter=query_filter,
            with_payload=True,
            limit=10,
        )

    def filter_dense_query_text_single(
        self, client: QdrantBase, query_filter: models.Filter
    ) -> models.QueryResponse:
        return client.query_points(
            collection_name=COLLECTION_NAME,
            query=self.dense_vector_query_text,
            query_filter=query_filter,
            with_payload=True,
            with_vectors=True,
            limit=10,
        )

    @classmethod
    def dense_query_text_scroll(
        cls, client: QdrantBase, query_filter: models.Filter
    ) -> models.QueryResponse:
        return client.query_points(
            collection_name=COLLECTION_NAME,
            using="text",
            query_filter=query_filter,
            with_payload=True,
            with_vectors=True,
            limit=10,
        )

    def dense_query_fusion(self, client: QdrantBase) -> models.QueryResponse:
        return client.query_points(
            collection_name=COLLECTION_NAME,
            prefetch=[
                models.Prefetch(
                    query=self.dense_vector_query_text,
                    using="text",
                )
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            with_payload=True,
            limit=10,
        )

    def deep_dense_queries_fusion(self, client: QdrantBase) -> models.QueryResponse:
        return client.query_points(
            collection_name=COLLECTION_NAME,
            prefetch=[
                models.Prefetch(
                    query=self.dense_vector_query_code,
                    using="code",
                    limit=30,
                    prefetch=[
                        models.Prefetch(
                            query=self.dense_vector_query_image,
                            using="image",
                            limit=40,
                            prefetch=[
                                models.Prefetch(
                                    query=self.dense_vector_query_text,
                                    using="text",
                                    limit=50,
                                )
                            ],
                        )
                    ],
                )
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            with_payload=True,
            limit=10,
        )

    def dense_queries_rescore(self, client: QdrantBase) -> models.QueryResponse:
        return client.query_points(
            collection_name=COLLECTION_NAME,
            prefetch=[
                models.Prefetch(
                    query=self.dense_vector_query_text,
                    using="text",
                ),
                models.Prefetch(
                    query=self.dense_vector_query_code,
                    using="code",
                ),
            ],
            query=self.dense_vector_query_image,
            using="image",
            with_payload=True,
            limit=10,
        )

    def dense_deep_queries_rescore(self, client: QdrantBase) -> models.QueryResponse:
        return client.query_points(
            collection_name=COLLECTION_NAME,
            prefetch=[
                models.Prefetch(
                    query=self.dense_vector_query_code,
                    using="code",
                    limit=30,
                    prefetch=[
                        models.Prefetch(
                            query=self.dense_vector_query_image,
                            using="image",
                            limit=40,
                            prefetch=[
                                models.Prefetch(
                                    query=self.dense_vector_query_text,
                                    using="text",
                                    limit=50,
                                )
                            ],
                        )
                    ],
                )
            ],
            query=self.dense_vector_query_image,
            using="image",
            with_payload=True,
            limit=10,
        )

    def dense_queries_prefetch_filtered(
        self, client: QdrantBase, query_filter: models.Filter
    ) -> models.QueryResponse:
        return client.query_points(
            collection_name=COLLECTION_NAME,
            prefetch=[
                models.Prefetch(
                    query=self.dense_vector_query_text,
                    using="text",
                    filter=query_filter,
                ),
                models.Prefetch(
                    query=self.dense_vector_query_code,
                    using="code",
                    filter=query_filter,
                ),
            ],
            query=self.dense_vector_query_image,
            using="image",
            with_payload=True,
            limit=10,
        )

    def dense_queries_prefetch_score_threshold(self, client: QdrantBase) -> models.QueryResponse:
        return client.query_points(
            collection_name=COLLECTION_NAME,
            prefetch=[
                models.Prefetch(
                    query=self.dense_vector_query_text, using="text", score_threshold=0.9
                ),
                models.Prefetch(
                    query=self.dense_vector_query_code,
                    using="code",
                    score_threshold=0.1,
                ),
            ],
            query=self.dense_vector_query_image,
            using="image",
            with_payload=True,
            limit=10,
        )

    def dense_queries_prefetch_parametrized(
        self, client: QdrantBase, search_params: models.SearchParams
    ) -> models.QueryResponse:
        return client.query_points(
            collection_name=COLLECTION_NAME,
            prefetch=[
                models.Prefetch(
                    query=self.dense_vector_query_text, using="text", params=search_params
                ),
            ],
            query=self.dense_vector_query_image,
            using="image",
            with_payload=True,
            limit=10,
        )

    def dense_queries_parametrized(
        self, client: QdrantBase, search_params: models.SearchParams
    ) -> models.QueryResponse:
        return client.query_points(
            collection_name=COLLECTION_NAME,
            query=self.dense_vector_query_image,
            using="image",
            limit=10,
            search_params=search_params,
        )

    def dense_queries_orderby(self, client: QdrantBase) -> models.QueryResponse:
        return client.query_points(
            collection_name=COLLECTION_NAME,
            prefetch=[
                models.Prefetch(
                    query=self.dense_vector_query_text,
                    using="text",
                ),
                models.Prefetch(
                    query=self.dense_vector_query_code,
                    using="code",
                ),
            ],
            query=models.OrderByQuery(
                order_by="rand_digit",
            ),
            with_payload=True,
            limit=10,
        )

    def deep_dense_queries_orderby(self, client: QdrantBase) -> models.QueryResponse:
        return client.query_points(
            collection_name=COLLECTION_NAME,
            prefetch=[
                models.Prefetch(
                    query=self.dense_vector_query_code,
                    using="code",
                    limit=30,
                    prefetch=[
                        models.Prefetch(
                            query=self.dense_vector_query_image,
                            using="image",
                            limit=40,
                            prefetch=[
                                models.Prefetch(
                                    query=self.dense_vector_query_text,
                                    using="text",
                                    limit=50,
                                )
                            ],
                        )
                    ],
                )
            ],
            query=models.OrderByQuery(
                order_by="rand_digit",
            ),
            with_payload=True,
            limit=10,
        )

    @classmethod
    def dense_recommend_image(cls, client: QdrantBase) -> models.QueryResponse:
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
    def dense_many_recommend(cls, client: QdrantBase) -> models.QueryResponse:
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
    def dense_discovery_image(cls, client: QdrantBase) -> models.QueryResponse:
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
    def dense_many_discover(cls, client: QdrantBase) -> models.QueryResponse:
        return client.query_points(
            collection_name=COLLECTION_NAME,
            query=models.DiscoverQuery(
                discover=models.DiscoverInput(
                    target=10,
                    context=[
                        models.ContextPair(positive=11, negative=19),
                        models.ContextPair(positive=12, negative=20),
                    ],
                )
            ),
            with_payload=True,
            limit=10,
            using="image",
        )

    @classmethod
    def dense_context_image(cls, client: QdrantBase, limit: int) -> models.QueryResponse:
        return client.query_points(
            collection_name=COLLECTION_NAME,
            query=models.ContextQuery(context=models.ContextPair(positive=11, negative=19)),
            with_payload=True,
            limit=limit,
            using="image",
        )

    def dense_query_lookup_from(
        self, client: QdrantBase, lookup_from: models.LookupLocation
    ) -> models.QueryResponse:
        return client.query_points(
            collection_name=COLLECTION_NAME,
            query=models.RecommendQuery(
                recommend=models.RecommendInput(positive=[1, 2], negative=[3, 4])
            ),
            using="text",
            limit=10,
            lookup_from=lookup_from,
        )

    @classmethod
    def no_query_no_prefetch(cls, client: QdrantBase) -> models.QueryResponse:
        return client.query_points(collection_name=COLLECTION_NAME, limit=10)


# ---- TESTS  ---- #


def test_dense_query_lookup_from_another_collection():
    fixture_points = generate_fixtures(10)

    secondary_collection_points = generate_fixtures(10)

    searcher = TestSimpleSearcher()

    local_client = init_local()
    init_client(local_client, fixture_points)
    init_client(local_client, secondary_collection_points, SECONDARY_COLLECTION_NAME)

    remote_client = init_remote()
    init_client(remote_client, fixture_points)
    init_client(remote_client, secondary_collection_points, SECONDARY_COLLECTION_NAME)

    compare_client_results(
        local_client,
        remote_client,
        searcher.dense_query_lookup_from,
        lookup_from=models.LookupLocation(collection=SECONDARY_COLLECTION_NAME, vector="text"),
    )


def test_dense_query_lookup_from_negative():
    fixture_points = generate_fixtures()

    secondary_collection_points = generate_fixtures(10)

    local_client = init_local()
    init_client(local_client, fixture_points)
    init_client(local_client, secondary_collection_points, SECONDARY_COLLECTION_NAME)

    remote_client = init_remote()
    init_client(remote_client, fixture_points)
    init_client(remote_client, secondary_collection_points, SECONDARY_COLLECTION_NAME)

    lookup_from = models.LookupLocation(collection="i-do-not-exist", vector="text")
    with pytest.raises(ValueError, match="Collection i-do-not-exist not found"):
        local_client.query_points(
            collection_name=COLLECTION_NAME,
            query=models.RecommendQuery(
                recommend=models.RecommendInput(positive=[1, 2], negative=[3, 4])
            ),
            using="text",
            limit=10,
            lookup_from=lookup_from,
        )
    with pytest.raises(UnexpectedResponse, match="Not found: Collection"):
        remote_client.query_points(
            collection_name=COLLECTION_NAME,
            query=models.RecommendQuery(
                recommend=models.RecommendInput(positive=[1, 2], negative=[3, 4])
            ),
            using="text",
            limit=10,
            lookup_from=lookup_from,
        )

    lookup_from = models.LookupLocation(
        collection=SECONDARY_COLLECTION_NAME, vector="i-do-not-exist"
    )
    with pytest.raises(ValueError, match="Vector i-do-not-exist not found"):
        local_client.query_points(
            collection_name=COLLECTION_NAME,
            query=models.RecommendQuery(
                recommend=models.RecommendInput(positive=[1, 2], negative=[3, 4])
            ),
            using="text",
            limit=10,
            lookup_from=lookup_from,
        )
    with pytest.raises(UnexpectedResponse, match="Not existing vector name error"):
        remote_client.query_points(
            collection_name=COLLECTION_NAME,
            query=models.RecommendQuery(
                recommend=models.RecommendInput(positive=[1, 2], negative=[3, 4])
            ),
            using="text",
            limit=10,
            lookup_from=lookup_from,
        )


def test_no_query_no_prefetch():
    fixture_points = generate_fixtures()

    searcher = TestSimpleSearcher()

    local_client = init_local()
    init_client(local_client, fixture_points)

    remote_client = init_remote()
    init_client(remote_client, fixture_points)

    compare_client_results(local_client, remote_client, searcher.no_query_no_prefetch)


def test_dense_query_filtered_prefetch():
    fixture_points = generate_fixtures()

    searcher = TestSimpleSearcher()

    local_client = init_local()
    init_client(local_client, fixture_points)

    remote_client = init_remote()
    init_client(remote_client, fixture_points)

    for i in range(100):
        query_filter = one_random_filter_please()
        try:
            compare_client_results(
                local_client,
                remote_client,
                searcher.dense_queries_prefetch_filtered,
                query_filter=query_filter,
            )
        except AssertionError as e:
            print(f"\nAttempt {i} failed with filter {query_filter}")
            raise e


def test_dense_query_prefetch_score_threshold():
    fixture_points = generate_fixtures()

    searcher = TestSimpleSearcher()

    local_client = init_local()
    init_client(local_client, fixture_points)

    remote_client = init_remote()
    init_client(remote_client, fixture_points)

    compare_client_results(
        local_client, remote_client, searcher.dense_queries_prefetch_score_threshold
    )


def test_dense_query_prefetch_parametrized():
    fixture_points = generate_fixtures()

    searcher = TestSimpleSearcher()

    local_client = init_local()
    init_client(local_client, fixture_points)

    remote_client = init_remote()
    init_client(remote_client, fixture_points)

    compare_client_results(
        local_client,
        remote_client,
        searcher.dense_queries_prefetch_parametrized,
        search_params={"exact": True},
    )
    compare_client_results(
        local_client,
        remote_client,
        searcher.dense_queries_prefetch_parametrized,
        search_params={"hnsw_ef": 128},
    )
    compare_client_results(
        local_client,
        remote_client,
        searcher.dense_queries_prefetch_parametrized,
        search_params={"indexed_only": True},
    )
    compare_client_results(
        local_client,
        remote_client,
        searcher.dense_queries_prefetch_parametrized,
        search_params={"quantization": {"ignore": True, "rescore": True, "oversampling": 2.0}},
    )


def test_dense_query_parametrized():
    fixture_points = generate_fixtures()

    searcher = TestSimpleSearcher()

    local_client = init_local()
    init_client(local_client, fixture_points)

    remote_client = init_remote()
    init_client(remote_client, fixture_points)

    compare_client_results(
        local_client,
        remote_client,
        searcher.dense_queries_parametrized,
        search_params={"exact": True},
    )
    compare_client_results(
        local_client,
        remote_client,
        searcher.dense_queries_parametrized,
        search_params={
            "hnsw_ef": 128,
            "indexed_only": True,
            "quantization": {"ignore": True, "rescore": True, "oversampling": 2.0},
        },
    )


def test_sparse_query():
    fixture_points = generate_sparse_fixtures()

    searcher = TestSimpleSearcher()

    local_client = init_local()
    init_client(local_client, fixture_points, sparse_vectors_config=sparse_vectors_config)

    remote_client = init_remote()
    init_client(remote_client, fixture_points, sparse_vectors_config=sparse_vectors_config)

    compare_client_results(local_client, remote_client, searcher.sparse_query_text)


def test_multivec_query():
    fixture_points = generate_multivector_fixtures()

    searcher = TestSimpleSearcher()

    local_client = init_local()
    init_client(local_client, fixture_points, vectors_config=multi_vector_config)

    remote_client = init_remote()
    init_client(remote_client, fixture_points, vectors_config=multi_vector_config)

    compare_client_results(local_client, remote_client, searcher.multivec_query_text)


def test_dense_query():
    fixture_points = generate_fixtures()

    searcher = TestSimpleSearcher()

    local_client = init_local()
    init_client(local_client, fixture_points)

    remote_client = init_remote()
    init_client(remote_client, fixture_points)

    compare_client_results(local_client, remote_client, searcher.dense_query_text)
    compare_client_results(local_client, remote_client, searcher.dense_query_image)
    compare_client_results(local_client, remote_client, searcher.dense_query_code)
    compare_client_results(local_client, remote_client, searcher.dense_query_text_offset)
    compare_client_results(local_client, remote_client, searcher.dense_query_text_with_vector)
    compare_client_results(local_client, remote_client, searcher.dense_query_score_threshold)
    compare_client_results(local_client, remote_client, searcher.dense_query_text_select_payload)
    compare_client_results(local_client, remote_client, searcher.dense_query_image_select_vector)
    compare_client_results(local_client, remote_client, searcher.dense_payload_exclude)

    for i in range(100):
        query_filter = one_random_filter_please()
        try:
            compare_client_results(
                local_client,
                remote_client,
                searcher.filter_dense_query_text,
                query_filter=query_filter,
            )
            compare_client_results(
                local_client,
                remote_client,
                searcher.dense_query_text_scroll,
                query_filter=query_filter,
            )
        except AssertionError as e:
            print(f"\nFailed with filter {query_filter}")
            raise e


def test_dense_query_orderby():
    fixture_points = generate_fixtures(200)

    searcher = TestSimpleSearcher()

    local_client = init_local()
    init_client(local_client, fixture_points)

    remote_client = init_remote()
    init_client(remote_client, fixture_points)

    remote_client.create_payload_index(
        COLLECTION_NAME, "rand_digit", models.PayloadSchemaType.INTEGER, wait=True
    )

    compare_client_results(local_client, remote_client, searcher.dense_queries_orderby)
    compare_client_results(local_client, remote_client, searcher.deep_dense_queries_orderby)


def test_dense_query_recommend():
    fixture_points = generate_fixtures()

    searcher = TestSimpleSearcher()

    local_client = init_local()
    init_client(local_client, fixture_points)

    remote_client = init_remote()
    init_client(remote_client, fixture_points)

    compare_client_results(local_client, remote_client, searcher.dense_recommend_image)
    compare_client_results(local_client, remote_client, searcher.dense_many_recommend)


def test_dense_query_rescore():
    fixture_points = generate_fixtures()

    searcher = TestSimpleSearcher()

    local_client = init_local()
    init_client(local_client, fixture_points)

    remote_client = init_remote()
    init_client(remote_client, fixture_points)

    compare_client_results(local_client, remote_client, searcher.dense_queries_rescore)
    compare_client_results(local_client, remote_client, searcher.dense_deep_queries_rescore)


def test_dense_query_fusion():
    fixture_points = generate_fixtures()

    searcher = TestSimpleSearcher()

    local_client = init_local()
    init_client(local_client, fixture_points)

    remote_client = init_remote()
    init_client(remote_client, fixture_points)

    compare_client_results(local_client, remote_client, searcher.dense_query_fusion)
    compare_client_results(local_client, remote_client, searcher.deep_dense_queries_fusion)


def test_dense_query_discovery_context():
    n_vectors = 250
    fixture_points = generate_fixtures(n_vectors)

    searcher = TestSimpleSearcher()

    local_client = init_local()
    init_client(local_client, fixture_points)

    remote_client = init_remote()
    init_client(remote_client, fixture_points)

    compare_client_results(local_client, remote_client, searcher.dense_discovery_image)
    compare_client_results(local_client, remote_client, searcher.dense_many_discover)
    compare_client_results(
        local_client,
        remote_client,
        searcher.dense_context_image,
        is_context_search=True,
        limit=n_vectors,
    )


def test_simple_opt_vectors_query():
    fixture_points = generate_fixtures(skip_vectors=True)

    searcher = TestSimpleSearcher()

    local_client = init_local()
    init_client(local_client, fixture_points)

    remote_client = init_remote()
    init_client(remote_client, fixture_points)

    compare_client_results(local_client, remote_client, searcher.dense_query_text)
    compare_client_results(local_client, remote_client, searcher.dense_query_image)
    compare_client_results(local_client, remote_client, searcher.dense_query_code)
    compare_client_results(local_client, remote_client, searcher.dense_query_text_offset)
    compare_client_results(local_client, remote_client, searcher.dense_query_text_with_vector)
    compare_client_results(local_client, remote_client, searcher.dense_query_score_threshold)
    compare_client_results(local_client, remote_client, searcher.dense_query_text_select_payload)
    compare_client_results(local_client, remote_client, searcher.dense_query_image_select_vector)
    compare_client_results(local_client, remote_client, searcher.dense_payload_exclude)

    for i in range(100):
        query_filter = one_random_filter_please()
        try:
            compare_client_results(
                local_client,
                remote_client,
                searcher.filter_dense_query_text,
                query_filter=query_filter,
            )
            compare_client_results(
                local_client,
                remote_client,
                searcher.dense_query_text_scroll,
                query_filter=query_filter,
            )
        except AssertionError as e:
            print(f"\nFailed with filter {query_filter}")
            raise e


def test_single_dense_vector():
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
                searcher.filter_dense_query_text_single,
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
                    searcher.filter_dense_query_text,
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
                    searcher.filter_dense_query_text,
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
        local_client.query_points(
            collection_name=COLLECTION_NAME, query=vector_invalid_type, using="text"
        )

    with pytest.raises(UnexpectedResponse):
        remote_client.query_points(
            collection_name=COLLECTION_NAME, query=vector_invalid_type, using="text"
        )


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
