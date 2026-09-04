from typing import Callable, Any

from grpc import RpcError
import numpy as np
import pytest

from qdrant_client import QdrantClient
from qdrant_client.client_base import QdrantBase
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.http.models import models, GroupsResult
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
    generate_sparse_fixtures,
    sparse_vectors_config,
    generate_multivector_fixtures,
    multi_vector_config,
)
from tests.fixtures.expressions import one_random_expression_please
from tests.fixtures.filters import one_random_filter_please
from tests.fixtures.points import (
    generate_random_sparse_vector,
    generate_random_multivector,
)
from tests.utils import read_version

SECONDARY_COLLECTION_NAME = "congruence_secondary_collection"


class TestSimpleSearcher:
    __test__ = False

    def __init__(self):
        # group by
        self.group_by = "city.geo"
        self.group_size = 3
        self.limit = 2  # number of groups

        # dense query vectors
        self.dense_vector_query_text = np.random.random(text_vector_size).tolist()
        self.dense_vector_query_text_bis = self.dense_vector_query_text
        self.dense_vector_query_text_bis[0] += 42.0  # slightly different vector
        self.dense_vector_query_image = np.random.random(image_vector_size).tolist()
        self.dense_vector_query_code = np.random.random(code_vector_size).tolist()

        # sparse query vectors
        self.sparse_vector_query_text = generate_random_sparse_vector(
            sparse_text_vector_size, density=0.3
        )
        self.sparse_vector_query_image = generate_random_sparse_vector(
            sparse_image_vector_size, density=0.2
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

    def multivec_query_code(self, client: QdrantBase) -> models.QueryResponse:
        return client.query_points(
            collection_name=COLLECTION_NAME,
            query=self.multivector_query_code,
            using="multi-code",
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

    def dense_query_text_np_array(self, client: QdrantBase) -> models.QueryResponse:
        return client.query_points(
            collection_name=COLLECTION_NAME,
            query=np.array(self.dense_vector_query_text),
            using="text",
            with_payload=True,
            limit=10,
        )

    @classmethod
    def dense_query_text_by_id(cls, client: QdrantBase) -> models.QueryResponse:
        return client.query_points(
            collection_name=COLLECTION_NAME,
            query=1,
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

    def dense_query_score_threshold(self, client: QdrantBase) -> list[models.ScoredPoint]:
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

    def dense_query_group(self, client: QdrantBase) -> GroupsResult:
        return client.query_points_groups(
            collection_name=COLLECTION_NAME,
            query=self.dense_vector_query_text,
            using="text",
            group_by=self.group_by,
            group_size=self.group_size,
            limit=self.limit,
            with_payload=models.PayloadSelectorInclude(include=[self.group_by]),
        )

    def dense_query_group_with_lookup(self, client: QdrantBase) -> GroupsResult:
        return client.query_points_groups(
            collection_name=COLLECTION_NAME,
            query=self.dense_vector_query_text,
            using="text",
            group_by=self.group_by,
            group_size=self.group_size,
            limit=self.limit,
            with_payload=models.PayloadSelectorInclude(include=[self.group_by]),
            with_lookup=SECONDARY_COLLECTION_NAME,
        )

    def dense_query_group_with_lookup_from(self, client: QdrantBase) -> GroupsResult:
        return client.query_points_groups(
            collection_name=COLLECTION_NAME,
            query=1,
            using="text",
            group_by=self.group_by,
            group_size=self.group_size,
            limit=self.limit,
            with_payload=models.PayloadSelectorInclude(include=[self.group_by]),
            lookup_from=models.LookupLocation(collection=SECONDARY_COLLECTION_NAME, vector="text"),
        )

    def filter_dense_query_group(
        self, client: QdrantBase, query_filter: models.Filter
    ) -> GroupsResult:
        return client.query_points_groups(
            collection_name=COLLECTION_NAME,
            query=self.dense_vector_query_text,
            query_filter=query_filter,
            using="text",
            group_by=self.group_by,
            group_size=self.group_size,
            limit=self.limit,
            with_payload=True,
        )

    def dense_queries_rescore_group(self, client: QdrantBase) -> GroupsResult:
        return client.query_points_groups(
            collection_name=COLLECTION_NAME,
            prefetch=[
                models.Prefetch(
                    query=self.dense_vector_query_text,
                    using="text",
                    limit=20,
                ),
            ],
            # slightly different vector for rescoring because group_by is not super accurate with rescoring
            query=self.dense_vector_query_text_bis,
            using="text",
            with_payload=models.PayloadSelectorInclude(include=[self.group_by]),
            group_by=self.group_by,
            group_size=self.group_size,
            limit=self.limit,
        )

    def dense_queries_rescore_group_single_prefetch(self, client: QdrantBase) -> GroupsResult:
        return client.query_points_groups(
            collection_name=COLLECTION_NAME,
            prefetch=models.Prefetch(
                query=self.dense_vector_query_text,
                prefetch=[
                    models.Prefetch(query=self.dense_vector_query_text, using="text", limit=30)
                ],
                using="text",
                limit=20,
            ),
            # slightly different vector for rescoring because group_by is not super accurate with rescoring
            query=self.dense_vector_query_text_bis,
            using="text",
            with_payload=models.PayloadSelectorInclude(include=[self.group_by]),
            group_by=self.group_by,
            group_size=self.group_size,
            limit=self.limit,
        )

    def dense_query_lookup_from_group(
        self, client: QdrantBase, lookup_from: models.LookupLocation
    ) -> GroupsResult:
        return client.query_points_groups(
            collection_name=COLLECTION_NAME,
            query=models.RecommendQuery(
                recommend=models.RecommendInput(positive=[1, 2], negative=[3, 4])
            ),
            using="text",
            lookup_from=lookup_from,
            group_by=self.group_by,
            group_size=self.group_size,
            limit=self.limit,
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
    def filter_query_scroll(
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

    def dense_query_rrf(self, client: QdrantBase) -> models.QueryResponse:
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

    def dense_query_parametrized_rrf(self, client: QdrantBase) -> models.QueryResponse:
        return client.query_points(
            collection_name=COLLECTION_NAME,
            prefetch=[
                models.Prefetch(
                    query=self.dense_vector_query_text,
                    using="text",
                )
            ],
            query=models.RrfQuery(rrf=models.Rrf(k=10)),
            with_payload=True,
            limit=10,
        )

    def dense_query_rrf_plain_prefetch(self, client: QdrantBase) -> models.QueryResponse:
        # dense_query_rrf has a list of prefetches, here we have just a prefetch
        return client.query_points(
            collection_name=COLLECTION_NAME,
            prefetch=models.Prefetch(
                query=self.dense_vector_query_text,
                using="text",
            ),
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            with_payload=True,
            limit=10,
        )

    def dense_query_dbsf(self, client: QdrantBase) -> models.QueryResponse:
        return client.query_points(
            collection_name=COLLECTION_NAME,
            prefetch=[
                models.Prefetch(
                    query=self.dense_vector_query_text,
                    using="text",
                ),
                models.Prefetch(query=self.dense_vector_query_code, using="code"),
            ],
            query=models.FusionQuery(fusion=models.Fusion.DBSF),
            with_payload=True,
            limit=10,
        )

    def dense_query_rrf_score_threshold(self, client: QdrantBase) -> list[models.ScoredPoint]:
        return client.query_points(
            collection_name=COLLECTION_NAME,
            prefetch=[
                models.Prefetch(
                    query=self.dense_vector_query_text,
                    using="text",
                )
            ],
            query=models.RrfQuery(rrf=models.Rrf(k=1)),
            with_payload=True,
            limit=10,
            score_threshold=0.25,  # should return 3 results: 1.0, 0.5, 0.3(3)
        ).points

    def dense_query_formula_score_threshold(self, client: QdrantBase) -> list[models.ScoredPoint]:
        return client.query_points(
            collection_name=COLLECTION_NAME,
            prefetch=[
                models.Prefetch(
                    query=self.dense_vector_query_text,
                    using="text",
                )
            ],
            query=models.FormulaQuery(formula=models.MultExpression(mult=["$score", 1.0])),
            with_payload=True,
            limit=10,
            score_threshold=1.0,
        ).points

    def deep_dense_queries_rrf(self, client: QdrantBase) -> models.QueryResponse:
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

    def deep_dense_queries_dbsf(self, client: QdrantBase) -> models.QueryResponse:
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
            query=models.FusionQuery(fusion=models.Fusion.DBSF),
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

    def dense_queries_root_filtered_prefetch(
        self, client: QdrantBase, query_filter: models.Filter
    ) -> models.QueryResponse:
        # prefetch limits are deliberately small: the filter has to be applied together with
        # each prefetch, not to the results of the fusion
        return client.query_points(
            collection_name=COLLECTION_NAME,
            prefetch=[
                models.Prefetch(
                    query=self.dense_vector_query_text,
                    using="text",
                    limit=5,
                ),
                models.Prefetch(
                    query=self.dense_vector_query_code,
                    using="code",
                    limit=5,
                ),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            query_filter=query_filter,
            with_payload=True,
            limit=10,
        )

    def dense_queries_root_filtered_prefetch_formula(
        self, client: QdrantBase, query_filter: models.Filter
    ) -> models.QueryResponse:
        return client.query_points(
            collection_name=COLLECTION_NAME,
            prefetch=[
                models.Prefetch(
                    query=self.dense_vector_query_text,
                    using="text",
                    limit=5,
                ),
            ],
            query=models.FormulaQuery(formula=models.MultExpression(mult=["$score", 1.0])),
            query_filter=query_filter,
            with_payload=True,
            limit=10,
        )

    def dense_queries_root_and_prefetch_filtered(
        self,
        client: QdrantBase,
        query_filter: models.Filter,
        prefetch_filter: models.Filter,
    ) -> models.QueryResponse:
        # the root filter is merged into the one of the prefetch clause by clause
        return client.query_points(
            collection_name=COLLECTION_NAME,
            prefetch=[
                models.Prefetch(
                    query=self.dense_vector_query_text,
                    using="text",
                    filter=prefetch_filter,
                    limit=5,
                ),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            query_filter=query_filter,
            with_payload=True,
            limit=10,
        )

    def deep_dense_queries_root_filtered_prefetch(
        self, client: QdrantBase, query_filter: models.Filter
    ) -> models.QueryResponse:
        # the root filter has to reach the leaves of the prefetch tree
        return client.query_points(
            collection_name=COLLECTION_NAME,
            prefetch=[
                models.Prefetch(
                    query=self.dense_vector_query_code,
                    using="code",
                    limit=5,
                    prefetch=[
                        models.Prefetch(
                            query=self.dense_vector_query_text,
                            using="text",
                            limit=10,
                        )
                    ],
                )
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            query_filter=query_filter,
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

    @classmethod
    def query_scroll_offset(cls, client: QdrantBase) -> models.QueryResponse:
        return client.query_points(
            collection_name=COLLECTION_NAME,
            with_payload=True,
            limit=10,
            offset=10,
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

    def dense_queries_prefetch_offset(self, client: QdrantBase) -> models.QueryResponse:
        return client.query_points(
            collection_name=COLLECTION_NAME,
            prefetch=[
                models.Prefetch(query=self.dense_vector_query_code, using="code", limit=30),
                models.Prefetch(query=self.dense_vector_query_text, using="text", limit=30),
            ],
            query=models.NearestQuery(nearest=self.dense_vector_query_image),
            using="image",
            with_payload=True,
            offset=10,
            limit=10,
        )

    def dense_query_prefetch_by_id(self, client: QdrantBase) -> models.QueryResponse:
        # a bare point id is a valid prefetch query, just like a bare vector
        return client.query_points(
            collection_name=COLLECTION_NAME,
            prefetch=models.Prefetch(query=1, using="text", limit=10),
            query=self.dense_vector_query_text,
            using="text",
            with_payload=True,
            limit=10,
        )

    def dense_query_nested_prefetch_by_id(self, client: QdrantBase) -> models.QueryResponse:
        # a bare point id has to be resolved at every level of the prefetch tree
        return client.query_points(
            collection_name=COLLECTION_NAME,
            prefetch=models.Prefetch(
                prefetch=models.Prefetch(query=2, using="text", limit=10),
                query=1,
                using="text",
                limit=5,
            ),
            query=self.dense_vector_query_text,
            using="text",
            with_payload=True,
            limit=10,
        )

    def dense_query_prefetch_by_id_lookup_from(
        self, client: QdrantBase, lookup_from: models.LookupLocation
    ) -> models.QueryResponse:
        return client.query_points(
            collection_name=COLLECTION_NAME,
            prefetch=models.Prefetch(query=1, using="text", limit=10, lookup_from=lookup_from),
            query=self.dense_vector_query_text,
            using="text",
            with_payload=True,
            limit=10,
        )

    def dense_query_text_nested_prefetch(self, client: QdrantBase) -> models.QueryResponse:
        return client.query_points(
            collection_name=COLLECTION_NAME,
            prefetch=[
                models.Prefetch(
                    prefetch=models.Prefetch(query=self.dense_vector_query_text, using="text"),
                    query=self.dense_vector_query_text,
                    using="text",
                ),
            ],
            query=self.dense_vector_query_text,
            using="text",
        )

    def dense_query_text_nested_single_prefetch(self, client: QdrantBase) -> models.QueryResponse:
        # a single prefetch object instead of a list of one: the nested prefetch must only feed
        # the prefetch it belongs to, it is not an extra source of the root query
        return client.query_points(
            collection_name=COLLECTION_NAME,
            prefetch=models.Prefetch(
                prefetch=models.Prefetch(
                    query=self.dense_vector_query_text,
                    using="text",
                    limit=10,
                ),
                query=self.dense_vector_query_code,
                using="code",
                limit=5,
            ),
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            with_payload=True,
            limit=10,
        )

    def dense_query_text_nested_single_prefetch_rescore(
        self, client: QdrantBase
    ) -> models.QueryResponse:
        return client.query_points(
            collection_name=COLLECTION_NAME,
            prefetch=models.Prefetch(
                prefetch=models.Prefetch(
                    query=self.dense_vector_query_text,
                    using="text",
                    limit=10,
                ),
                query=self.dense_vector_query_code,
                using="code",
                limit=5,
            ),
            query=self.dense_vector_query_text,
            using="text",
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

    @classmethod
    def dense_query_lookup_from(
        cls, client: QdrantBase, lookup_from: models.LookupLocation
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

    @classmethod
    def random_query(cls, client: QdrantBase) -> models.QueryResponse:
        result = client.query_points(
            collection_name=COLLECTION_NAME,
            query=models.SampleQuery(sample=models.Sample.RANDOM),
            limit=100,
        )

        # sort to be able to compare
        result.points.sort(key=lambda point: point.id)

        return result

    @classmethod
    def random_query_offset(cls, client: QdrantBase) -> models.QueryResponse:
        result = client.query_points(
            collection_name=COLLECTION_NAME,
            query=models.SampleQuery(sample=models.Sample.RANDOM),
            limit=100,
            offset=10,
        )  # make sure that offset does not affect the number of points in the result

        # sort to be able to compare
        result.points.sort(key=lambda point: point.id)

        return result

    @staticmethod
    def score_boosting(
        client: QdrantBase, formula: models.FormulaQuery, point_id: int
    ) -> models.QueryResponse | str:
        def comparable_error(exception: Exception):
            non_finite_message = "produced a non-finite number"
            too_long_non_finite_message_end = "...'"
            math_domain_error_message = "math domain error"
            unexpected_type_message = "in the payload and/or in the formula defaults"

            if (
                non_finite_message in str(exception)
                or math_domain_error_message in str(exception)  # local mode
                or str(exception).endswith(
                    too_long_non_finite_message_end
                )  # remote abrupt traceback
            ):
                # 0^-5 causes non-finite in core, math domain error in local mode
                return non_finite_message
            elif unexpected_type_message in str(exception):
                return unexpected_type_message
            raise exception

        prefetch = models.Prefetch(
            filter=models.Filter(must=[models.HasIdCondition(has_id=[point_id])]),
            limit=1,
            using="text",
        )

        try:
            result = client.query_points(
                collection_name=COLLECTION_NAME,
                prefetch=prefetch,
                query=formula,
                limit=1,
            )
        except ValueError as e:  # local mode error
            return comparable_error(e)
        except UnexpectedResponse as e:  # rest error
            return comparable_error(e)
        except RpcError as e:  # grpc error
            return comparable_error(e)

        return result

    def default_mmr_query(self, client: QdrantBase) -> models.QueryResponse:
        return client.query_points(
            collection_name=COLLECTION_NAME,
            query=models.NearestQuery(
                nearest=self.dense_vector_query_text,
                mmr=models.Mmr(),
            ),
            using="text",
            limit=10,
        )

    def mmr_query_parametrized(self, client: QdrantBase) -> models.QueryResponse:
        return client.query_points(
            collection_name=COLLECTION_NAME,
            query=models.NearestQuery(
                nearest=self.dense_vector_query_text,
                mmr=models.Mmr(diversity=0.3, candidates_limit=30),
            ),
            using="text",
            limit=10,
        )

    def mmr_query_parametrized_score_threshold(self, client: QdrantBase) -> models.QueryResponse:
        return client.query_points(
            collection_name=COLLECTION_NAME,
            query=models.NearestQuery(
                nearest=self.dense_vector_query_text,
                mmr=models.Mmr(diversity=0.3, candidates_limit=30),
            ),
            using="text",
            score_threshold=0.9,
            limit=10,
        )

    def mmr_query_parametrized_dot(self, client: QdrantBase) -> models.QueryResponse:
        return client.query_points(
            collection_name=COLLECTION_NAME,
            query=models.NearestQuery(
                nearest=self.dense_vector_query_image,
                mmr=models.Mmr(diversity=0.3, candidates_limit=30),
            ),
            using="image",
            limit=10,
        )

    def mmr_query_parametrized_dot_score_threshold(
        self, client: QdrantBase
    ) -> models.QueryResponse:
        result = client.query_points(
            collection_name=COLLECTION_NAME,
            query=models.NearestQuery(
                nearest=self.dense_vector_query_image,
                mmr=models.Mmr(diversity=0.3, candidates_limit=30),
            ),
            using="image",
            score_threshold=30.0,
            limit=10,
        )
        return result

    def mmr_query_parametrized_euclid(self, client: QdrantBase) -> models.QueryResponse:
        return client.query_points(
            collection_name=COLLECTION_NAME,
            query=models.NearestQuery(
                nearest=self.dense_vector_query_code,
                mmr=models.Mmr(diversity=0.3, candidates_limit=30),
            ),
            using="code",
            limit=10,
        )

    def mmr_query_parametrized_euclid_score_threshold(
        self, client: QdrantBase
    ) -> models.QueryResponse:
        return client.query_points(
            collection_name=COLLECTION_NAME,
            query=models.NearestQuery(
                nearest=self.dense_vector_query_code,
                mmr=models.Mmr(diversity=0.3, candidates_limit=30),
            ),
            score_threshold=3.0,
            using="code",
            limit=10,
        )

    def relevance_feedback_query(self, client: QdrantBase) -> models.QueryResponse:
        return client.query_points(
            collection_name=COLLECTION_NAME,
            query=models.RelevanceFeedbackQuery(
                relevance_feedback=models.RelevanceFeedbackInput(
                    target=self.dense_vector_query_text,
                    feedback=[
                        models.FeedbackItem(example=1, score=0.9),
                        models.FeedbackItem(example=2, score=0.8),
                        models.FeedbackItem(example=3, score=0.7),
                        models.FeedbackItem(example=4, score=0.6),
                    ],
                    strategy=models.NaiveFeedbackStrategy(
                        naive=models.NaiveFeedbackStrategyParams(a=0.5, b=1.0, c=0.7)
                    ),
                )
            ),
            using="text",
            with_payload=True,
            limit=10,
        )

    def relevance_feedback_query_euclid(self, client: QdrantBase) -> models.QueryResponse:
        # `code` is a euclid vector. Feedback scores come from the internal core
        # distance, which is oriented higher-is-better whatever the metric, so the
        # results still have to come back best-first instead of farthest-first.
        return client.query_points(
            collection_name=COLLECTION_NAME,
            query=models.RelevanceFeedbackQuery(
                relevance_feedback=models.RelevanceFeedbackInput(
                    target=self.dense_vector_query_code,
                    feedback=[
                        models.FeedbackItem(example=1, score=0.9),
                        models.FeedbackItem(example=2, score=0.8),
                        models.FeedbackItem(example=3, score=0.7),
                        models.FeedbackItem(example=4, score=0.6),
                    ],
                    strategy=models.NaiveFeedbackStrategy(
                        naive=models.NaiveFeedbackStrategyParams(a=0.5, b=1.0, c=0.7)
                    ),
                )
            ),
            using="code",
            with_payload=True,
            limit=10,
        )


def group_by_keys():
    return ["maybe", "rand_digit", "two_words", "city.name", "maybe_null", "id"]


def init_clients(fixture_points, **kwargs) -> tuple[QdrantClient, QdrantClient, QdrantClient]:
    local_client = init_local()
    http_client = init_remote()
    grpc_client = init_remote(prefer_grpc=True)

    init_client(local_client, fixture_points, **kwargs)
    init_client(http_client, fixture_points, **kwargs)

    return local_client, http_client, grpc_client


def compare_clients_results(
    local_client: QdrantClient,
    http_client: QdrantClient,
    grpc_client: QdrantClient,
    foo: Callable[[QdrantBase, Any], Any],
    **kwargs: Any,
):
    compare_client_results(local_client, http_client, foo, **kwargs)
    compare_client_results(http_client, grpc_client, foo, **kwargs)


# ---- TESTS  ---- #


def test_dense_query_lookup_from_another_collection():
    fixture_points = generate_fixtures(10)

    secondary_collection_points = generate_fixtures(10)

    searcher = TestSimpleSearcher()

    local_client, http_client, grpc_client = init_clients(fixture_points)

    init_client(local_client, secondary_collection_points, SECONDARY_COLLECTION_NAME)
    init_client(http_client, secondary_collection_points, SECONDARY_COLLECTION_NAME)

    compare_clients_results(
        local_client,
        http_client,
        grpc_client,
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
    major, minor, patch, dev = read_version()
    if not dev and None not in (major, minor, patch) and (major, minor, patch) < (1, 10, 1):
        pytest.skip("Works as of version 1.10.1")

    fixture_points = generate_fixtures()

    searcher = TestSimpleSearcher()

    local_client, http_client, grpc_client = init_clients(fixture_points)

    compare_clients_results(local_client, http_client, grpc_client, searcher.no_query_no_prefetch)
    compare_clients_results(http_client, grpc_client, grpc_client, searcher.no_query_no_prefetch)

    compare_clients_results(local_client, http_client, grpc_client, searcher.query_scroll_offset)
    compare_clients_results(http_client, grpc_client, grpc_client, searcher.query_scroll_offset)


def test_dense_query_prefetch_by_id():
    fixture_points = generate_fixtures()

    searcher = TestSimpleSearcher()

    local_client, http_client, grpc_client = init_clients(fixture_points)

    for query in (
        searcher.dense_query_prefetch_by_id,
        searcher.dense_query_nested_prefetch_by_id,
    ):
        compare_clients_results(local_client, http_client, grpc_client, query)


def test_dense_query_prefetch_by_id_lookup_from_another_collection():
    fixture_points = generate_fixtures(10)

    secondary_collection_points = generate_fixtures(10)

    searcher = TestSimpleSearcher()

    local_client, http_client, grpc_client = init_clients(fixture_points)

    init_client(local_client, secondary_collection_points, SECONDARY_COLLECTION_NAME)
    init_client(http_client, secondary_collection_points, SECONDARY_COLLECTION_NAME)

    compare_clients_results(
        local_client,
        http_client,
        grpc_client,
        searcher.dense_query_prefetch_by_id_lookup_from,
        lookup_from=models.LookupLocation(collection=SECONDARY_COLLECTION_NAME, vector="text"),
    )


def test_dense_query_nested_prefetch():
    fixture_points = generate_fixtures()

    searcher = TestSimpleSearcher()

    local_client, http_client, grpc_client = init_clients(fixture_points)

    compare_clients_results(
        local_client, http_client, grpc_client, searcher.dense_query_text_nested_prefetch
    )


def test_dense_query_nested_single_prefetch():
    fixture_points = generate_fixtures()

    searcher = TestSimpleSearcher()

    local_client, http_client, grpc_client = init_clients(fixture_points)

    for query in (
        searcher.dense_query_text_nested_single_prefetch,
        searcher.dense_query_text_nested_single_prefetch_rescore,
    ):
        compare_clients_results(local_client, http_client, grpc_client, query)


def test_dense_query_filtered_prefetch():
    fixture_points = generate_fixtures()

    searcher = TestSimpleSearcher()

    local_client, http_client, grpc_client = init_clients(fixture_points)

    for i in range(100):
        query_filter = one_random_filter_please()
        try:
            compare_clients_results(
                local_client,
                http_client,
                grpc_client,
                searcher.dense_queries_prefetch_filtered,
                query_filter=query_filter,
            )
        except AssertionError as e:
            print(f"\nAttempt {i} failed with filter {query_filter}")
            raise e


def test_dense_query_root_filtered_prefetch():
    fixture_points = generate_fixtures()

    searcher = TestSimpleSearcher()

    local_client, http_client, grpc_client = init_clients(fixture_points)

    for i in range(10):
        query_filter = one_random_filter_please()
        prefetch_filter = one_random_filter_please()
        queries = [
            (searcher.dense_queries_root_filtered_prefetch, {}),
            (searcher.dense_queries_root_filtered_prefetch_formula, {}),
            (searcher.deep_dense_queries_root_filtered_prefetch, {}),
            (
                searcher.dense_queries_root_and_prefetch_filtered,
                {"prefetch_filter": prefetch_filter},
            ),
        ]
        for query, extra_kwargs in queries:
            try:
                compare_clients_results(
                    local_client,
                    http_client,
                    grpc_client,
                    query,
                    query_filter=query_filter,
                    **extra_kwargs,
                )
            except AssertionError as e:
                print(f"\nAttempt {i} of {query.__name__} failed with filter {query_filter}")
                raise e


def test_dense_query_prefetch_score_threshold():
    fixture_points = generate_fixtures()

    searcher = TestSimpleSearcher()

    local_client, http_client, grpc_client = init_clients(fixture_points)

    compare_clients_results(
        local_client, http_client, grpc_client, searcher.dense_queries_prefetch_score_threshold
    )


def test_dense_query_prefetch_parametrized():
    fixture_points = generate_fixtures()

    searcher = TestSimpleSearcher()

    local_client, http_client, grpc_client = init_clients(fixture_points)

    compare_clients_results(
        local_client,
        http_client,
        grpc_client,
        searcher.dense_queries_prefetch_parametrized,
        search_params={"exact": True},
    )
    compare_clients_results(
        local_client,
        http_client,
        grpc_client,
        searcher.dense_queries_prefetch_parametrized,
        search_params={"hnsw_ef": 128},
    )
    compare_clients_results(
        local_client,
        http_client,
        grpc_client,
        searcher.dense_queries_prefetch_parametrized,
        search_params={"indexed_only": True},
    )
    compare_clients_results(
        local_client,
        http_client,
        grpc_client,
        searcher.dense_queries_prefetch_parametrized,
        search_params={"quantization": {"ignore": True, "rescore": True, "oversampling": 2.0}},
    )


def test_dense_query_parametrized():
    fixture_points = generate_fixtures()

    searcher = TestSimpleSearcher()

    local_client, http_client, grpc_client = init_clients(fixture_points)

    compare_clients_results(
        local_client,
        http_client,
        grpc_client,
        searcher.dense_queries_parametrized,
        search_params={"exact": True},
    )
    compare_clients_results(
        local_client,
        http_client,
        grpc_client,
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

    local_client, http_client, grpc_client = init_clients(
        fixture_points, sparse_vectors_config=sparse_vectors_config
    )

    compare_clients_results(local_client, http_client, grpc_client, searcher.sparse_query_text)


def test_multivec_query():
    fixture_points = generate_multivector_fixtures()

    searcher = TestSimpleSearcher()

    local_client, http_client, grpc_client = init_clients(
        fixture_points, vectors_config=multi_vector_config
    )

    compare_clients_results(local_client, http_client, grpc_client, searcher.multivec_query_text)
    compare_clients_results(local_client, http_client, grpc_client, searcher.multivec_query_code)


def test_dense_query():
    fixture_points = generate_fixtures()

    searcher = TestSimpleSearcher()

    local_client, http_client, grpc_client = init_clients(fixture_points)
    compare_clients_results(local_client, http_client, grpc_client, searcher.dense_query_text)
    compare_clients_results(local_client, http_client, grpc_client, searcher.dense_query_image)
    compare_clients_results(local_client, http_client, grpc_client, searcher.dense_query_code)
    compare_clients_results(
        local_client, http_client, grpc_client, searcher.dense_query_text_offset
    )
    compare_clients_results(
        local_client, http_client, grpc_client, searcher.dense_query_text_with_vector
    )
    compare_clients_results(
        local_client, http_client, grpc_client, searcher.dense_query_score_threshold
    )
    compare_clients_results(
        local_client, http_client, grpc_client, searcher.dense_query_text_select_payload
    )
    compare_clients_results(
        local_client, http_client, grpc_client, searcher.dense_query_image_select_vector
    )
    compare_clients_results(local_client, http_client, grpc_client, searcher.dense_payload_exclude)
    compare_clients_results(
        local_client, http_client, grpc_client, searcher.dense_queries_prefetch_offset
    )

    for i in range(100):
        query_filter = one_random_filter_please()
        try:
            compare_clients_results(
                local_client,
                http_client,
                grpc_client,
                searcher.filter_dense_query_text,
                query_filter=query_filter,
            )
            compare_clients_results(
                local_client,
                http_client,
                grpc_client,
                searcher.filter_query_scroll,
                query_filter=query_filter,
            )
        except AssertionError as e:
            print(f"\nFailed with filter {query_filter}")
            raise e


def test_dense_query_score_threshold_boundary():
    def query_score_threshold(
        client: QdrantBase, query: models.Query, thresholds: list[float]
    ) -> list[list[models.ScoredPoint]]:
        return [
            client.query_points(
                collection_name=COLLECTION_NAME,
                query=query,
                limit=10,
                score_threshold=threshold,
                with_payload=False,
            ).points
            for threshold in thresholds
        ]

    recommend = models.RecommendQuery(
        recommend=models.RecommendInput(
            positive=[[0.5, 0.0]], strategy=models.RecommendStrategy.SUM_SCORES
        )
    )
    discover = models.DiscoverQuery(
        discover=models.DiscoverInput(
            target=[1.0, 0.0],
            context=[models.ContextPair(positive=[1.0, 0.0], negative=[-1.0, 0.0])],
        )
    )

    # Vectors are picked so that every score is exactly representable in float32,
    # which makes the equality boundary reproducible on both sides. Each threshold is
    # equal to the score of one of the points: that point must be dropped, since a
    # threshold keeps only strictly better scores.
    cases = [
        # distance, vectors, query, thresholds
        # nearest queries are scored by the metric itself
        (
            models.Distance.COSINE,
            [[1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [-1.0, 0.0]],
            [1.0, 0.0],
            [1.0, 0.0, -1.0],
        ),
        (models.Distance.DOT, [[2.0, 0.0], [1.0, 0.0], [0.5, 0.0]], [1.0, 0.0], [2.0, 1.0, 0.5]),
        (
            models.Distance.EUCLID,
            [[0.5, 0.0], [1.0, 0.0], [2.0, 0.0]],
            [0.0, 0.0],
            [0.5, 1.0, 2.0],
        ),
        (
            models.Distance.MANHATTAN,
            [[0.5, 0.0], [1.0, 0.0], [2.0, 0.0]],
            [0.0, 0.0],
            [0.5, 1.0, 2.0],
        ),
        # recommend and discovery score higher-is-better whatever the metric is, but
        # the threshold keeps following the metric's own direction, so on euclid the
        # best-scoring point already stops the scan
        (
            models.Distance.DOT,
            [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]],
            recommend,
            [1.5, 1.0, 0.5, 0.0],
        ),
        (
            models.Distance.EUCLID,
            [[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]],
            recommend,
            [-0.25, -2.25],
        ),
        (
            models.Distance.COSINE,
            [[1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [-1.0, 0.0]],
            discover,
            [1.75, 0.5, -0.75],
        ),
    ]

    for distance, vectors, query, thresholds in cases:
        fixture_points = [
            models.PointStruct(id=idx, vector=vector)
            for idx, vector in enumerate(vectors, start=1)
        ]

        local_client, http_client, grpc_client = init_clients(
            fixture_points,
            vectors_config=models.VectorParams(size=len(vectors[0]), distance=distance),
        )

        try:
            compare_clients_results(
                local_client,
                http_client,
                grpc_client,
                query_score_threshold,
                query=query,
                thresholds=thresholds,
            )
        except AssertionError as e:
            print(f"\nFailed with distance {distance} and query {type(query).__name__}")
            raise e


def test_dense_query_orderby():
    fixture_points = generate_fixtures(200)

    searcher = TestSimpleSearcher()

    local_client, http_client, grpc_client = init_clients(fixture_points)

    http_client.create_payload_index(
        COLLECTION_NAME, "rand_digit", models.PayloadSchemaType.INTEGER, wait=True
    )

    compare_clients_results(local_client, http_client, grpc_client, searcher.dense_queries_orderby)
    compare_clients_results(
        local_client, http_client, grpc_client, searcher.deep_dense_queries_orderby
    )


def test_dense_query_recommend():
    fixture_points = generate_fixtures()

    searcher = TestSimpleSearcher()

    local_client, http_client, grpc_client = init_clients(fixture_points)

    compare_clients_results(local_client, http_client, grpc_client, searcher.dense_recommend_image)
    compare_clients_results(local_client, http_client, grpc_client, searcher.dense_many_recommend)


def test_dense_query_rescore():
    fixture_points = generate_fixtures()

    searcher = TestSimpleSearcher()

    local_client, http_client, grpc_client = init_clients(fixture_points)

    compare_clients_results(local_client, http_client, grpc_client, searcher.dense_queries_rescore)
    compare_clients_results(
        local_client, http_client, grpc_client, searcher.dense_deep_queries_rescore
    )


def test_dense_query_fusion():
    fixture_points = generate_fixtures()

    searcher = TestSimpleSearcher()

    local_client, http_client, grpc_client = init_clients(fixture_points)

    compare_clients_results(local_client, http_client, grpc_client, searcher.dense_query_rrf)
    compare_clients_results(
        local_client, http_client, grpc_client, searcher.dense_query_rrf_plain_prefetch
    )
    compare_clients_results(local_client, http_client, grpc_client, searcher.dense_query_dbsf)
    compare_clients_results(
        local_client, http_client, grpc_client, searcher.deep_dense_queries_rrf
    )
    compare_clients_results(
        local_client, http_client, grpc_client, searcher.deep_dense_queries_dbsf
    )
    compare_clients_results(
        local_client, http_client, grpc_client, searcher.dense_query_parametrized_rrf
    )
    compare_clients_results(
        local_client, http_client, grpc_client, searcher.dense_query_rrf_score_threshold
    )
    compare_clients_results(
        local_client, http_client, grpc_client, searcher.dense_query_formula_score_threshold
    )


def test_dense_query_discovery_context():
    n_vectors = 250
    fixture_points = generate_fixtures(n_vectors)

    searcher = TestSimpleSearcher()

    local_client, http_client, grpc_client = init_clients(fixture_points)

    compare_clients_results(local_client, http_client, grpc_client, searcher.dense_discovery_image)
    compare_clients_results(local_client, http_client, grpc_client, searcher.dense_many_discover)
    compare_clients_results(
        local_client,
        http_client,
        grpc_client,
        searcher.dense_context_image,
        is_context_search=True,
        limit=n_vectors,
    )


def test_simple_opt_vectors_query():
    fixture_points = generate_fixtures(skip_vectors=True)

    searcher = TestSimpleSearcher()

    local_client, http_client, grpc_client = init_clients(fixture_points)

    compare_clients_results(local_client, http_client, grpc_client, searcher.dense_query_text)
    compare_clients_results(local_client, http_client, grpc_client, searcher.dense_query_image)
    compare_clients_results(local_client, http_client, grpc_client, searcher.dense_query_code)
    compare_clients_results(
        local_client, http_client, grpc_client, searcher.dense_query_text_offset
    )
    compare_clients_results(
        local_client, http_client, grpc_client, searcher.dense_query_text_with_vector
    )
    compare_clients_results(
        local_client, http_client, grpc_client, searcher.dense_query_score_threshold
    )
    compare_clients_results(
        local_client, http_client, grpc_client, searcher.dense_query_text_select_payload
    )
    compare_clients_results(
        local_client, http_client, grpc_client, searcher.dense_query_image_select_vector
    )
    compare_clients_results(local_client, http_client, grpc_client, searcher.dense_payload_exclude)

    for i in range(100):
        query_filter = one_random_filter_please()
        try:
            compare_clients_results(
                local_client,
                http_client,
                grpc_client,
                searcher.filter_dense_query_text,
                query_filter=query_filter,
            )
            compare_clients_results(
                local_client,
                http_client,
                grpc_client,
                searcher.filter_query_scroll,
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

    local_client, http_client, grpc_client = init_clients(
        fixture_points, vectors_config=vectors_config
    )

    for i in range(100):
        query_filter = one_random_filter_please()
        try:
            compare_clients_results(
                local_client,
                http_client,
                grpc_client,
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

        local_client.close()
        local_client_2 = init_local(tmpdir)

        http_client = init_remote()
        grpc_client = init_remote(prefer_grpc=True)
        init_client(http_client, fixture_points)

        http_client.set_payload(COLLECTION_NAME, {"test": f"test"}, payload_update_filter)

        payload_update_filter = one_random_filter_please()
        local_client_2.set_payload(COLLECTION_NAME, {"test": "test2"}, payload_update_filter)
        http_client.set_payload(COLLECTION_NAME, {"test": "test2"}, payload_update_filter)

        for i in range(10):
            query_filter = one_random_filter_please()
            try:
                compare_clients_results(
                    local_client_2,
                    http_client,
                    grpc_client,
                    searcher.filter_dense_query_text,
                    query_filter=query_filter,
                )
            except AssertionError as e:
                print(f"\nFailed with filter {query_filter}")
                local_client_2.close()
                raise e
        local_client_2.close()


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
        local_client.close()
        local_client_2 = init_local(tmpdir)

        count_after_load = local_client_2.count(COLLECTION_NAME)

        assert count_after_load == count_before_load

        http_client = init_remote()
        grpc_client = init_remote(prefer_grpc=True)
        init_client(http_client, fixture_points)

        http_client.set_payload(COLLECTION_NAME, {"test": f"test"}, payload_update_filter)

        payload_update_filter = one_random_filter_please()
        local_client_2.set_payload(COLLECTION_NAME, {"test": "test2"}, payload_update_filter)
        http_client.set_payload(COLLECTION_NAME, {"test": "test2"}, payload_update_filter)

        for i in range(10):
            query_filter = one_random_filter_please()
            try:
                compare_clients_results(
                    local_client_2,
                    http_client,
                    grpc_client,
                    searcher.filter_dense_query_text,
                    query_filter=query_filter,
                )
            except AssertionError as e:
                print(f"\nFailed with filter {query_filter}")
                local_client_2.close()
                raise e
        local_client_2.close()


def test_query_invalid_vector_type():
    import grpc

    fixture_points = generate_fixtures()

    local_client, http_client, grpc_client = init_clients(fixture_points)

    vector_invalid_type = [1, 2, 3, 4]
    with pytest.raises(ValueError):
        local_client.query_points(
            collection_name=COLLECTION_NAME, query=vector_invalid_type, using="text"
        )

    with pytest.raises(UnexpectedResponse):
        http_client.query_points(
            collection_name=COLLECTION_NAME, query=vector_invalid_type, using="text"
        )

    with pytest.raises(grpc.RpcError):
        grpc_client.query_points(
            collection_name=COLLECTION_NAME, query=vector_invalid_type, using="text"
        )


def test_query_with_nan():
    fixture_points = generate_fixtures()

    local_client, http_client, grpc_client = init_clients(fixture_points)

    vector = np.random.random(text_vector_size)
    vector[4] = np.nan
    query = vector.tolist()
    with pytest.raises(AssertionError):
        local_client.query_points(COLLECTION_NAME, query=query, using="text")

    with pytest.raises(UnexpectedResponse):
        http_client.query_points(COLLECTION_NAME, query=query, using="text")

    # TODO: this doesn't fail, instead it returns points with `nan` score
    # with pytest.raises(UnexpectedResponse):
    # print(grpc_client.query_points(COLLECTION_NAME, query=query, using="text"))

    single_vector_config = models.VectorParams(
        size=text_vector_size, distance=models.Distance.COSINE
    )

    local_client.delete_collection(COLLECTION_NAME)
    local_client.create_collection(COLLECTION_NAME, vectors_config=single_vector_config)

    http_client.delete_collection(COLLECTION_NAME)
    http_client.create_collection(COLLECTION_NAME, vectors_config=single_vector_config)

    fixture_points = generate_fixtures(vectors_sizes=text_vector_size)
    init_client(local_client, fixture_points, vectors_config=single_vector_config)
    init_client(http_client, fixture_points, vectors_config=single_vector_config)

    with pytest.raises(AssertionError):
        print(local_client.query_points(COLLECTION_NAME, query=query))

    with pytest.raises(UnexpectedResponse):
        http_client.query_points(COLLECTION_NAME, query=query)

    # TODO: this doesn't fail, instead it returns points with `nan` score
    # with pytest.raises(UnexpectedResponse):
    #     print(grpc_client.query_points(COLLECTION_NAME, query=query))


def test_flat_query_dense_interface():
    fixture_points = generate_fixtures()

    searcher = TestSimpleSearcher()

    local_client, http_client, grpc_client = init_clients(fixture_points)

    compare_clients_results(local_client, http_client, grpc_client, searcher.dense_query_text)
    compare_clients_results(
        local_client, http_client, grpc_client, searcher.dense_query_text_np_array
    )
    compare_clients_results(
        local_client, http_client, grpc_client, searcher.dense_query_text_by_id
    )


def test_flat_query_sparse_interface():
    fixture_points = generate_sparse_fixtures()

    searcher = TestSimpleSearcher()

    local_client, http_client, grpc_client = init_clients(
        fixture_points, sparse_vectors_config=sparse_vectors_config
    )

    compare_clients_results(local_client, http_client, grpc_client, searcher.sparse_query_text)


def test_flat_query_multivector_interface():
    fixture_points = generate_multivector_fixtures()

    searcher = TestSimpleSearcher()

    local_client, http_client, grpc_client = init_clients(
        fixture_points, vectors_config=multi_vector_config
    )

    compare_clients_results(local_client, http_client, grpc_client, searcher.multivec_query_text)


def test_original_input_persistence():
    # this test is not supposed to compare outputs, but to check that we're not modifying input structures
    # it used to fail when we were modifying input structures in local mode
    # the reason was that we were replacing point id with a sparse vector, and then, when we needed a dense vector
    # from the same point id, we already had point id replaced with a sparse vector
    num_points = 50
    vectors_config = {"text": models.VectorParams(size=50, distance=models.Distance.COSINE)}
    sparse_vectors_config = {"sparse-text": models.SparseVectorParams()}
    fixture_points = generate_fixtures(vectors_sizes={"text": 50}, num=num_points)
    sparse_fixture_points = generate_sparse_fixtures(num=num_points)
    points = [
        models.PointStruct(
            id=point.id,
            payload=point.payload,
            vector={
                "text": point.vector["text"],
                "sparse-text": sparse_point.vector["sparse-text"],
            },
        )
        for point, sparse_point in zip(fixture_points, sparse_fixture_points)
    ]
    dense_vector_name = "text"
    sparse_vector_name = "sparse-text"
    local_client, http_client, grpc_client = init_clients(
        points, vectors_config=vectors_config, sparse_vectors_config=sparse_vectors_config
    )

    point_id = 1
    shared_instance = models.RecommendInput(positive=[point_id], negative=[])
    prefetch = [
        models.Prefetch(
            query=models.RecommendQuery(recommend=shared_instance),
            using=sparse_vector_name,
        ),
    ]
    local_client.query_points(
        collection_name=COLLECTION_NAME,
        prefetch=prefetch,
        query=models.RecommendQuery(recommend=shared_instance),
        using=dense_vector_name,
    )

    shared_instance = models.RecommendInput(positive=[point_id], negative=[])
    prefetch = [
        models.Prefetch(
            query=models.RecommendQuery(recommend=shared_instance),
            using=sparse_vector_name,
        ),
    ]
    http_client.query_points(
        collection_name=COLLECTION_NAME,
        prefetch=prefetch,
        query=models.RecommendQuery(recommend=shared_instance),
        using=dense_vector_name,
    )

    grpc_client.query_points(
        collection_name=COLLECTION_NAME,
        prefetch=prefetch,
        query=models.RecommendQuery(recommend=shared_instance),
        using=dense_vector_name,
    )


def test_query_group():
    fixture_points = generate_fixtures()

    secondary_collection_points = generate_fixtures(10)

    searcher = TestSimpleSearcher()

    local_client, http_client, grpc_client = init_clients(fixture_points)

    init_client(local_client, secondary_collection_points, SECONDARY_COLLECTION_NAME)
    init_client(http_client, secondary_collection_points, SECONDARY_COLLECTION_NAME)

    http_client.create_payload_index(
        COLLECTION_NAME, field_name="id", field_schema=models.PayloadSchemaType.INTEGER
    )
    http_client.create_payload_index(
        COLLECTION_NAME, field_name="rand_digit", field_schema=models.PayloadSchemaType.INTEGER
    )
    http_client.create_payload_index(
        COLLECTION_NAME, field_name="two_words", field_schema=models.PayloadSchemaType.KEYWORD
    )
    http_client.create_payload_index(
        COLLECTION_NAME,
        field_name="city.name",
        field_schema=models.PayloadSchemaType.KEYWORD,
    )
    http_client.create_payload_index(
        COLLECTION_NAME,
        field_name="maybe",
        field_schema=models.PayloadSchemaType.KEYWORD,
    )
    http_client.create_payload_index(
        COLLECTION_NAME,
        field_name="maybe_null",
        field_schema=models.PayloadSchemaType.KEYWORD,
    )

    searcher.group_size = 5
    searcher.limit = 3
    for key in group_by_keys():
        searcher.group_by = key
        compare_clients_results(local_client, http_client, grpc_client, searcher.dense_query_group)
        compare_clients_results(
            local_client, http_client, grpc_client, searcher.dense_query_group_with_lookup
        )
        compare_clients_results(
            local_client, http_client, grpc_client, searcher.dense_query_group_with_lookup_from
        )

        compare_clients_results(
            local_client, http_client, grpc_client, searcher.dense_queries_rescore_group
        )
        compare_clients_results(
            local_client,
            http_client,
            grpc_client,
            searcher.dense_queries_rescore_group_single_prefetch,
        )
        compare_clients_results(
            local_client,
            http_client,
            grpc_client,
            searcher.dense_query_lookup_from_group,
            lookup_from=models.LookupLocation(collection=SECONDARY_COLLECTION_NAME, vector="text"),
        )

    searcher.group_by = "city.name"

    for i in range(100):
        query_filter = one_random_filter_please()
        try:
            compare_clients_results(
                local_client,
                http_client,
                grpc_client,
                searcher.filter_dense_query_group,
                query_filter=query_filter,
            )
        except AssertionError as e:
            print(f"\nFailed with filter {query_filter}")
            raise e


def test_random_sampling():
    fixture_points = generate_fixtures(100)

    searcher = TestSimpleSearcher()

    local_client, http_client, grpc_client = init_clients(fixture_points)

    compare_clients_results(local_client, http_client, grpc_client, searcher.random_query)
    compare_clients_results(local_client, http_client, grpc_client, searcher.random_query_offset)


def test_formula_query():
    points_count = 100
    fixture_points = generate_fixtures(points_count)

    searcher = TestSimpleSearcher()

    local_client, http_client, grpc_client = init_clients(fixture_points)

    defaults = {
        "rand_digit": 5,
        "maybe_null": None,
        "mixed_type": 0.3,
        "city.geo": {"lon": 0.4, "lat": 0.5},
    }

    for _ in range(50):
        formula = models.FormulaQuery(
            formula=one_random_expression_please(max_depth=2), defaults=defaults
        )

        # We need to score point by point to make sure that the errors that come up correspond to the same point.
        #
        # Otherwise, we can have discrepancy where one point produced one error,
        # and another caused a different error in the other client
        for point_id in range(points_count):
            try:
                compare_clients_results(
                    local_client,
                    http_client,
                    grpc_client,
                    searcher.score_boosting,
                    formula=formula,
                    point_id=point_id,
                )
            except Exception as e:
                print(f"\nFailed with formula {formula} on point {fixture_points[point_id]}")
                raise e


def test_empty_collection_bm25_search():
    local_client, http_client, grpc_client = init_clients(
        [],
        vectors_config={},
        sparse_vectors_config={"sparse": models.SparseVectorParams(modifier=models.Modifier.IDF)},
    )

    query_vector = models.SparseVector(indices=[14, 73], values=[0.3, 0.2])

    def search_please(client: QdrantBase) -> models.QueryResponse:
        return client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            using="sparse",
            limit=10,
        )

    compare_clients_results(local_client, http_client, grpc_client, search_please)


def test_mmr_queries():
    fixture_points = generate_fixtures()

    searcher = TestSimpleSearcher()

    local_client, http_client, grpc_client = init_clients(fixture_points)

    compare_clients_results(local_client, http_client, grpc_client, searcher.default_mmr_query)
    compare_clients_results(
        local_client, http_client, grpc_client, searcher.mmr_query_parametrized
    )
    compare_clients_results(
        local_client, http_client, grpc_client, searcher.mmr_query_parametrized_score_threshold
    )
    compare_clients_results(
        local_client, http_client, grpc_client, searcher.mmr_query_parametrized_dot
    )
    compare_clients_results(
        local_client, http_client, grpc_client, searcher.mmr_query_parametrized_dot_score_threshold
    )

    compare_clients_results(
        local_client, http_client, grpc_client, searcher.mmr_query_parametrized_euclid
    )
    compare_clients_results(
        local_client,
        http_client,
        grpc_client,
        searcher.mmr_query_parametrized_euclid_score_threshold,
    )


def test_mmr_tie_breaking():
    """MMR ordering when candidates tie exactly, which is where local mode used to diverge.

    Core seeds the selection with the most relevant candidate and then picks the best MMR score
    with Rust's `max_by_key`, which returns the *last* maximum on ties, while `np.argmax` returns
    the first one. Core also holds the pending candidates in an `IndexSet` and drops the selected
    one with `swap_remove`, which moves the last candidate into the freed slot and so changes the
    order candidates are visited in.

    Relevance scores are kept distinct on purpose. Core resolves a tie in *relevance* by the
    order search returned the candidates in, and `Ord for ScoredPoint` compares score only,
    with no tiebreak on id, so equally scored points come back in the order they sit in the
    segment. `upsert_points_impl` derives that order from `AHashMap` key iteration, which is
    randomly seeded per operation, so it is stable for repeated queries against one collection
    but reshuffles on every rebuild of the fixture - regardless of segment count. Every test
    run rebuilds the collection, so only ties in the MMR score can be asserted on.

    All coordinates are exact binary fractions, so the MMR ties are exact in f32 both locally
    and in core.
    """

    def mmr_query(client: QdrantBase, query, using=None) -> models.QueryResponse:
        return client.query_points(
            collection_name=COLLECTION_NAME,
            query=models.NearestQuery(nearest=query, mmr=models.Mmr()),
            using=using,
            # these points carry no payload, and grpc reports that as None while rest reports {}
            with_payload=False,
            limit=10,
        )

    # dense, DOT: query is [1, 0, 0, 0], so relevance is the first coordinate.
    # Once id 1 is selected, ids 2 and 4 - the first and the last of the pending candidates -
    # both end up with an MMR score of -0.5. `swap_remove` visits the pending candidates as
    # [4, 2, 3], while an order-preserving removal would visit them as [2, 3, 4], so the two
    # disagree on which of the tied candidates is the *last* maximum.
    clients = init_clients(
        [
            models.PointStruct(id=1, vector=[2.0, 1.0, 0.0, 0.0]),  # relevance 2.0, selected first
            models.PointStruct(id=2, vector=[1.0, 0.0, 0.0, 0.0]),  # relevance 1.0, MMR -0.5
            models.PointStruct(id=3, vector=[0.5, 1.0, 0.0, 0.0]),  # relevance 0.5, MMR -0.75
            models.PointStruct(id=4, vector=[0.25, 0.75, 0.0, 0.0]),  # relevance 0.25, MMR -0.5
        ],
        vectors_config=models.VectorParams(size=4, distance=models.Distance.DOT),
    )
    compare_clients_results(*clients, mmr_query, query=[1.0, 0.0, 0.0, 0.0])

    # the same tie with EUCLID, to show the tie-breaking is not specific to DOT
    clients = init_clients(
        [
            models.PointStruct(id=1, vector=[0.25, 0.0]),  # relevance -0.0625, selected first
            models.PointStruct(id=2, vector=[0.5, 0.25]),  # relevance -0.3125, MMR -0.09375
            models.PointStruct(id=3, vector=[0.5, 0.5]),  # relevance -0.5, MMR -0.09375
            models.PointStruct(id=4, vector=[0.75, 0.5]),  # relevance -0.8125, MMR -0.15625
        ],
        vectors_config=models.VectorParams(size=2, distance=models.Distance.EUCLID),
    )
    compare_clients_results(*clients, mmr_query, query=[0.0, 0.0])

    # the same tie on a MAX_SIM multivector field, where the divergence was first spotted
    clients = init_clients(
        [
            # the extra vector is never the closest one, it only exercises the MAX_SIM reduction
            models.PointStruct(
                id=1, vector={"multi": [[2.0, 1.0, 0.0, 0.0], [0.0, 0.0, -1.0, 0.0]]}
            ),
            models.PointStruct(id=2, vector={"multi": [[1.0, 0.0, 0.0, 0.0]]}),
            models.PointStruct(id=3, vector={"multi": [[0.5, 0.5, 0.0, 0.0]]}),
            models.PointStruct(id=4, vector={"multi": [[0.25, 1.0, 0.0, 0.0]]}),
        ],
        vectors_config={
            "multi": models.VectorParams(
                size=4,
                distance=models.Distance.DOT,
                multivector_config=models.MultiVectorConfig(
                    comparator=models.MultiVectorComparator.MAX_SIM
                ),
            )
        },
    )
    compare_clients_results(*clients, mmr_query, query=[[1.0, 0.0, 0.0, 0.0]], using="multi")


def test_relevance_feedback_queries():
    fixture_points = generate_fixtures()

    searcher = TestSimpleSearcher()

    local_client, http_client, grpc_client = init_clients(fixture_points)

    compare_clients_results(
        local_client, http_client, grpc_client, searcher.relevance_feedback_query
    )
    compare_clients_results(
        local_client, http_client, grpc_client, searcher.relevance_feedback_query_euclid
    )
