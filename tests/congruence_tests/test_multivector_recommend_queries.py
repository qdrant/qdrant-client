from typing import Optional


from qdrant_client.client_base import QdrantBase
from qdrant_client.http.models import models

from tests.congruence_tests.test_common import (
    COLLECTION_NAME,
    compare_client_results,
    generate_multivector_fixtures,
    init_client,
    init_local,
    init_remote,
    multi_vector_config,
)

secondary_collection_name = "congruence_secondary_collection"


class TestSimpleRecommendation:
    __test__ = False

    @classmethod
    def simple_recommend_image(cls, client: QdrantBase) -> list[models.ScoredPoint]:
        return client.query_points(
            collection_name=COLLECTION_NAME,
            query=models.RecommendQuery(
                recommend=models.RecommendInput(
                    positive=[10],
                    negative=[],
                )
            ),
            with_payload=True,
            limit=10,
            using="multi-image",
        ).points

    @classmethod
    def simple_recommend_text(cls, client: QdrantBase) -> list[models.ScoredPoint]:
        return client.query_points(
            collection_name=COLLECTION_NAME,
            query=models.RecommendQuery(
                recommend=models.RecommendInput(
                    positive=[10],
                    negative=[],
                )
            ),
            with_payload=True,
            limit=10,
            using="multi-text",
        ).points

    @classmethod
    def simple_recommend_code(cls, client: QdrantBase) -> list[models.ScoredPoint]:
        return client.query_points(
            collection_name=COLLECTION_NAME,
            query=models.RecommendQuery(
                recommend=models.RecommendInput(
                    positive=[10, 11],
                    negative=[12],
                )
            ),
            with_payload=True,
            limit=10,
            using="multi-code",
        ).points

    @classmethod
    def simple_recommend_negative(cls, client: QdrantBase) -> list[models.ScoredPoint]:
        return client.query_points(
            collection_name=COLLECTION_NAME,
            query=models.RecommendQuery(
                recommend=models.RecommendInput(
                    positive=[10],
                    negative=[15, 7],
                )
            ),
            with_payload=True,
            limit=10,
            using="multi-image",
        ).points

    @classmethod
    def recommend_from_another_collection(
        cls, client: QdrantBase, positive_point_id: Optional[int] = None
    ) -> list[models.ScoredPoint]:
        return client.query_points(
            collection_name=COLLECTION_NAME,
            query=models.RecommendQuery(
                recommend=models.RecommendInput(
                    positive=[10] if positive_point_id is None else [positive_point_id],
                    negative=[15, 7] if positive_point_id is None else [],
                )
            ),
            with_payload=True,
            limit=10,
            using="multi-image",
            lookup_from=models.LookupLocation(
                collection=secondary_collection_name,
                vector="multi-image",
            ),
        ).points

    @classmethod
    def best_score_recommend(cls, client: QdrantBase) -> list[models.ScoredPoint]:
        return client.query_points(
            collection_name=COLLECTION_NAME,
            query=models.RecommendQuery(
                recommend=models.RecommendInput(
                    positive=[10, 20],
                    strategy=models.RecommendStrategy.BEST_SCORE,
                )
            ),
            with_payload=True,
            limit=10,
            using="multi-image",
        ).points

    @classmethod
    def best_score_recommend_euclid(cls, client: QdrantBase) -> list[models.ScoredPoint]:
        return client.query_points(
            collection_name=COLLECTION_NAME,
            query=models.RecommendQuery(
                recommend=models.RecommendInput(
                    positive=[10],
                    negative=[],
                    strategy=models.RecommendStrategy.BEST_SCORE,
                )
            ),
            with_payload=True,
            limit=10,
            using="multi-code",
        ).points

    @classmethod
    def only_negatives_best_score_recommend(cls, client: QdrantBase) -> list[models.ScoredPoint]:
        return client.query_points(
            collection_name=COLLECTION_NAME,
            query=models.RecommendQuery(
                recommend=models.RecommendInput(
                    positive=None, negative=[10, 12], strategy=models.RecommendStrategy.BEST_SCORE
                )
            ),
            with_payload=True,
            limit=10,
            using="multi-image",
        ).points

    @classmethod
    def only_negatives_best_score_recommend_euclid(
        cls, client: QdrantBase
    ) -> list[models.ScoredPoint]:
        return client.query_points(
            collection_name=COLLECTION_NAME,
            query=models.RecommendQuery(
                recommend=models.RecommendInput(
                    positive=None, negative=[10, 12], strategy="best_score"
                )
            ),
            with_payload=True,
            limit=10,
            using="multi-code",
        ).points

    @classmethod
    def sum_scores_recommend(cls, client: QdrantBase) -> list[models.ScoredPoint]:
        return client.query_points(
            collection_name=COLLECTION_NAME,
            query=models.RecommendQuery(
                recommend=models.RecommendInput(
                    positive=[
                        10,
                        20,
                    ],
                    negative=[],
                    strategy=models.RecommendStrategy.SUM_SCORES,
                )
            ),
            with_payload=True,
            limit=10,
            using="multi-image",
        ).points

    @classmethod
    def sum_scores_recommend_euclid(cls, client: QdrantBase) -> list[models.ScoredPoint]:
        return client.query_points(
            collection_name=COLLECTION_NAME,
            query=models.RecommendQuery(
                recommend=models.RecommendInput(
                    positive=[
                        10,
                    ],
                    negative=[],
                    strategy=models.RecommendStrategy.SUM_SCORES,
                )
            ),
            with_payload=True,
            limit=10,
            using="multi-code",
        ).points

    @classmethod
    def only_negatives_sum_scores_recommend(cls, client: QdrantBase) -> list[models.ScoredPoint]:
        return client.query_points(
            collection_name=COLLECTION_NAME,
            query=models.RecommendQuery(
                recommend=models.RecommendInput(
                    positive=None, negative=[10, 12], strategy=models.RecommendStrategy.SUM_SCORES
                )
            ),
            with_payload=True,
            limit=10,
            using="multi-image",
        ).points

    @classmethod
    def only_negatives_sum_scores_recommend_euclid(
        cls, client: QdrantBase
    ) -> list[models.ScoredPoint]:
        return client.query_points(
            collection_name=COLLECTION_NAME,
            query=models.RecommendQuery(
                recommend=models.RecommendInput(
                    positive=None,
                    negative=[10, 12],
                    strategy="sum_scores",  # type: ignore  # check it works with a literal
                )
            ),
            with_payload=True,
            limit=10,
            using="multi-code",
        ).points

    @staticmethod
    def recommend_batch(client: QdrantBase) -> list[list[models.ScoredPoint]]:
        return [
            response.points
            for response in client.query_batch_points(
                collection_name=COLLECTION_NAME,
                requests=[
                    models.QueryRequest(
                        query=models.RecommendQuery(
                            recommend=models.RecommendInput(
                                positive=[10],
                                negative=[],
                                strategy=models.RecommendStrategy.AVERAGE_VECTOR,
                            )
                        ),
                        limit=2,
                        using="multi-image",
                        lookup_from=models.LookupLocation(
                            collection=secondary_collection_name,
                            vector="multi-image",
                        ),
                    ),
                    models.QueryRequest(
                        query=models.RecommendQuery(
                            recommend=models.RecommendInput(
                                positive=[3],
                                negative=[],
                                strategy=models.RecommendStrategy.BEST_SCORE,
                            )
                        ),
                        limit=2,
                        using="multi-code",
                    ),
                ],
            )
        ]


def test_simple_recommend() -> None:
    fixture_points = generate_multivector_fixtures(100)

    secondary_collection_points = generate_multivector_fixtures(50)

    searcher = TestSimpleRecommendation()

    local_client = init_local()
    init_client(local_client, fixture_points, vectors_config=multi_vector_config)
    init_client(
        local_client,
        secondary_collection_points,
        secondary_collection_name,
        vectors_config=multi_vector_config,
    )

    remote_client = init_remote()
    init_client(remote_client, fixture_points, vectors_config=multi_vector_config)
    init_client(
        remote_client,
        secondary_collection_points,
        secondary_collection_name,
        vectors_config=multi_vector_config,
    )
    compare_client_results(local_client, remote_client, searcher.simple_recommend_image)
    compare_client_results(local_client, remote_client, searcher.simple_recommend_text)
    compare_client_results(local_client, remote_client, searcher.simple_recommend_code)
    compare_client_results(local_client, remote_client, searcher.simple_recommend_negative)
    compare_client_results(local_client, remote_client, searcher.recommend_from_another_collection)
    compare_client_results(local_client, remote_client, searcher.best_score_recommend)
    compare_client_results(local_client, remote_client, searcher.best_score_recommend_euclid)
    compare_client_results(
        local_client, remote_client, searcher.only_negatives_best_score_recommend
    )
    compare_client_results(
        local_client, remote_client, searcher.only_negatives_best_score_recommend_euclid
    )
    #
    compare_client_results(local_client, remote_client, searcher.sum_scores_recommend)
    compare_client_results(local_client, remote_client, searcher.sum_scores_recommend_euclid)
    compare_client_results(
        local_client, remote_client, searcher.only_negatives_sum_scores_recommend
    )
    compare_client_results(
        local_client, remote_client, searcher.only_negatives_sum_scores_recommend_euclid
    )

    compare_client_results(local_client, remote_client, searcher.recommend_batch)
