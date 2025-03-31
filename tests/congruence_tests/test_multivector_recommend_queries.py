import numpy as np

from qdrant_client.client_base import QdrantBase
from qdrant_client.http.models import models

from tests.congruence_tests.test_common import (
    COLLECTION_NAME,
    compare_client_results,
    generate_multivector_fixtures,
    image_vector_size,
    init_client,
    init_local,
    init_remote,
    multi_vector_config,
)

secondary_collection_name = "congruence_secondary_collection"


class TestSimpleRecommendation:
    __test__ = False

    def __init__(self):
        self.query_image = np.random.random(image_vector_size).tolist()

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
                    positive=[10, 20],
                    negative=[11, 21],
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
                        20,
                    ],
                    negative=[11, 21],
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
                                strategy=models.RecommendStrategy.BEST_SCORE,
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
                                positive=[4],
                                negative=[],
                                strategy=models.RecommendStrategy.SUM_SCORES,
                            )
                        ),
                        limit=2,
                        using="multi-image",
                    ),
                ],
            )
        ]


def test_simple_recommend() -> None:
    fixture_points = generate_multivector_fixtures()

    secondary_collection_points = generate_multivector_fixtures(100)

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

    compare_client_results(local_client, remote_client, searcher.best_score_recommend)
    compare_client_results(local_client, remote_client, searcher.best_score_recommend_euclid)
    compare_client_results(
        local_client, remote_client, searcher.only_negatives_best_score_recommend
    )
    compare_client_results(
        local_client, remote_client, searcher.only_negatives_best_score_recommend_euclid
    )

    compare_client_results(local_client, remote_client, searcher.sum_scores_recommend)
    compare_client_results(local_client, remote_client, searcher.sum_scores_recommend_euclid)
    compare_client_results(
        local_client, remote_client, searcher.only_negatives_sum_scores_recommend
    )
    compare_client_results(
        local_client, remote_client, searcher.only_negatives_sum_scores_recommend_euclid
    )

    compare_client_results(local_client, remote_client, searcher.recommend_batch)
