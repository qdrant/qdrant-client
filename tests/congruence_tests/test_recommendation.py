import numpy as np
import pytest

from qdrant_client.client_base import QdrantBase
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.http.models import models
from tests.congruence_tests.test_common import (
    COLLECTION_NAME,
    compare_client_results,
    generate_fixtures,
    image_vector_size,
    init_client,
    init_local,
    init_remote,
)
from tests.fixtures.filters import one_random_filter_please

secondary_collection_name = "congruence_secondary_collection"


class TestSimpleRecommendation:
    __test__ = False

    def __init__(self):
        self.query_image = np.random.random(image_vector_size).tolist()

    @classmethod
    def simple_recommend_image(cls, client: QdrantBase) -> list[models.ScoredPoint]:
        return client.query_points(
            collection_name=COLLECTION_NAME,
            query=models.RecommendQuery(
                recommend=models.RecommendInput(positive=[10], negative=[])
            ),
            with_payload=True,
            limit=10,
            using="image",
        ).points

    @classmethod
    def many_recommend(cls, client: QdrantBase) -> list[models.ScoredPoint]:
        return client.query_points(
            collection_name=COLLECTION_NAME,
            query=models.RecommendQuery(recommend=models.RecommendInput(positive=[10, 19])),
            with_payload=True,
            limit=10,
            using="image",
        ).points

    @classmethod
    def simple_recommend_negative(cls, client: QdrantBase) -> list[models.ScoredPoint]:
        return client.query_points(
            collection_name=COLLECTION_NAME,
            query=models.RecommendQuery(
                recommend=models.RecommendInput(positive=[10], negative=[15, 7])
            ),
            with_payload=True,
            limit=10,
            using="image",
        ).points

    @classmethod
    def recommend_from_another_collection(
        cls, client: QdrantBase, positive_point_id: int | None = None
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
            using="image",
            lookup_from=models.LookupLocation(
                collection=secondary_collection_name,
                vector="image",
            ),
        ).points

    @classmethod
    def filter_recommend_text(
        cls, client: QdrantBase, query_filter: models.Filter
    ) -> list[models.ScoredPoint]:
        return client.query_points(
            collection_name=COLLECTION_NAME,
            query=models.RecommendQuery(recommend=models.RecommendInput(positive=[10])),
            query_filter=query_filter,
            with_payload=True,
            limit=10,
            using="text",
        ).points

    @classmethod
    def best_score_recommend(cls, client: QdrantBase) -> list[models.ScoredPoint]:
        return client.query_points(
            collection_name=COLLECTION_NAME,
            query=models.RecommendQuery(
                recommend=models.RecommendInput(
                    positive=[10, 20],
                    negative=[],
                    strategy=models.RecommendStrategy.BEST_SCORE,
                )
            ),
            with_payload=True,
            limit=10,
            using="image",
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
            using="code",
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
            using="image",
        ).points

    @classmethod
    def only_negatives_best_score_recommend_euclid(
        cls, client: QdrantBase
    ) -> list[models.ScoredPoint]:
        return client.query_points(
            collection_name=COLLECTION_NAME,
            query=models.RecommendQuery(
                recommend=models.RecommendInput(
                    positive=None,
                    negative=[10, 12],
                    strategy="best_score",  # type: ignore  # check it works with a literal
                )
            ),
            with_payload=True,
            limit=10,
            using="code",
        ).points

    @classmethod
    def sum_scores_recommend(cls, client: QdrantBase) -> list[models.ScoredPoint]:
        return client.query_points(
            collection_name=COLLECTION_NAME,
            query=models.RecommendQuery(
                recommend=models.RecommendInput(
                    positive=[10, 20], negative=[], strategy=models.RecommendStrategy.SUM_SCORES
                )
            ),
            with_payload=True,
            limit=10,
            using="image",
        ).points

    @classmethod
    def sum_scores_recommend_euclid(cls, client: QdrantBase) -> list[models.ScoredPoint]:
        return client.query_points(
            collection_name=COLLECTION_NAME,
            query=models.RecommendQuery(
                recommend=models.RecommendInput(
                    positive=[10, 20],
                    negative=[11, 21],
                    strategy=models.RecommendStrategy.SUM_SCORES,
                )
            ),
            with_payload=True,
            limit=10,
            using="code",
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
            using="image",
        ).points

    @classmethod
    def only_negatives_sum_scores_recommend_euclid(
        cls, client: QdrantBase
    ) -> list[models.ScoredPoint]:
        return client.query_points(
            collection_name=COLLECTION_NAME,
            query=models.RecommendQuery(
                recommend=models.RecommendInput(
                    positive=None, negative=[10, 12], strategy="sum_scores"
                )  # type: ignore  # check it works with a literal
            ),
            with_payload=True,
            limit=10,
            using="code",
        ).points

    @classmethod
    def avg_vector_recommend(cls, client: QdrantBase) -> list[models.ScoredPoint]:
        return client.query_points(
            collection_name=COLLECTION_NAME,
            query=models.RecommendQuery(
                recommend=models.RecommendInput(
                    positive=[10, 13],
                    negative=[],
                    strategy=models.RecommendStrategy.AVERAGE_VECTOR,
                )
            ),
            with_payload=True,
            limit=10,
            using="image",
        ).points

    def recommend_from_raw_vectors(self, client: QdrantBase) -> list[models.ScoredPoint]:
        return client.query_points(
            collection_name=COLLECTION_NAME,
            query=models.RecommendQuery(
                recommend=models.RecommendInput(positive=[self.query_image], negative=[])
            ),
            with_payload=True,
            limit=10,
            using="image",
        ).points

    def recommend_from_raw_vectors_and_ids(self, client: QdrantBase) -> list[models.ScoredPoint]:
        return client.query_points(
            collection_name=COLLECTION_NAME,
            query=models.RecommendQuery(
                recommend=models.RecommendInput(positive=[self.query_image, 10], negative=[]),
            ),
            with_payload=True,
            limit=10,
            using="image",
        ).points

    @staticmethod
    def recommend_batch(client: QdrantBase) -> list[models.QueryResponse]:
        return client.query_batch_points(
            collection_name=COLLECTION_NAME,
            requests=[
                models.QueryRequest(
                    query=models.RecommendQuery(
                        recommend=models.RecommendInput(
                            positive=[3],
                            negative=None,
                            strategy=models.RecommendStrategy.AVERAGE_VECTOR,
                        )
                    ),
                    limit=1,
                    using="image",
                ),
                models.QueryRequest(
                    query=models.RecommendQuery(
                        recommend=models.RecommendInput(
                            positive=[10],
                            negative=[],
                            strategy=models.RecommendStrategy.BEST_SCORE,
                        )
                    ),
                    limit=2,
                    using="image",
                    lookup_from=models.LookupLocation(
                        collection=secondary_collection_name,
                        vector="image",
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
                    using="image",
                ),
            ],
        )


def test_recommend_from_another_collection():
    fixture_points = generate_fixtures(10)

    secondary_collection_points = generate_fixtures(10)

    searcher = TestSimpleRecommendation()
    local_client = init_local()
    init_client(local_client, fixture_points)
    init_client(local_client, secondary_collection_points, secondary_collection_name)

    remote_client = init_remote()
    init_client(remote_client, fixture_points)
    init_client(remote_client, secondary_collection_points, secondary_collection_name)

    for i in range(10):
        compare_client_results(
            local_client,
            remote_client,
            searcher.recommend_from_another_collection,
            positive_point_id=i,
        )


def test_simple_recommend() -> None:
    fixture_points = generate_fixtures()

    secondary_collection_points = generate_fixtures(100)

    searcher = TestSimpleRecommendation()

    local_client = init_local()
    init_client(local_client, fixture_points)
    init_client(local_client, secondary_collection_points, secondary_collection_name)

    remote_client = init_remote()
    init_client(remote_client, fixture_points)
    init_client(remote_client, secondary_collection_points, secondary_collection_name)

    compare_client_results(local_client, remote_client, searcher.simple_recommend_image)
    compare_client_results(local_client, remote_client, searcher.many_recommend)
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
    compare_client_results(local_client, remote_client, searcher.sum_scores_recommend)
    compare_client_results(local_client, remote_client, searcher.sum_scores_recommend_euclid)
    compare_client_results(
        local_client, remote_client, searcher.only_negatives_sum_scores_recommend
    )
    compare_client_results(
        local_client, remote_client, searcher.only_negatives_sum_scores_recommend_euclid
    )
    compare_client_results(local_client, remote_client, searcher.avg_vector_recommend)
    compare_client_results(local_client, remote_client, searcher.recommend_from_raw_vectors)
    compare_client_results(
        local_client, remote_client, searcher.recommend_from_raw_vectors_and_ids
    )
    compare_client_results(local_client, remote_client, searcher.recommend_batch)

    for _ in range(10):
        query_filter = one_random_filter_please()
        try:
            compare_client_results(
                local_client,
                remote_client,
                searcher.filter_recommend_text,
                query_filter=query_filter,
            )
        except AssertionError as e:
            print(f"\nFailed with filter {query_filter}")
            raise e


def test_query_with_nan():
    fixture_points = generate_fixtures()
    vector = np.random.random(image_vector_size)
    vector[0] = np.nan
    vector = vector.tolist()
    using = "image"

    local_client = init_local()
    remote_client = init_remote()

    init_client(local_client, fixture_points)
    init_client(remote_client, fixture_points)

    with pytest.raises(AssertionError):
        local_client.query_points(
            collection_name=COLLECTION_NAME,
            query=models.RecommendQuery(
                recommend=models.RecommendInput(positive=[vector], negative=[])
            ),
            using=using,
        )

    with pytest.raises(UnexpectedResponse):
        remote_client.query_points(
            collection_name=COLLECTION_NAME,
            query=models.RecommendQuery(
                recommend=models.RecommendInput(positive=[vector], negative=[])
            ),
            using=using,
        )

    with pytest.raises(AssertionError):
        local_client.query_points(
            collection_name=COLLECTION_NAME,
            query=models.RecommendQuery(
                recommend=models.RecommendInput(positive=[1], negative=[vector]),
            ),
            using=using,
        )

    with pytest.raises(UnexpectedResponse):
        remote_client.query_points(
            collection_name=COLLECTION_NAME,
            query=models.RecommendQuery(
                recommend=models.RecommendInput(positive=[1], negative=[vector]),
            ),
            using=using,
        )
