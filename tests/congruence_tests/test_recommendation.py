from typing import List, Optional

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
    def simple_recommend_image(cls, client: QdrantBase) -> List[models.ScoredPoint]:
        return client.recommend(
            collection_name=COLLECTION_NAME,
            positive=[10],
            negative=[],
            with_payload=True,
            limit=10,
            using="image",
        )

    @classmethod
    def many_recommend(cls, client: QdrantBase) -> List[models.ScoredPoint]:
        return client.recommend(
            collection_name=COLLECTION_NAME,
            positive=[10, 19],
            with_payload=True,
            limit=10,
            using="image",
        )

    @classmethod
    def simple_recommend_negative(cls, client: QdrantBase) -> List[models.ScoredPoint]:
        return client.recommend(
            collection_name=COLLECTION_NAME,
            positive=[10],
            negative=[15, 7],
            with_payload=True,
            limit=10,
            using="image",
        )

    @classmethod
    def recommend_from_another_collection(
        cls, client: QdrantBase, positive_point_id: Optional[int] = None
    ) -> List[models.ScoredPoint]:
        return client.recommend(
            collection_name=COLLECTION_NAME,
            positive=[10] if positive_point_id is None else [positive_point_id],
            negative=[15, 7] if positive_point_id is None else [],
            with_payload=True,
            limit=10,
            using="image",
            lookup_from=models.LookupLocation(
                collection=secondary_collection_name,
                vector="image",
            ),
        )

    @classmethod
    def filter_recommend_text(
        cls, client: QdrantBase, query_filter: models.Filter
    ) -> List[models.ScoredPoint]:
        return client.recommend(
            collection_name=COLLECTION_NAME,
            positive=[10],
            query_filter=query_filter,
            with_payload=True,
            limit=10,
            using="text",
        )

    @classmethod
    def best_score_recommend(cls, client: QdrantBase) -> List[models.ScoredPoint]:
        return client.recommend(
            collection_name=COLLECTION_NAME,
            positive=[
                10,
                20,
            ],
            negative=[],
            with_payload=True,
            limit=10,
            using="image",
            strategy=models.RecommendStrategy.BEST_SCORE,
        )

    @classmethod
    def best_score_recommend_euclid(cls, client: QdrantBase) -> List[models.ScoredPoint]:
        return client.recommend(
            collection_name=COLLECTION_NAME,
            positive=[
                10,
                20,
            ],
            negative=[11, 21],
            with_payload=True,
            limit=10,
            using="code",
            strategy=models.RecommendStrategy.BEST_SCORE,
        )

    @classmethod
    def only_negatives_best_score_recommend(cls, client: QdrantBase) -> List[models.ScoredPoint]:
        return client.recommend(
            collection_name=COLLECTION_NAME,
            positive=None,
            negative=[10, 12],
            with_payload=True,
            limit=10,
            using="image",
            strategy=models.RecommendStrategy.BEST_SCORE,
        )

    @classmethod
    def only_negatives_best_score_recommend_euclid(
        cls, client: QdrantBase
    ) -> List[models.ScoredPoint]:
        return client.recommend(
            collection_name=COLLECTION_NAME,
            positive=None,
            negative=[10, 12],
            with_payload=True,
            limit=10,
            using="code",
            strategy="best_score",  # type: ignore  # check it works with a literal
        )

    @classmethod
    def avg_vector_recommend(cls, client: QdrantBase) -> List[models.ScoredPoint]:
        return client.recommend(
            collection_name=COLLECTION_NAME,
            positive=[10, 13],
            negative=[],
            with_payload=True,
            limit=10,
            using="image",
            strategy=models.RecommendStrategy.AVERAGE_VECTOR,
        )

    def recommend_from_raw_vectors(self, client: QdrantBase) -> List[models.ScoredPoint]:
        return client.recommend(
            collection_name=COLLECTION_NAME,
            positive=[self.query_image],
            negative=[],
            with_payload=True,
            limit=10,
            using="image",
        )

    def recommend_from_raw_vectors_and_ids(self, client: QdrantBase) -> List[models.ScoredPoint]:
        return client.recommend(
            collection_name=COLLECTION_NAME,
            positive=[self.query_image, 10],
            negative=[],
            with_payload=True,
            limit=10,
            using="image",
        )

    @staticmethod
    def recommend_batch(client: QdrantBase) -> List[List[models.ScoredPoint]]:
        return client.recommend_batch(
            collection_name=COLLECTION_NAME,
            requests=[
                models.RecommendRequest(
                    positive=[3],
                    negative=[],
                    limit=1,
                    using="image",
                    strategy=models.RecommendStrategy.AVERAGE_VECTOR,
                ),
                models.RecommendRequest(
                    positive=[10],
                    negative=[],
                    limit=2,
                    using="image",
                    strategy=models.RecommendStrategy.BEST_SCORE,
                    lookup_from=models.LookupLocation(
                        collection=secondary_collection_name,
                        vector="image",
                    ),
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
        local_client.recommend(
            collection_name=COLLECTION_NAME,
            positive=[vector],
            negative=[],
            using=using,
        )

    with pytest.raises(UnexpectedResponse):
        remote_client.recommend(
            collection_name=COLLECTION_NAME,
            positive=[vector],
            negative=[],
            using=using,
        )

    with pytest.raises(AssertionError):
        local_client.recommend(
            collection_name=COLLECTION_NAME,
            positive=[1],
            negative=[vector],
            using=using,
        )

    with pytest.raises(UnexpectedResponse):
        remote_client.recommend(
            collection_name=COLLECTION_NAME,
            positive=[1],
            negative=[vector],
            using=using,
        )
