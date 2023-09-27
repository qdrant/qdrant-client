from curses import raw
from typing import List

import numpy as np

from qdrant_client.client_base import QdrantBase
from qdrant_client.http.models import models
from tests.congruence_tests.test_common import (
    COLLECTION_NAME,
    compare_client_results,
    generate_fixtures,
    init_client,
    init_local,
    init_remote,
    image_vector_size,
)
from tests.fixtures.filters import one_random_filter_please

secondary_collection_name = "secondary_collection"

class TestSimpleRecommendation:
    
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
    def recommend_from_another_collection(cls, client: QdrantBase) -> List[models.ScoredPoint]:
        return client.recommend(
            collection_name=COLLECTION_NAME,
            positive=[10],
            negative=[15, 7],
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
            positive=[10, 20,],
            negative=[],
            with_payload=True,
            limit=10,
            using="image",
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


def test_simple_recommend() -> None:
    fixture_records = generate_fixtures()

    secondary_collection_records = generate_fixtures(100)

    searcher = TestSimpleRecommendation()

    local_client = init_local()
    init_client(local_client, fixture_records)
    init_client(local_client, secondary_collection_records, secondary_collection_name)

    remote_client = init_remote()
    init_client(remote_client, fixture_records)
    init_client(remote_client, secondary_collection_records, secondary_collection_name)

    compare_client_results(local_client, remote_client, searcher.simple_recommend_image)
    compare_client_results(local_client, remote_client, searcher.many_recommend)
    compare_client_results(local_client, remote_client, searcher.simple_recommend_negative)
    compare_client_results(local_client, remote_client, searcher.recommend_from_another_collection)
    compare_client_results(local_client, remote_client, searcher.best_score_recommend)
    compare_client_results(local_client, remote_client, searcher.only_negatives_best_score_recommend)
    compare_client_results(local_client, remote_client, searcher.avg_vector_recommend)
    compare_client_results(local_client, remote_client, searcher.recommend_from_raw_vectors)
    compare_client_results(local_client, remote_client, searcher.recommend_from_raw_vectors_and_ids)

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
