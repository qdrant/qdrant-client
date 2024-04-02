from typing import List

import numpy as np
import pytest

from qdrant_client.client_base import QdrantBase
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.http.models import models
from tests.congruence_tests.test_common import (
    COLLECTION_NAME,
    compare_client_results,
    generate_fixtures,
    generate_sparse_fixtures,
    image_vector_size,
    init_client,
    init_local,
    init_remote,
    sparse_image_vector_size,
    sparse_vectors_config,
)
from tests.fixtures.filters import one_random_filter_please
from tests.fixtures.points import random_sparse_vectors

secondary_collection_name = "congruence_secondary_collection"


class TestSimpleRecommendation:
    __test__ = False

    def __init__(self):
        self.query_image = random_sparse_vectors({"sparse-image": sparse_image_vector_size})[
            "sparse-image"
        ]

    @classmethod
    def simple_recommend_image(cls, client: QdrantBase) -> List[models.ScoredPoint]:
        return client.recommend(
            collection_name=COLLECTION_NAME,
            positive=[10],
            negative=[],
            with_payload=True,
            limit=10,
            using="sparse-image",
        )

    @classmethod
    def many_recommend(cls, client: QdrantBase) -> List[models.ScoredPoint]:
        return client.recommend(
            collection_name=COLLECTION_NAME,
            positive=[10, 19],
            with_payload=True,
            limit=10,
            using="sparse-image",
        )

    @classmethod
    def simple_recommend_negative(cls, client: QdrantBase) -> List[models.ScoredPoint]:
        return client.recommend(
            collection_name=COLLECTION_NAME,
            positive=[10],
            negative=[15, 7],
            with_payload=True,
            limit=10,
            using="sparse-image",
        )

    @classmethod
    def recommend_from_another_collection(cls, client: QdrantBase) -> List[models.ScoredPoint]:
        return client.recommend(
            collection_name=COLLECTION_NAME,
            positive=[10],
            negative=[15, 7],
            with_payload=True,
            limit=10,
            using="sparse-image",
            lookup_from=models.LookupLocation(
                collection=secondary_collection_name,
                vector="sparse-image",
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
            using="sparse-text",
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
            using="sparse-image",
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
            using="sparse-code",
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
            using="sparse-image",
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
            using="sparse-code",
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
            using="sparse-image",
            strategy=models.RecommendStrategy.AVERAGE_VECTOR,
        )

    def recommend_from_raw_vectors(self, client: QdrantBase) -> List[models.ScoredPoint]:
        return client.recommend(
            collection_name=COLLECTION_NAME,
            positive=[self.query_image],
            negative=[],
            with_payload=True,
            limit=10,
            using="sparse-image",
        )

    def recommend_from_raw_vectors_and_ids(self, client: QdrantBase) -> List[models.ScoredPoint]:
        return client.recommend(
            collection_name=COLLECTION_NAME,
            positive=[self.query_image, 10],
            negative=[],
            with_payload=True,
            limit=10,
            using="sparse-image",
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
                    using="sparse-image",
                    strategy=models.RecommendStrategy.AVERAGE_VECTOR,
                ),
                models.RecommendRequest(
                    positive=[10],
                    negative=[],
                    limit=2,
                    using="sparse-image",
                    strategy=models.RecommendStrategy.BEST_SCORE,
                    lookup_from=models.LookupLocation(
                        collection=secondary_collection_name,
                        vector="sparse-image",
                    ),
                ),
            ],
        )


def test_simple_recommend() -> None:
    fixture_points = generate_sparse_fixtures()

    secondary_collection_points = generate_sparse_fixtures(100)

    searcher = TestSimpleRecommendation()

    local_client = init_local()
    init_client(
        local_client,
        fixture_points,
        vectors_config={},
        sparse_vectors_config=sparse_vectors_config,
    )
    init_client(
        local_client,
        secondary_collection_points,
        secondary_collection_name,
        vectors_config={},
        sparse_vectors_config=sparse_vectors_config,
    )

    remote_client = init_remote()
    init_client(
        remote_client,
        fixture_points,
        vectors_config={},
        sparse_vectors_config=sparse_vectors_config,
    )
    init_client(
        remote_client,
        secondary_collection_points,
        secondary_collection_name,
        vectors_config={},
        sparse_vectors_config=sparse_vectors_config,
    )

    compare_client_results(local_client, remote_client, searcher.simple_recommend_image)
    compare_client_results(local_client, remote_client, searcher.many_recommend)
    compare_client_results(local_client, remote_client, searcher.simple_recommend_negative)
    compare_client_results(local_client, remote_client, searcher.recommend_from_another_collection)
    compare_client_results(local_client, remote_client, searcher.best_score_recommend)
    compare_client_results(local_client, remote_client, searcher.best_score_recommend_euclid)
    compare_client_results(
        local_client, remote_client, searcher.only_negatives_best_score_recommend
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
    fixture_points = generate_sparse_fixtures()
    sparse_vector_dict = random_sparse_vectors({"sparse-image": sparse_image_vector_size})
    sparse_vector = sparse_vector_dict["sparse-image"]
    sparse_vector.values[0] = np.nan
    using = "sparse-image"

    local_client = init_local()
    remote_client = init_remote()

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
        local_client.recommend(
            collection_name=COLLECTION_NAME,
            positive=[sparse_vector],
            negative=[],
            using=using,
        )

    with pytest.raises(UnexpectedResponse):
        remote_client.recommend(
            collection_name=COLLECTION_NAME,
            positive=[sparse_vector],
            negative=[],
            using=using,
        )

    with pytest.raises(AssertionError):
        local_client.recommend(
            collection_name=COLLECTION_NAME,
            positive=[1],
            negative=[sparse_vector],
            using=using,
        )

    with pytest.raises(UnexpectedResponse):
        remote_client.recommend(
            collection_name=COLLECTION_NAME,
            positive=[1],
            negative=[sparse_vector],
            using=using,
        )
