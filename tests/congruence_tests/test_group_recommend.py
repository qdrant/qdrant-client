from qdrant_client.client_base import QdrantBase
from qdrant_client.http.models import models
from tests.congruence_tests.test_common import (
    COLLECTION_NAME,
    compare_client_results,
    generate_fixtures,
    init_client,
    init_local,
    init_remote,
)
from tests.fixtures.filters import one_random_filter_please

secondary_collection_name = "congruence_secondary_collection"


class TestGroupRecommendation:
    __test__ = False

    def __init__(self):
        self.group_by = "rand_digit"
        self.group_size = 1

    def simple_recommend_groups_image(self, client: QdrantBase) -> models.GroupsResult:
        return client.recommend_groups(
            collection_name=COLLECTION_NAME,
            positive=[10],
            negative=[],
            with_payload=models.PayloadSelectorExclude(exclude=["city.geo", "rand_number"]),
            limit=10,
            using="image",
            group_by=self.group_by,
            group_size=self.group_size,
            search_params=models.SearchParams(exact=True),
        )

    def simple_recommend_groups_best_scores(self, client: QdrantBase) -> models.GroupsResult:
        return client.recommend_groups(
            collection_name=COLLECTION_NAME,
            positive=[10],
            negative=[],
            with_payload=models.PayloadSelectorExclude(exclude=["city.geo", "rand_number"]),
            limit=10,
            using="image",
            strategy=models.RecommendStrategy.BEST_SCORE,
            group_by=self.group_by,
            group_size=self.group_size,
            search_params=models.SearchParams(exact=True),
        )

    def many_recommend_groups(self, client: QdrantBase) -> models.GroupsResult:
        return client.recommend_groups(
            collection_name=COLLECTION_NAME,
            positive=[10, 19],
            with_payload=models.PayloadSelectorExclude(exclude=["city.geo", "rand_number"]),
            limit=10,
            using="image",
            group_by=self.group_by,
            group_size=self.group_size,
            search_params=models.SearchParams(exact=True),
        )

    def simple_recommend_groups_negative(self, client: QdrantBase) -> models.GroupsResult:
        return client.recommend_groups(
            collection_name=COLLECTION_NAME,
            positive=[10],
            negative=[15, 7],
            with_payload=models.PayloadSelectorExclude(exclude=["city.geo", "rand_number"]),
            limit=10,
            using="image",
            group_by=self.group_by,
            group_size=self.group_size,
            search_params=models.SearchParams(exact=True),
        )

    def recommend_groups_from_another_collection(self, client: QdrantBase) -> models.GroupsResult:
        return client.recommend_groups(
            collection_name=COLLECTION_NAME,
            positive=[10],
            negative=[15, 7],
            with_payload=models.PayloadSelectorExclude(exclude=["city.geo", "rand_number"]),
            limit=10,
            using="image",
            lookup_from=models.LookupLocation(
                collection=secondary_collection_name,
                vector="image",
            ),
            group_by=self.group_by,
            group_size=self.group_size,
            search_params=models.SearchParams(exact=True),
        )

    def filter_recommend_groups_text(
        self, client: QdrantBase, query_filter: models.Filter
    ) -> models.GroupsResult:
        return client.recommend_groups(
            collection_name=COLLECTION_NAME,
            positive=[10],
            query_filter=query_filter,
            with_payload=models.PayloadSelectorExclude(exclude=["city.geo", "rand_number"]),
            limit=10,
            using="text",
            group_by=self.group_by,
            group_size=self.group_size,
            search_params=models.SearchParams(exact=True),
        )


def group_by_keys():
    return ["id", "rand_digit", "two_words", "city.name", "maybe", "maybe_null"]


def test_simple_recommend_groups() -> None:
    fixture_points = generate_fixtures()

    secondary_collection_points = generate_fixtures(100)

    recommender = TestGroupRecommendation()

    local_client = init_local()
    init_client(local_client, fixture_points)
    init_client(local_client, secondary_collection_points, secondary_collection_name)

    remote_client = init_remote()
    init_client(remote_client, fixture_points)
    init_client(remote_client, secondary_collection_points, secondary_collection_name)

    for group_size in (3, 5):
        recommender.group_size = group_size
        compare_client_results(
            local_client, remote_client, recommender.simple_recommend_groups_image
        )
        compare_client_results(
            local_client, remote_client, recommender.simple_recommend_groups_best_scores
        )
        compare_client_results(local_client, remote_client, recommender.many_recommend_groups)
        compare_client_results(
            local_client, remote_client, recommender.simple_recommend_groups_negative
        )
        compare_client_results(
            local_client,
            remote_client,
            recommender.recommend_groups_from_another_collection,
        )

    for key in group_by_keys():
        recommender.group_by = key
        compare_client_results(
            local_client, remote_client, recommender.simple_recommend_groups_image
        )
        compare_client_results(local_client, remote_client, recommender.many_recommend_groups)
        compare_client_results(
            local_client, remote_client, recommender.simple_recommend_groups_negative
        )
        compare_client_results(
            local_client,
            remote_client,
            recommender.recommend_groups_from_another_collection,
        )

    recommender.group_by = "rand_digit"

    for i in range(10):
        query_filter = one_random_filter_please()
        try:
            compare_client_results(
                local_client,
                remote_client,
                recommender.filter_recommend_groups_text,
                query_filter=query_filter,
            )
        except AssertionError as e:
            print(f"\nFailed with filter {query_filter}")
            raise e
