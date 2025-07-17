import numpy as np
import pytest

from qdrant_client.client_base import QdrantBase
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.http.models import models
from tests.congruence_tests.test_common import (
    COLLECTION_NAME,
    code_vector_size,
    compare_client_results,
    image_vector_size,
    init_client,
    init_local,
    init_remote,
    text_vector_size,
    multi_vector_config,
    generate_multivector_fixtures,
)
from tests.fixtures.points import generate_random_multivector


class TestSimpleSearcher:
    __test__ = False

    def __init__(self):
        self.query_text = generate_random_multivector(text_vector_size, 10)
        self.query_image = generate_random_multivector(image_vector_size, 10)
        self.query_code = generate_random_multivector(code_vector_size, 10)

    def simple_search_text(self, client: QdrantBase) -> list[models.ScoredPoint]:
        return client.query_points(
            collection_name=COLLECTION_NAME,
            query=self.query_text,
            with_payload=True,
            using="multi-text",
            limit=10,
        ).points

    def simple_search_image(self, client: QdrantBase) -> list[models.ScoredPoint]:
        return client.query_points(
            collection_name=COLLECTION_NAME,
            query=self.query_image,
            using="multi-image",
            with_payload=True,
            limit=10,
        ).points

    def simple_search_code(self, client: QdrantBase) -> list[models.ScoredPoint]:
        return client.query_points(
            collection_name=COLLECTION_NAME,
            query=self.query_code,
            using="multi-code",
            with_payload=True,
            limit=10,
        ).points

    def simple_search_unnamed(self, client: QdrantBase) -> list[models.ScoredPoint]:
        return client.query_points(
            collection_name=COLLECTION_NAME,
            query=self.query_text,
            with_payload=True,
            limit=10,
        ).points

    def default_mmr_query(self, client: QdrantBase) -> models.QueryResponse:
        return client.query_points(
            collection_name=COLLECTION_NAME,
            query=models.NearestQuery(
                nearest=self.query_text,
                mmr=models.Mmr(),
            ),
            using="multi-text",
            limit=10,
        )

    def mmr_query_parametrized(self, client: QdrantBase) -> models.QueryResponse:
        return client.query_points(
            collection_name=COLLECTION_NAME,
            query=models.NearestQuery(
                nearest=self.query_text, mmr=models.Mmr(diversity=0.3, candidates_limit=30)
            ),
            using="multi-text",
            limit=10,
        )

    def mmr_query_parametrized_score_threshold(self, client: QdrantBase) -> models.QueryResponse:
        return client.query_points(
            collection_name=COLLECTION_NAME,
            query=models.NearestQuery(
                nearest=self.query_text, mmr=models.Mmr(diversity=0.3, candidates_limit=30)
            ),
            using="multi-text",
            score_threshold=7.0,
            limit=10,
        )

    def mmr_query_parametrized_dot(self, client: QdrantBase) -> models.QueryResponse:
        return client.query_points(
            collection_name=COLLECTION_NAME,
            query=models.NearestQuery(
                nearest=self.query_image, mmr=models.Mmr(diversity=0.3, candidates_limit=30)
            ),
            using="multi-image",
            limit=10,
        )

    def mmr_query_parametrized_dot_score_threshold(
        self, client: QdrantBase
    ) -> models.QueryResponse:
        return client.query_points(
            collection_name=COLLECTION_NAME,
            query=models.NearestQuery(
                nearest=self.query_image, mmr=models.Mmr(diversity=0.3, candidates_limit=30)
            ),
            score_threshold=260.0,
            using="multi-image",
            limit=10,
        )

    def mmr_query_parametrized_euclid(self, client: QdrantBase) -> models.QueryResponse:
        return client.query_points(
            collection_name=COLLECTION_NAME,
            query=models.NearestQuery(
                nearest=self.query_code, mmr=models.Mmr(diversity=0.3, candidates_limit=30)
            ),
            using="multi-code",
            limit=10,
        )

    def mmr_query_parametrized_euclid_score_threshold(
        self, client: QdrantBase
    ) -> models.QueryResponse:
        return client.query_points(
            collection_name=COLLECTION_NAME,
            query=models.NearestQuery(
                nearest=self.query_code, mmr=models.Mmr(diversity=0.3, candidates_limit=30)
            ),
            using="multi-code",
            score_threshold=-100.0,
            limit=10,
        )


def test_simple():
    fixture_points = generate_multivector_fixtures(100)

    searcher = TestSimpleSearcher()

    local_client = init_local()
    init_client(local_client, fixture_points, vectors_config=multi_vector_config)

    remote_client = init_remote()
    init_client(remote_client, fixture_points, vectors_config=multi_vector_config)

    compare_client_results(local_client, remote_client, searcher.simple_search_text)
    compare_client_results(local_client, remote_client, searcher.simple_search_image)
    compare_client_results(local_client, remote_client, searcher.simple_search_code)


def test_mmr():
    fixture_points = generate_multivector_fixtures(10)

    searcher = TestSimpleSearcher()

    local_client = init_local()
    init_client(local_client, fixture_points, vectors_config=multi_vector_config)

    remote_client = init_remote()
    init_client(remote_client, fixture_points, vectors_config=multi_vector_config)

    compare_client_results(local_client, remote_client, searcher.default_mmr_query)
    compare_client_results(local_client, remote_client, searcher.mmr_query_parametrized)
    compare_client_results(
        local_client, remote_client, searcher.mmr_query_parametrized_score_threshold
    )
    compare_client_results(local_client, remote_client, searcher.mmr_query_parametrized_dot)
    compare_client_results(
        local_client, remote_client, searcher.mmr_query_parametrized_dot_score_threshold
    )
    compare_client_results(local_client, remote_client, searcher.mmr_query_parametrized_euclid)
    compare_client_results(
        local_client, remote_client, searcher.mmr_query_parametrized_euclid_score_threshold
    )


def test_single_vector():
    fixture_points = generate_multivector_fixtures(num=100, vectors_sizes=text_vector_size)

    searcher = TestSimpleSearcher()

    vectors_config = models.VectorParams(
        size=text_vector_size,
        distance=models.Distance.DOT,
        multivector_config=models.MultiVectorConfig(
            comparator=models.MultiVectorComparator.MAX_SIM
        ),
    )

    local_client = init_local()
    init_client(local_client, fixture_points, vectors_config=vectors_config)

    remote_client = init_remote()
    init_client(remote_client, fixture_points, vectors_config=vectors_config)

    compare_client_results(local_client, remote_client, searcher.simple_search_unnamed)


def test_search_with_persistence():
    import tempfile

    fixture_points = generate_multivector_fixtures(100)
    searcher = TestSimpleSearcher()
    with tempfile.TemporaryDirectory() as tmpdir:
        local_client = init_local(tmpdir)
        init_client(local_client, fixture_points, vectors_config=multi_vector_config)

        del local_client
        local_client_2 = init_local(tmpdir)

        remote_client = init_remote()
        init_client(remote_client, fixture_points, vectors_config=multi_vector_config)

        compare_client_results(local_client_2, remote_client, searcher.simple_search_text)
        compare_client_results(local_client_2, remote_client, searcher.simple_search_image)
        compare_client_results(local_client_2, remote_client, searcher.simple_search_code)


def test_search_invalid_vector_type():
    fixture_points = generate_multivector_fixtures(100)

    local_client = init_local()
    init_client(local_client, fixture_points, vectors_config=multi_vector_config)

    remote_client = init_remote()
    init_client(remote_client, fixture_points, vectors_config=multi_vector_config)

    vector_invalid_type = {"multi-text": [1, 2, 3, 4]}
    with pytest.raises(ValueError):
        local_client.search(collection_name=COLLECTION_NAME, query_vector=vector_invalid_type)

    with pytest.raises(ValueError):
        remote_client.search(collection_name=COLLECTION_NAME, query_vector=vector_invalid_type)


def test_query_with_nan():
    fixture_points = generate_multivector_fixtures(10)

    local_client = init_local()
    init_client(local_client, fixture_points, vectors_config=multi_vector_config)

    remote_client = init_remote()
    init_client(remote_client, fixture_points, vectors_config=multi_vector_config)

    vector = generate_random_multivector(text_vector_size, 10)
    vector[0][4] = np.nan
    with pytest.raises(AssertionError):
        local_client.query_points(COLLECTION_NAME, query=vector, using="multi-text")
    with pytest.raises(UnexpectedResponse):
        remote_client.query_points(COLLECTION_NAME, query=vector, using="multi-text")

    single_multi_vector_config = models.VectorParams(
        size=text_vector_size,
        distance=models.Distance.COSINE,
        multivector_config=models.MultiVectorConfig(
            comparator=models.MultiVectorComparator.MAX_SIM
        ),
    )

    local_client.delete_collection(COLLECTION_NAME)
    local_client.create_collection(COLLECTION_NAME, vectors_config=single_multi_vector_config)

    remote_client.delete_collection(COLLECTION_NAME)
    remote_client.create_collection(COLLECTION_NAME, vectors_config=single_multi_vector_config)

    fixture_points = generate_multivector_fixtures(vectors_sizes=text_vector_size)
    init_client(local_client, fixture_points, vectors_config=single_multi_vector_config)
    init_client(remote_client, fixture_points, vectors_config=single_multi_vector_config)

    with pytest.raises(AssertionError):
        local_client.query_points(COLLECTION_NAME, query=vector)
    with pytest.raises(UnexpectedResponse):
        remote_client.query_points(COLLECTION_NAME, query=vector)
