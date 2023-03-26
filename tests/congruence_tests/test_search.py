import numpy as np

from qdrant_client.client_base import QdrantBase
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
)


class TestSimpleSearcher:
    def __init__(self):
        self.query_text = np.random.random(text_vector_size).tolist()
        self.query_image = np.random.random(image_vector_size).tolist()
        self.query_code = np.random.random(code_vector_size).tolist()

    def simple_search_text(self, client: QdrantBase):
        return client.search(
            collection_name=COLLECTION_NAME,
            query_vector=("text", self.query_text),
            with_payload=True,
            limit=10,
        )

    def simple_search_image(self, client: QdrantBase):
        return client.search(
            collection_name=COLLECTION_NAME,
            query_vector=("image", self.query_image),
            with_payload=True,
            limit=10,
        )

    def simple_search_code(self, client: QdrantBase):
        return client.search(
            collection_name=COLLECTION_NAME,
            query_vector=("code", self.query_code),
            with_payload=True,
            limit=10,
        )

    def simple_search_text_offset(self, client: QdrantBase):
        return client.search(
            collection_name=COLLECTION_NAME,
            query_vector=("text", self.query_text),
            with_payload=True,
            limit=10,
            offset=10,
        )

    def search_score_threshold(self, client: QdrantBase):
        res1 = client.search(
            collection_name=COLLECTION_NAME,
            query_vector=("text", self.query_text),
            with_payload=True,
            limit=10,
            score_threshold=0.9,
        )

        res2 = client.search(
            collection_name=COLLECTION_NAME,
            query_vector=("text", self.query_text),
            with_payload=True,
            limit=10,
            score_threshold=0.95,
        )

        res3 = client.search(
            collection_name=COLLECTION_NAME,
            query_vector=("text", self.query_text),
            with_payload=True,
            limit=10,
            score_threshold=0.1,
        )

        return res1 + res2 + res3

    def simple_search_text_select_payload(self, client: QdrantBase):
        return client.search(
            collection_name=COLLECTION_NAME,
            query_vector=("text", self.query_text),
            with_payload=["text_array", "nested.id"],
            limit=10,
        )

    def search_payload_exclude(self, client: QdrantBase):
        return client.search(
            collection_name=COLLECTION_NAME,
            query_vector=("text", self.query_text),
            with_payload=models.PayloadSelectorExclude(exclude=["text_array", "nested.id"]),
            limit=10,
        )

    def simple_search_image_select_vector(self, client: QdrantBase):
        return client.search(
            collection_name=COLLECTION_NAME,
            query_vector=("image", self.query_image),
            with_payload=False,
            with_vectors=["image", "code"],
            limit=10,
        )


def test_simple_search():
    fixture_records = generate_fixtures()

    searcher = TestSimpleSearcher()

    local_client = init_local()
    init_client(local_client, fixture_records)

    remote_client = init_remote()
    init_client(remote_client, fixture_records)

    compare_client_results(local_client, remote_client, searcher.simple_search_text)
    compare_client_results(local_client, remote_client, searcher.simple_search_image)
    compare_client_results(local_client, remote_client, searcher.simple_search_code)
    compare_client_results(local_client, remote_client, searcher.simple_search_text_offset)
    compare_client_results(local_client, remote_client, searcher.search_score_threshold)
    compare_client_results(local_client, remote_client, searcher.simple_search_text_select_payload)
    compare_client_results(local_client, remote_client, searcher.simple_search_image_select_vector)
    compare_client_results(local_client, remote_client, searcher.search_payload_exclude)
