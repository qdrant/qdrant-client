import numpy as np
import pytest

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
    sparse_text_vector_size,
    generate_sparse_fixtures,
    sparse_vectors_config,
    generate_multivector_fixtures,
    multi_vector_config,
)
from tests.fixtures.points import generate_random_sparse_vector, generate_random_multivector


class TestQueryBatchSearcher:
    __test__ = False

    def __init__(self):
        self.dense_vector_query_batch_text = []
        self.dense_vector_query_batch_image = []
        self.dense_vector_query_batch_code = []

        self.sparse_vector_query_batch_text = []
        self.sparse_vector_query_batch_image = []
        self.sparse_vector_query_batch_code = []

        self.multivector_query_batch_text = []
        self.multivector_query_batch_image = []
        self.multivector_query_batch_code = []

        for _ in range(4):
            self.dense_vector_query_batch_text.append(
                models.QueryRequest(
                    query=np.random.random(text_vector_size).tolist(),
                    prefetch=models.Prefetch(
                        query=np.random.random(text_vector_size).tolist(), limit=5, using="text"
                    ),
                    limit=5,
                    using="text",
                    with_payload=True,
                )
            )
            self.dense_vector_query_batch_image.append(
                models.QueryRequest(
                    query=np.random.random(image_vector_size).tolist(),
                    limit=5,
                    using="image",
                    with_payload=True,
                )
            )
            self.dense_vector_query_batch_code.append(
                models.QueryRequest(
                    query=np.random.random(code_vector_size).tolist(),
                    limit=5,
                    using="code",
                    with_payload=True,
                )
            )

            self.sparse_vector_query_batch_text.append(
                models.QueryRequest(
                    query=generate_random_sparse_vector(sparse_text_vector_size, density=0.3),
                    limit=5,
                    using="sparse-text",
                    with_payload=True,
                )
            )

            self.multivector_query_batch_text.append(
                models.QueryRequest(
                    query=generate_random_multivector(text_vector_size, 3),
                    limit=5,
                    using="multi-text",
                    with_payload=True,
                )
            )
            self.multivector_query_batch_image.append(
                models.QueryRequest(
                    query=generate_random_multivector(text_vector_size, 3),
                    limit=5,
                    using="multi-image",
                    with_payload=True,
                )
            )
            self.multivector_query_batch_code.append(
                models.QueryRequest(
                    query=generate_random_multivector(text_vector_size, 3),
                    limit=5,
                    using="multi-code",
                    with_payload=True,
                )
            )

    def sparse_query_batch_text(
        self, client: QdrantBase, collection_name: str = COLLECTION_NAME
    ) -> models.QueryResponse:
        return client.query_batch_points(
            collection_name=collection_name, requests=self.sparse_vector_query_batch_text
        )

    def multivec_query_batch_text(
        self, client: QdrantBase, collection_name: str = COLLECTION_NAME
    ) -> models.QueryResponse:
        return client.query_batch_points(
            collection_name=collection_name, requests=self.multivector_query_batch_text
        )

    def dense_query_batch_text(
        self, client: QdrantBase, collection_name: str = COLLECTION_NAME
    ) -> models.QueryResponse:
        return client.query_batch_points(
            collection_name=collection_name, requests=self.dense_vector_query_batch_text
        )

    def dense_query_batch_image(
        self, client: QdrantBase, collection_name: str = COLLECTION_NAME
    ) -> models.QueryResponse:
        return client.query_batch_points(
            collection_name=collection_name, requests=self.dense_vector_query_batch_image
        )

    def dense_query_batch_code(
        self, client: QdrantBase, collection_name: str = COLLECTION_NAME
    ) -> models.QueryResponse:
        return client.query_batch_points(
            collection_name=collection_name, requests=self.dense_vector_query_batch_code
        )

    @staticmethod
    def dense_query_batch_empty(
        client: QdrantBase, collection_name: str = COLLECTION_NAME
    ) -> models.QueryResponse:
        return client.query_batch_points(collection_name=collection_name, requests=[])


# ---- TESTS  ---- #


def test_sparse_query_batch(collection_name: str):
    fixture_points = generate_sparse_fixtures()

    searcher = TestQueryBatchSearcher()

    local_client = init_local()
    init_client(
        local_client, fixture_points, collection_name, sparse_vectors_config=sparse_vectors_config
    )

    remote_client = init_remote()
    init_client(
        remote_client, fixture_points, collection_name, sparse_vectors_config=sparse_vectors_config
    )

    compare_client_results(
        local_client,
        remote_client,
        searcher.sparse_query_batch_text,
        collection_name=collection_name,
    )


def test_multivec_query_batch(collection_name: str):
    fixture_points = generate_multivector_fixtures()

    searcher = TestQueryBatchSearcher()

    local_client = init_local()
    init_client(local_client, fixture_points, collection_name, vectors_config=multi_vector_config)

    remote_client = init_remote()
    init_client(remote_client, fixture_points, collection_name, vectors_config=multi_vector_config)

    compare_client_results(
        local_client,
        remote_client,
        searcher.multivec_query_batch_text,
        collection_name=collection_name,
    )


@pytest.mark.parametrize("prefer_grpc", (False, True))
def test_dense_query_batch(prefer_grpc: bool, collection_name: str):
    fixture_points = generate_fixtures()

    searcher = TestQueryBatchSearcher()

    local_client = init_local()
    init_client(local_client, fixture_points, collection_name)

    remote_client = init_remote(prefer_grpc=prefer_grpc)
    init_client(remote_client, fixture_points, collection_name)

    compare_client_results(
        local_client,
        remote_client,
        searcher.dense_query_batch_text,
        collection_name=collection_name,
    )
    compare_client_results(
        local_client,
        remote_client,
        searcher.dense_query_batch_image,
        collection_name=collection_name,
    )
    compare_client_results(
        local_client,
        remote_client,
        searcher.dense_query_batch_code,
        collection_name=collection_name,
    )
    compare_client_results(
        local_client,
        remote_client,
        searcher.dense_query_batch_empty,
        collection_name=collection_name,
    )
