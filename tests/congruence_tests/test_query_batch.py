from typing import List, Union

import numpy as np
import pytest

from qdrant_client.client_base import QdrantBase
from qdrant_client.http.exceptions import UnexpectedResponse
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
    sparse_image_vector_size,
    sparse_code_vector_size,
    generate_sparse_fixtures,
    sparse_vectors_config,
    generate_multivector_fixtures,
    multi_vector_config,
)
from tests.fixtures.filters import one_random_filter_please
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
            self.dense_vector_query_batch_text.append(np.random.random(text_vector_size).tolist())
            self.dense_vector_query_batch_image.append(
                np.random.random(image_vector_size).tolist()
            )
            self.dense_vector_query_batch_code.append(np.random.random(code_vector_size).tolist())

            self.sparse_vector_query_batch_text.append(
                generate_random_sparse_vector(sparse_text_vector_size, density=0.3)
            )
            self.sparse_vector_query_batch_image.append(
                generate_random_sparse_vector(sparse_image_vector_size, density=0.2)
            )
            self.sparse_vector_query_batch_code.append(
                generate_random_sparse_vector(sparse_code_vector_size, density=0.1)
            )

            self.multivector_query_batch_text.append(
                generate_random_multivector(text_vector_size, 3)
            )
            self.multivector_query_batch_image.append(
                generate_random_multivector(text_vector_size, 3)
            )
            self.multivector_query_batch_code.append(
                generate_random_multivector(text_vector_size, 3)
            )

    def sparse_query_batch_text(
        self, client: QdrantBase
    ) -> Union[List[models.ScoredPoint], models.QueryResponse]:
        return client.query_batch_points(
            collection_name=COLLECTION_NAME, requests=self.sparse_vector_query_batch_text
        )

    def multivec_query_batch_text(
        self, client: QdrantBase
    ) -> Union[List[models.ScoredPoint], models.QueryResponse]:
        return client.query_batch_points(
            collection_name=COLLECTION_NAME, requests=self.multivector_query_batch_text
        )


# ---- TESTS  ---- #


def test_sparse_query():
    fixture_points = generate_sparse_fixtures()

    searcher = TestQueryBatchSearcher()

    local_client = init_local()
    init_client(local_client, fixture_points, sparse_vectors_config=sparse_vectors_config)

    remote_client = init_remote()
    init_client(remote_client, fixture_points, sparse_vectors_config=sparse_vectors_config)

    compare_client_results(local_client, remote_client, searcher.sparse_query_batch_text)
