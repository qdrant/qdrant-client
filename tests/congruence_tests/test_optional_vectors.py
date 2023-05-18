from typing import List

import numpy as np

from qdrant_client.client_base import QdrantBase
from qdrant_client.http.models import models
from tests.congruence_tests.test_common import (
    COLLECTION_NAME,
    NUM_VECTORS,
    code_vector_size,
    compare_client_results,
    generate_fixtures,
    image_vector_size,
    init_client,
    init_local,
    init_remote,
    text_vector_size,
)
from tests.fixtures.filters import one_random_filter_please


def test_simple_opt_vectors_search():
    fixture_records = generate_fixtures()

    local_client = init_local()
    init_client(local_client, fixture_records)

    remote_client = init_remote()
    init_client(remote_client, fixture_records)

    ids_to_delete = [x for x in range(NUM_VECTORS) if x % 5 == 0]

    vectors_to_retrieve = [x for x in range(20)]

    local_client.delete_vectors(
        collection_name=COLLECTION_NAME,
        vectors=["image"],
        points_selector=ids_to_delete,
    )
    remote_client.delete_vectors(
        collection_name=COLLECTION_NAME,
        vectors=["image"],
        points_selector=ids_to_delete,
    )

    print("--------------------")

    res = local_client.retrieve(
        collection_name=COLLECTION_NAME,
        ids=[1, 2, 3, 4, 5, 6],
        with_payload=False,
        with_vectors=True,
    )

    for point in res:
        print(point.vector.keys())

    print("--------------------")

    res = remote_client.retrieve(
        collection_name=COLLECTION_NAME,
        ids=[1, 2, 3, 4, 5, 6],
        with_payload=False,
        with_vectors=True,
    )

    for point in res:
        print(point.vector.keys())

    compare_client_results(
        local_client,
        remote_client,
        lambda c: c.retrieve(
            COLLECTION_NAME, vectors_to_retrieve, with_payload=False, with_vectors=True
        ),
    )
