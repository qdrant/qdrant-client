import numpy as np

from qdrant_client.http.models import models
from tests.congruence_tests.test_common import (
    COLLECTION_NAME,
    NUM_VECTORS,
    compare_client_results,
    generate_fixtures,
    image_vector_size,
    init_client,
    init_local,
    init_remote,
)


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

    compare_client_results(
        local_client,
        remote_client,
        lambda c: sorted(
            c.retrieve(
                COLLECTION_NAME,
                vectors_to_retrieve,
                with_payload=False,
                with_vectors=["image", "code"],
            ),
            key=lambda x: x.id,
        ),
    )

    new_vector = np.random.rand(image_vector_size).tolist()
    update_vectors = [
        models.PointVectors(
            id=i,
            vector={"image": new_vector},
        )
        for i in range(6)
    ]

    local_client.update_vectors(
        collection_name=COLLECTION_NAME,
        vectors=update_vectors,
    )

    remote_client.update_vectors(
        collection_name=COLLECTION_NAME,
        vectors=update_vectors,
    )

    compare_client_results(
        local_client,
        remote_client,
        lambda c: sorted(
            c.retrieve(
                COLLECTION_NAME,
                vectors_to_retrieve,
                with_payload=False,
                with_vectors=["image", "code"],
            ),
            key=lambda x: x.id,
        ),
    )
