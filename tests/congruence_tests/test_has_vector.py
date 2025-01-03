from qdrant_client import models
from tests.congruence_tests.test_common import (
    COLLECTION_NAME,
    compare_client_results,
    generate_fixtures,
    generate_sparse_fixtures,
    init_local,
    init_client,
    init_remote,
    sparse_vectors_config,
)


def test_has_vector(local_client, remote_client):
    points = generate_fixtures(100, skip_vectors=True)

    local_client.upload_points(COLLECTION_NAME, points)
    remote_client.upload_points(COLLECTION_NAME, points, wait=True)

    local_client.upload_points(COLLECTION_NAME, points)
    remote_client.upload_points(COLLECTION_NAME, points, wait=True)

    compare_client_results(
        local_client,
        remote_client,
        lambda c: c.scroll(
            COLLECTION_NAME,
            limit=50,
            scroll_filter=models.Filter(must=[models.HasVectorCondition(has_vector="image")]),
        ),
    )


def test_has_vector_sparse():
    points = generate_sparse_fixtures(100, skip_vectors=True)

    local_client = init_local()
    init_client(local_client, [], sparse_vectors_config=sparse_vectors_config)

    remote_client = init_remote()
    init_client(remote_client, [], sparse_vectors_config=sparse_vectors_config)

    local_client.upload_points(COLLECTION_NAME, points)
    remote_client.upload_points(COLLECTION_NAME, points, wait=True)

    compare_client_results(
        local_client,
        remote_client,
        lambda c: c.scroll(
            COLLECTION_NAME,
            limit=50,
            scroll_filter=models.Filter(
                must=[models.HasVectorCondition(has_vector="sparse-image")]
            ),
        ),
    )
