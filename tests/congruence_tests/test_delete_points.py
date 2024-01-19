from qdrant_client.http.models import NamedSparseVector, NamedVector
from tests.congruence_tests.test_common import (
    COLLECTION_NAME,
    compare_client_results,
    compare_collections,
    generate_fixtures,
    generate_sparse_fixtures,
    init_client,
    init_local,
    init_remote,
    sparse_vectors_config,
)


def test_delete_points(local_client, remote_client):
    points = generate_fixtures(100)
    vector = points[0].vector["image"]

    local_client.upload_points(COLLECTION_NAME, points)
    remote_client.upload_points(COLLECTION_NAME, points, wait=True)

    compare_client_results(
        local_client,
        remote_client,
        lambda c: c.search(COLLECTION_NAME, query_vector=NamedVector(name="image", vector=vector)),
    )

    found_ids = [
        scored_point.id
        for scored_point in local_client.search(
            COLLECTION_NAME, query_vector=NamedVector(name="image", vector=vector)
        )
    ]

    local_client.delete(COLLECTION_NAME, found_ids)
    remote_client.delete(COLLECTION_NAME, found_ids)

    compare_collections(local_client, remote_client, 100, attrs=("points_count",))

    compare_client_results(
        local_client,
        remote_client,
        lambda c: c.search(COLLECTION_NAME, query_vector=NamedVector(name="image", vector=vector)),
    )


def test_delete_sparse_points():
    points = generate_sparse_fixtures(100)
    vector = points[0].vector["sparse-image"]

    local_client = init_local()
    init_client(local_client, [], sparse_vectors_config=sparse_vectors_config)

    remote_client = init_remote()
    init_client(remote_client, [], sparse_vectors_config=sparse_vectors_config)

    local_client.upload_points(COLLECTION_NAME, points)
    remote_client.upload_points(COLLECTION_NAME, points, wait=True)

    compare_client_results(
        local_client,
        remote_client,
        lambda c: c.search(
            COLLECTION_NAME, query_vector=NamedSparseVector(name="sparse-image", vector=vector)
        ),
    )

    found_ids = [
        scored_point.id
        for scored_point in local_client.search(
            COLLECTION_NAME, query_vector=NamedSparseVector(name="sparse-image", vector=vector)
        )
    ]

    local_client.delete(COLLECTION_NAME, found_ids)
    remote_client.delete(COLLECTION_NAME, found_ids)

    compare_collections(local_client, remote_client, 100, attrs=("points_count",))

    compare_client_results(
        local_client,
        remote_client,
        lambda c: c.search(
            COLLECTION_NAME, query_vector=NamedSparseVector(name="sparse-image", vector=vector)
        ),
    )
