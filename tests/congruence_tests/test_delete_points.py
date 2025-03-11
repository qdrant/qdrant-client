import uuid

from qdrant_client.client_base import QdrantBase
from qdrant_client.http.models import NamedSparseVector, NamedVector
from tests.congruence_tests.test_common import (
    COLLECTION_NAME,
    compare_client_results,
    compare_collections,
    generate_fixtures,
    generate_sparse_fixtures,
    init_client,
    sparse_vectors_config,
    initialize_fixture_collection,
)


def test_delete_points(local_client: QdrantBase, remote_client: QdrantBase, collection_name: str):
    points = generate_fixtures(100)
    vector = points[0].vector["image"]

    initialize_fixture_collection(local_client, collection_name)
    initialize_fixture_collection(remote_client, collection_name)
    local_client.upload_points(collection_name, points)
    remote_client.upload_points(collection_name, points, wait=True)

    compare_client_results(
        local_client,
        remote_client,
        lambda c: c.search(collection_name, query_vector=NamedVector(name="image", vector=vector)),
    )

    found_ids = [
        scored_point.id
        for scored_point in local_client.search(
            collection_name, query_vector=NamedVector(name="image", vector=vector)
        )
    ]

    local_client.delete(collection_name, found_ids)
    remote_client.delete(collection_name, found_ids)

    compare_collections(
        local_client, remote_client, 100, attrs=("points_count",), collection_name=collection_name
    )

    compare_client_results(
        local_client,
        remote_client,
        lambda c: c.search(collection_name, query_vector=NamedVector(name="image", vector=vector)),
    )

    # delete non-existent points
    local_client.delete(collection_name, found_ids)
    remote_client.delete(collection_name, found_ids)

    compare_collections(
        local_client, remote_client, 100, attrs=("points_count",), collection_name=collection_name
    )

    compare_client_results(
        local_client,
        remote_client,
        lambda c: c.search(collection_name, query_vector=NamedVector(name="image", vector=vector)),
    )


def test_delete_sparse_points(
    local_client: QdrantBase, remote_client: QdrantBase, collection_name: str
):
    points = generate_sparse_fixtures(100)
    vector = points[0].vector["sparse-image"]

    init_client(local_client, [], collection_name, sparse_vectors_config=sparse_vectors_config)
    init_client(remote_client, [], collection_name, sparse_vectors_config=sparse_vectors_config)

    local_client.upload_points(collection_name, points)
    remote_client.upload_points(collection_name, points, wait=True)

    compare_client_results(
        local_client,
        remote_client,
        lambda c: c.search(
            collection_name, query_vector=NamedSparseVector(name="sparse-image", vector=vector)
        ),
    )

    found_ids = [
        scored_point.id
        for scored_point in local_client.search(
            collection_name, query_vector=NamedSparseVector(name="sparse-image", vector=vector)
        )
    ]

    local_client.delete(collection_name, found_ids)
    remote_client.delete(collection_name, found_ids)

    compare_collections(
        local_client, remote_client, 100, attrs=("points_count",), collection_name=collection_name
    )

    compare_client_results(
        local_client,
        remote_client,
        lambda c: c.search(
            collection_name, query_vector=NamedSparseVector(name="sparse-image", vector=vector)
        ),
    )
