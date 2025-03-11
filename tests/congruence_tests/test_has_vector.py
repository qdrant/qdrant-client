from qdrant_client.client_base import QdrantBase
from qdrant_client import models
from tests.congruence_tests.test_common import (
    compare_client_results,
    generate_fixtures,
    generate_sparse_fixtures,
    init_local,
    init_client,
    init_remote,
    sparse_vectors_config,
    generate_multivector_fixtures,
    multi_vector_config,
    initialize_fixture_collection,
)


def test_has_vector(local_client: QdrantBase, remote_client: QdrantBase, collection_name: str):
    points = generate_fixtures(100, skip_vectors=True)

    initialize_fixture_collection(local_client, collection_name)
    initialize_fixture_collection(remote_client, collection_name)

    local_client.upload_points(collection_name, points)
    remote_client.upload_points(collection_name, points, wait=True)

    local_client.upload_points(collection_name, points)
    remote_client.upload_points(collection_name, points, wait=True)

    compare_client_results(
        local_client,
        remote_client,
        lambda c: c.scroll(
            collection_name,
            limit=50,
            scroll_filter=models.Filter(must=[models.HasVectorCondition(has_vector="image")]),
        )[0],
    )


def test_has_vector_sparse(collection_name: str):
    points = generate_sparse_fixtures(100, skip_vectors=True)

    local_client = init_local()
    init_client(local_client, [], collection_name, sparse_vectors_config=sparse_vectors_config)

    remote_client = init_remote()
    init_client(remote_client, [], collection_name, sparse_vectors_config=sparse_vectors_config)

    local_client.upload_points(collection_name, points)
    remote_client.upload_points(collection_name, points, wait=True)

    compare_client_results(
        local_client,
        remote_client,
        lambda c: c.scroll(
            collection_name,
            limit=50,
            scroll_filter=models.Filter(
                must=[models.HasVectorCondition(has_vector="sparse-image")]
            ),
        )[0],
    )


def test_has_vector_multi(collection_name: str):
    points = generate_multivector_fixtures(100, skip_vectors=True)

    local_client = init_local()
    init_client(local_client, [], collection_name, vectors_config=multi_vector_config)

    remote_client = init_remote()
    init_client(remote_client, [], collection_name, vectors_config=multi_vector_config)

    local_client.upload_points(collection_name, points)
    remote_client.upload_points(collection_name, points, wait=True)

    local_client.upload_points(collection_name, points)
    remote_client.upload_points(collection_name, points, wait=True)

    compare_client_results(
        local_client,
        remote_client,
        lambda c: c.scroll(
            collection_name,
            limit=50,
            scroll_filter=models.Filter(must=[models.HasVectorCondition(has_vector="multi-code")]),
        )[0],
    )
