import uuid
from time import sleep
from typing import Callable, Any

from qdrant_client.http import models
from qdrant_client.http.exceptions import UnexpectedResponse
from tests.congruence_tests.test_common import (
    compare_collections,
    generate_fixtures,
    init_client,
    init_local,
    init_remote,
)


COLLECTION_NAME = "test_collection"


def test_get_collection():
    # when running tests in parallel, it will fail because other tests create collections
    # and the length of collections will be different
    wait_for(test_recreate_collection)
    wait_for(test_collection_exists)
    wait_for(test_init_from)
    wait_for(test_config_variations)

    fixture_points = generate_fixtures()

    remote_client = init_remote()

    remote_collections = remote_client.get_collections()

    for collection in remote_collections.collections:
        remote_client.delete_collection(collection.name)

    collection_name = f"{COLLECTION_NAME}_{uuid.uuid4().hex}"
    local_client = init_local()
    init_client(local_client, fixture_points, collection_name)

    init_client(remote_client, fixture_points, collection_name)

    local_collections = local_client.get_collections()

    remote_collections = remote_client.get_collections()

    assert len(local_collections.collections) == len(remote_collections.collections)

    local_collection = local_collections.collections[0].name
    remote_collection = remote_collections.collections[0].name

    assert local_collection == remote_collection

    local_collection_info = local_client.get_collection(local_collection)

    remote_collection_info = remote_client.get_collection(remote_collection)

    assert local_collection_info.points_count == remote_collection_info.points_count

    assert (
        local_collection_info.config.params.vectors == remote_collection_info.config.params.vectors
    )


def test_recreate_collection():
    # this method has been marked as deprecated and should be removed in qdrant-client v1.12
    local_client = init_local()
    http_client = init_remote()
    grpc_client = init_remote(prefer_grpc=True)

    vector_params = models.VectorParams(size=20, distance=models.Distance.COSINE)

    collection_name = f"{COLLECTION_NAME}_{uuid.uuid4().hex}"
    local_client.recreate_collection(collection_name, vectors_config=vector_params)
    http_client.recreate_collection(collection_name, vectors_config=vector_params)

    assert local_client.collection_exists(collection_name)
    assert http_client.collection_exists(collection_name)

    http_client.delete_collection(collection_name)
    grpc_client.recreate_collection(collection_name, vectors_config=vector_params)
    assert grpc_client.collection_exists(collection_name)


def test_collection_exists():
    remote_client = init_remote()
    local_client = init_local()

    collection_name = f"{COLLECTION_NAME}_{uuid.uuid4().hex}"
    assert not remote_client.collection_exists(collection_name + "_not_exists")
    assert not local_client.collection_exists(collection_name + "_not_exists")

    vector_params = models.VectorParams(size=2, distance=models.Distance.COSINE)

    try:
        remote_client.delete_collection(collection_name)
    except UnexpectedResponse:
        pass  # collection does not exist

    remote_client.create_collection(collection_name, vectors_config=vector_params)

    try:
        local_client.delete_collection(collection_name)
    except ValueError:
        pass  # collection does not exist

    local_client.create_collection(collection_name, vectors_config=vector_params)

    assert remote_client.collection_exists(collection_name)
    assert local_client.collection_exists(collection_name)


def test_init_from():
    vector_size = 2

    remote_client = init_remote()
    local_client = init_local()

    points = generate_fixtures(vectors_sizes=vector_size)
    vector_params = models.VectorParams(size=vector_size, distance=models.Distance.COSINE)

    collection_name = f"{COLLECTION_NAME}_{uuid.uuid4().hex}"
    if remote_client.collection_exists(collection_name):
        remote_client.delete_collection(collection_name=collection_name)
    remote_client.create_collection(collection_name=collection_name, vectors_config=vector_params)

    if local_client.collection_exists(collection_name):
        local_client.delete_collection(collection_name=collection_name)
    local_client.create_collection(collection_name=collection_name, vectors_config=vector_params)

    remote_client.upload_points(collection_name, points, wait=True)
    local_client.upload_points(collection_name, points)
    compare_collections(remote_client, local_client, len(points), collection_name=collection_name)

    new_collection_name = collection_name + "_new"
    if remote_client.collection_exists(new_collection_name):
        remote_client.delete_collection(new_collection_name)
    remote_client.create_collection(
        new_collection_name, vectors_config=vector_params, init_from=collection_name
    )

    if local_client.collection_exists(new_collection_name):
        local_client.delete_collection(new_collection_name)
    local_client.create_collection(
        new_collection_name, vectors_config=vector_params, init_from=collection_name
    )

    # init_from is performed asynchronously, so we need to retry
    wait_for(
        compare_collections,
        remote_client,
        local_client,
        len(points),
        collection_name=new_collection_name,
    )

    # try with models.InitFrom
    if remote_client.collection_exists(new_collection_name):
        remote_client.delete_collection(new_collection_name)
    remote_client.create_collection(
        new_collection_name,
        vectors_config=vector_params,
        init_from=models.InitFrom(collection=collection_name),
    )
    if local_client.collection_exists(new_collection_name):
        local_client.delete_collection(new_collection_name)
    local_client.create_collection(
        new_collection_name,
        vectors_config=vector_params,
        init_from=models.InitFrom(collection=collection_name),
    )

    # init_from is performed asynchronously, so we need to retry
    wait_for(
        compare_collections,
        remote_client,
        local_client,
        len(points),
        collection_name=new_collection_name,
    )


def test_config_variations():
    collection_name = f"{COLLECTION_NAME}_{uuid.uuid4().hex}"

    def check_variation(vectors_config, sparse_vectors_config):
        if remote_client.collection_exists(collection_name):
            remote_client.delete_collection(collection_name)
        if local_client.collection_exists(collection_name):
            local_client.delete_collection(collection_name)

        remote_client.create_collection(
            collection_name,
            vectors_config=vectors_config,
            sparse_vectors_config=sparse_vectors_config,
        )
        local_client.create_collection(
            collection_name,
            vectors_config=vectors_config,
            sparse_vectors_config=sparse_vectors_config,
        )

        remote_client_config_params = remote_client.get_collection(collection_name).config.params
        local_client_config_params = local_client.get_collection(collection_name).config.params

        assert remote_client_config_params.vectors == local_client_config_params.vectors
        assert (
            remote_client_config_params.sparse_vectors == local_client_config_params.sparse_vectors
        )

        remote_grpc_client.delete_collection(collection_name)
        remote_grpc_client.create_collection(
            collection_name,
            vectors_config=vectors_config,
            sparse_vectors_config=sparse_vectors_config,
        )

        assert remote_client_config_params.vectors == local_client_config_params.vectors
        assert (
            remote_client_config_params.sparse_vectors == local_client_config_params.sparse_vectors
        )

    remote_client = init_remote()
    remote_grpc_client = init_remote(prefer_grpc=True)
    local_client = init_local()

    vectors_config = models.VectorParams(size=2, distance=models.Distance.COSINE)

    sparse_vectors_config = {"sparse": models.SparseVectorParams()}

    check_variation(vectors_config, sparse_vectors_config)
    check_variation(vectors_config, None)
    check_variation(None, sparse_vectors_config)
    check_variation({"text": vectors_config}, sparse_vectors_config)
    check_variation({"text": vectors_config}, None)
    check_variation(None, None)


def wait_for(condition: Callable[..., None], *args: Any, **kwargs: Any) -> None:
    for _ in range(10):
        try:
            condition(*args, **kwargs)
        except AssertionError:
            sleep(0.5)
            continue
        break
