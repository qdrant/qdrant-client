from time import sleep
from typing import Callable

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
    fixture_points = generate_fixtures()

    remote_client = init_remote()

    remote_collections = remote_client.get_collections()

    for collection in remote_collections.collections:
        remote_client.delete_collection(collection.name)

    local_client = init_local()
    init_client(local_client, fixture_points)

    init_client(remote_client, fixture_points)

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

    local_client.recreate_collection(COLLECTION_NAME, vectors_config=vector_params)
    http_client.recreate_collection(COLLECTION_NAME, vectors_config=vector_params)

    assert local_client.collection_exists(COLLECTION_NAME)
    assert http_client.collection_exists(COLLECTION_NAME)

    http_client.delete_collection(COLLECTION_NAME)
    grpc_client.recreate_collection(COLLECTION_NAME, vectors_config=vector_params)
    assert grpc_client.collection_exists(COLLECTION_NAME)


def test_collection_exists():
    remote_client = init_remote()
    local_client = init_local()

    assert not remote_client.collection_exists(COLLECTION_NAME + "_not_exists")
    assert not local_client.collection_exists(COLLECTION_NAME + "_not_exists")

    vector_params = models.VectorParams(size=2, distance=models.Distance.COSINE)

    try:
        remote_client.delete_collection(COLLECTION_NAME)
    except UnexpectedResponse:
        pass  # collection does not exist

    remote_client.create_collection(COLLECTION_NAME, vectors_config=vector_params)

    try:
        local_client.delete_collection(COLLECTION_NAME)
    except ValueError:
        pass  # collection does not exist

    local_client.create_collection(COLLECTION_NAME, vectors_config=vector_params)

    assert remote_client.collection_exists(COLLECTION_NAME)
    assert local_client.collection_exists(COLLECTION_NAME)


def test_init_from():
    vector_size = 2

    remote_client = init_remote()
    local_client = init_local()

    points = generate_fixtures(vectors_sizes=vector_size)
    vector_params = models.VectorParams(size=vector_size, distance=models.Distance.COSINE)

    if remote_client.collection_exists(COLLECTION_NAME):
        remote_client.delete_collection(collection_name=COLLECTION_NAME)
    remote_client.create_collection(collection_name=COLLECTION_NAME, vectors_config=vector_params)

    if local_client.collection_exists(COLLECTION_NAME):
        local_client.delete_collection(collection_name=COLLECTION_NAME)
    local_client.create_collection(collection_name=COLLECTION_NAME, vectors_config=vector_params)

    remote_client.upload_points(COLLECTION_NAME, points, wait=True)
    local_client.upload_points(COLLECTION_NAME, points)
    compare_collections(remote_client, local_client, len(points), collection_name=COLLECTION_NAME)

    new_collection_name = COLLECTION_NAME + "_new"
    if remote_client.collection_exists(new_collection_name):
        remote_client.delete_collection(new_collection_name)
    remote_client.create_collection(
        new_collection_name, vectors_config=vector_params, init_from=COLLECTION_NAME
    )

    if local_client.collection_exists(new_collection_name):
        local_client.delete_collection(new_collection_name)
    local_client.create_collection(
        new_collection_name, vectors_config=vector_params, init_from=COLLECTION_NAME
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
        init_from=models.InitFrom(collection=COLLECTION_NAME),
    )
    if local_client.collection_exists(new_collection_name):
        local_client.delete_collection(new_collection_name)
    local_client.create_collection(
        new_collection_name,
        vectors_config=vector_params,
        init_from=models.InitFrom(collection=COLLECTION_NAME),
    )

    # init_from is performed asynchronously, so we need to retry
    wait_for(
        compare_collections,
        remote_client,
        local_client,
        len(points),
        collection_name=new_collection_name,
    )


def wait_for(condition: Callable, *args, **kwargs):
    for i in range(0, 10):
        try:
            condition(*args, **kwargs)
        except AssertionError:
            sleep(0.5)
            continue
        break
