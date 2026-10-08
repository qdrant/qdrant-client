from time import sleep
from typing import Callable

import grpc
import pytest

from qdrant_client.http import models
from qdrant_client.http.exceptions import UnexpectedResponse
from tests.congruence_tests.test_common import (
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

    with pytest.raises(ValueError, match="Collection name must not be empty"):
        local_client.collection_exists("")

    with pytest.raises(ValueError, match="Collection name must not be empty"):
        remote_client.collection_exists("")


def test_config_variations():
    def check_variation(vectors_config, sparse_vectors_config):
        if remote_client.collection_exists(COLLECTION_NAME):
            remote_client.delete_collection(COLLECTION_NAME)
        if local_client.collection_exists(COLLECTION_NAME):
            local_client.delete_collection(COLLECTION_NAME)

        remote_client.create_collection(
            COLLECTION_NAME,
            vectors_config=vectors_config,
            sparse_vectors_config=sparse_vectors_config,
        )
        local_client.create_collection(
            COLLECTION_NAME,
            vectors_config=vectors_config,
            sparse_vectors_config=sparse_vectors_config,
        )

        remote_client_config_params = remote_client.get_collection(COLLECTION_NAME).config.params
        local_client_config_params = local_client.get_collection(COLLECTION_NAME).config.params

        assert remote_client_config_params.vectors == local_client_config_params.vectors
        assert (
            remote_client_config_params.sparse_vectors == local_client_config_params.sparse_vectors
        )

        remote_grpc_client.delete_collection(COLLECTION_NAME)
        remote_grpc_client.create_collection(
            COLLECTION_NAME,
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


def test_create_collection_empty_sparse_vector_name():
    collection_name = "test_empty_sparse_vector_name"
    error_message = "Sparse vector name cannot be empty"

    local_client = init_local()
    http_client = init_remote()
    grpc_client = init_remote(prefer_grpc=True)

    if http_client.collection_exists(collection_name):
        http_client.delete_collection(collection_name)

    dense_params = models.VectorParams(size=2, distance=models.Distance.COSINE)
    configs = [
        (None, {"": models.SparseVectorParams()}),
        (None, {"sparse": models.SparseVectorParams(), "": models.SparseVectorParams()}),
        (dense_params, {"": models.SparseVectorParams()}),
        ({"text": dense_params}, {"": models.SparseVectorParams()}),
    ]

    for vectors_config, sparse_vectors_config in configs:
        with pytest.raises(ValueError, match=error_message):
            local_client.create_collection(
                collection_name,
                vectors_config=vectors_config,
                sparse_vectors_config=sparse_vectors_config,
            )
        assert not local_client.collection_exists(collection_name)

        with pytest.raises(UnexpectedResponse, match=error_message):
            http_client.create_collection(
                collection_name,
                vectors_config=vectors_config,
                sparse_vectors_config=sparse_vectors_config,
            )

        with pytest.raises(grpc.RpcError, match=error_message):
            grpc_client.create_collection(
                collection_name,
                vectors_config=vectors_config,
                sparse_vectors_config=sparse_vectors_config,
            )
        assert not http_client.collection_exists(collection_name)


@pytest.mark.parametrize("prefer_grpc", [False, True])
def test_update_collection_metadata_null_removes_key(prefer_grpc):
    local_client = init_local()
    remote_client = init_remote(prefer_grpc=prefer_grpc)
    vector_params = models.VectorParams(size=2, distance=models.Distance.COSINE)

    # top-level nulls remove keys from existing metadata, nested nulls are stored as is;
    # a collection without metadata stores the update as is, nulls included
    new_metadata = {"a": None, "missing": None, "b": {"c": None}}
    for initial_metadata in [{"a": 1, "keep": 1, "kept": None}, None]:
        for client in (local_client, remote_client):
            if client.collection_exists(COLLECTION_NAME):
                client.delete_collection(COLLECTION_NAME)
            client.create_collection(
                COLLECTION_NAME, vectors_config=vector_params, metadata=initial_metadata
            )
            client.update_collection(COLLECTION_NAME, metadata=new_metadata)

        assert (
            local_client.get_collection(COLLECTION_NAME).config.metadata
            == remote_client.get_collection(COLLECTION_NAME).config.metadata
        )


def wait_for(condition: Callable, *args, **kwargs):
    for i in range(0, 10):
        try:
            condition(*args, **kwargs)
        except AssertionError:
            sleep(0.5)
            continue
        break
