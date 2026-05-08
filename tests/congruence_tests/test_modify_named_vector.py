import numpy as np

from qdrant_client.http import models
from tests.congruence_tests.test_common import (
    COLLECTION_NAME,
    compare_client_results,
    init_client,
    init_local,
    init_remote,
    text_vector_size,
    image_vector_size,
)


def test_create_and_delete_vector_name():
    local_client = init_local()
    http_client = init_remote()
    grpc_client = init_remote(prefer_grpc=True)

    vectors_config = {
        "text": models.VectorParams(size=text_vector_size, distance=models.Distance.COSINE),
    }

    init_client(local_client, [], vectors_config=vectors_config)
    init_client(http_client, [], vectors_config=vectors_config)

    dense_config = models.DenseVectorNameConfig(
        dense=models.DenseVectorConfig(
            size=image_vector_size,
            distance=models.Distance.COSINE,
            multivector_config=models.MultiVectorConfig(
                comparator=models.MultiVectorComparator.MAX_SIM,
            ),
        )
    )
    sparse_config = models.SparseVectorNameConfig(
        sparse=models.SparseVectorConfig(modifier=models.Modifier.IDF)
    )

    # Test create via HTTP
    local_client.create_vector_name(COLLECTION_NAME, "multi-image", dense_config)
    local_client.create_vector_name(COLLECTION_NAME, "sparse-idf", sparse_config)
    http_client.create_vector_name(COLLECTION_NAME, "multi-image", dense_config)
    http_client.create_vector_name(COLLECTION_NAME, "sparse-idf", sparse_config)

    local_info = local_client.get_collection(COLLECTION_NAME)
    http_info = http_client.get_collection(COLLECTION_NAME)
    assert "multi-image" in local_info.config.params.vectors
    assert local_info.config.params.vectors["multi-image"].multivector_config is not None
    assert "sparse-idf" in local_info.config.params.sparse_vectors
    assert "multi-image" in http_info.config.params.vectors
    assert "sparse-idf" in http_info.config.params.sparse_vectors

    points = [
        models.PointStruct(
            id=i,
            vector={
                "text": np.random.rand(text_vector_size).tolist(),
                "multi-image": [np.random.rand(image_vector_size).tolist() for _ in range(3)],
                "sparse-idf": models.SparseVector(
                    indices=list(range(10)), values=np.random.rand(10).tolist()
                ),
            },
        )
        for i in range(20)
    ]

    local_client.upsert(COLLECTION_NAME, points)
    http_client.upsert(COLLECTION_NAME, points, wait=True)

    compare_client_results(
        local_client,
        http_client,
        lambda c: sorted(
            c.retrieve(COLLECTION_NAME, list(range(10)), with_vectors=True, with_payload=False),
            key=lambda x: x.id,
        ),
    )

    # Delete both new vectors via HTTP, then verify
    local_client.delete_vector_name(COLLECTION_NAME, "multi-image")
    local_client.delete_vector_name(COLLECTION_NAME, "sparse-idf")
    http_client.delete_vector_name(COLLECTION_NAME, "multi-image")
    http_client.delete_vector_name(COLLECTION_NAME, "sparse-idf")

    local_info = local_client.get_collection(COLLECTION_NAME)
    http_info = http_client.get_collection(COLLECTION_NAME)
    assert "multi-image" not in local_info.config.params.vectors
    assert "sparse-idf" not in (local_info.config.params.sparse_vectors or {})
    assert "text" in local_info.config.params.vectors
    assert "multi-image" not in http_info.config.params.vectors
    assert "sparse-idf" not in (http_info.config.params.sparse_vectors or {})

    compare_client_results(
        local_client,
        http_client,
        lambda c: sorted(
            c.retrieve(COLLECTION_NAME, list(range(10)), with_vectors=True, with_payload=False),
            key=lambda x: x.id,
        ),
    )

    # Re-test create + delete via gRPC on the same remote collection
    grpc_client.create_vector_name(COLLECTION_NAME, "multi-image", dense_config)
    grpc_client.create_vector_name(COLLECTION_NAME, "sparse-idf", sparse_config)

    grpc_info = grpc_client.get_collection(COLLECTION_NAME)
    assert "multi-image" in grpc_info.config.params.vectors
    assert "sparse-idf" in grpc_info.config.params.sparse_vectors

    grpc_client.delete_vector_name(COLLECTION_NAME, "multi-image")
    grpc_client.delete_vector_name(COLLECTION_NAME, "sparse-idf")

    grpc_info = grpc_client.get_collection(COLLECTION_NAME)
    assert "multi-image" not in grpc_info.config.params.vectors
    assert "sparse-idf" not in (grpc_info.config.params.sparse_vectors or {})
