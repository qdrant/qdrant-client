import numpy as np
import pytest

from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.migrate import migrate

VECTOR_NUMBER = 1000


@pytest.fixture
def source_client() -> QdrantClient:
    client = QdrantClient(":memory:")
    yield client
    client.close()


@pytest.fixture
def dest_client() -> QdrantClient:
    client = QdrantClient()
    yield client
    client.close()


def test_single_vector_collection(source_client: QdrantClient, dest_client: QdrantClient) -> None:
    single_vector_collection_kwargs = {
        "collection_name": "single_vector_collection",
        "vectors_config": models.VectorParams(size=10, distance=models.Distance.COSINE),
    }
    source_client.recreate_collection(**single_vector_collection_kwargs)
    dest_client.recreate_collection(**single_vector_collection_kwargs)

    source_client.upload_collection(
        single_vector_collection_kwargs["collection_name"],
        vectors=np.random.randn(
            VECTOR_NUMBER, single_vector_collection_kwargs["vectors_config"].size
        ),
    )

    migrate(source_client, dest_client)


def test_multiple_vectors_collection(
    source_client: QdrantClient, dest_client: QdrantClient
) -> None:
    multiple_vectors_collection_kwargs = {
        "collection_name": "multiple_vectors_collection",
        "vectors_config": {
            "text": models.VectorParams(size=10, distance=models.Distance.EUCLID),
            "image": models.VectorParams(size=11, distance=models.Distance.COSINE),
        },
    }
    source_client.recreate_collection(**multiple_vectors_collection_kwargs)
    dest_client.recreate_collection(**multiple_vectors_collection_kwargs)
    source_client.upload_collection(
        multiple_vectors_collection_kwargs["collection_name"],
        vectors={
            "text": np.random.randn(
                VECTOR_NUMBER,
                multiple_vectors_collection_kwargs["vectors_config"]["text"].size,
            ),
            "image": np.random.randn(
                VECTOR_NUMBER,
                multiple_vectors_collection_kwargs["vectors_config"]["image"].size,
            ),
        },
    )


def test_different_distances(source_client: QdrantClient, dest_client: QdrantClient) -> None:
    collection_name = "single_vector_collection"
    cosine_params = models.VectorParams(size=10, distance=models.Distance.COSINE)
    euclid_params = models.VectorParams(size=10, distance=models.Distance.EUCLID)

    source_client.recreate_collection(collection_name, vectors_config=cosine_params)
    dest_client.recreate_collection(collection_name, vectors_config=euclid_params)

    with pytest.raises(AssertionError):
        migrate(source_client, dest_client)


def test_different_vector_sizes(source_client: QdrantClient, dest_client: QdrantClient) -> None:
    collection_name = "single_vector_collection"
    small_vector_params = models.VectorParams(size=10, distance=models.Distance.COSINE)
    big_vector_params = models.VectorParams(size=100, distance=models.Distance.COSINE)

    source_client.recreate_collection(collection_name, vectors_config=small_vector_params)
    dest_client.recreate_collection(collection_name, vectors_config=big_vector_params)

    with pytest.raises(AssertionError):
        migrate(source_client, dest_client)


def test_single_vs_multiple_vectors(
    source_client: QdrantClient, dest_client: QdrantClient
) -> None:
    collection_name = "test_collection"
    single_vector_params = {"text": models.VectorParams(size=10, distance=models.Distance.COSINE)}
    multiple_vectors_params = {
        "text": models.VectorParams(size=10, distance=models.Distance.COSINE),
        "image": models.VectorParams(size=11, distance=models.Distance.COSINE),
    }

    source_client.recreate_collection(collection_name, vectors_config=single_vector_params)
    dest_client.recreate_collection(collection_name, vectors_config=multiple_vectors_params)

    with pytest.raises(AssertionError):
        migrate(source_client, dest_client)
