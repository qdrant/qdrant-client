from qdrant_client import QdrantClient
from qdrant_client.http import models


def migrate(source_client: QdrantClient, dest_client: QdrantClient, batch_size: int = 100) -> None:
    source_collections = source_client.get_collections().collections
    dest_collections = dest_client.get_collections().collections

    source_collection_names = {collection.name for collection in source_collections}
    dest_collection_names = {collection.name for collection in dest_collections}

    missing_collections = source_collection_names - dest_collection_names
    assert (
        not missing_collections
    ), f"Destination client should have all collections from source client. Missing collections: {missing_collections}"

    compare_collections(list(source_collection_names), source_client, dest_client)

    for collection_name in source_collection_names:
        migrate_collection(collection_name, source_client, dest_client, batch_size)


def compare_collections(
    source_collection_names: list[str],
    source_client: QdrantClient,
    dest_client: QdrantClient,
) -> bool:
    for collection_name in source_collection_names:
        source_collection = source_client.get_collection(collection_name)
        source_vector_params = source_collection.config.params.vectors
        dest_collection = dest_client.get_collection(collection_name)
        dest_vector_params = dest_collection.config.params.vectors

        if isinstance(source_vector_params, models.VectorParams):
            assert isinstance(dest_vector_params, models.VectorParams), "Mismatched vector params"
            assert (
                source_vector_params.size == dest_vector_params.size
            ), "Vector size should be equal"
            assert (
                source_vector_params.distance == dest_vector_params.distance
            ), "Distance should be equal"

        elif isinstance(source_vector_params, dict):
            for key, source_vector_param in source_vector_params.items():
                dest_vector_param = dest_vector_params[key]
                assert (
                    source_vector_param.size == dest_vector_param.size
                ), f"Vector size is not equal for {key} in {collection_name}"
                assert (
                    source_vector_param.distance == dest_vector_param.distance
                ), f"Distance is not the same for {key} in {collection_name}"

        return True


def migrate_collection(
    collection_name: str,
    source_client: QdrantClient,
    dest_client: QdrantClient,
    batch_size: int = 100,
) -> None:
    records, next_offset = source_client.scroll(
        collection_name, limit=batch_size, with_vectors=True
    )
    dest_client.upload_records(collection_name, records)
    while next_offset:
        records, next_offset = source_client.scroll(
            collection_name, offset=next_offset, limit=batch_size, with_vectors=True
        )
        dest_client.upload_records(collection_name, records)

    source_client_vectors_count = source_client.get_collection(collection_name).vectors_count
    dest_client_vectors_count = dest_client.get_collection(collection_name).vectors_count
    assert (
        source_client_vectors_count == dest_client_vectors_count
    ), f"Migration failed, vectors count are not equal: source vector count {source_client_vectors_count}, dest vector count {dest_client_vectors_count}"


if __name__ == "__main__":
    import numpy as np

    VECTOR_NUMBER = 1000

    local_client = QdrantClient(":memory:")
    remote_client = QdrantClient()

    single_vector_collection_kwargs = {
        "collection_name": "single_vector_collection",
        "vectors_config": models.VectorParams(size=10, distance=models.Distance.COSINE),
    }
    multiple_vectors_collection_kwargs = {
        "collection_name": "multiple_vectors_collection",
        "vectors_config": {
            "text": models.VectorParams(size=10, distance=models.Distance.EUCLID),
            "image": models.VectorParams(size=11, distance=models.Distance.COSINE),
        },
    }
    local_client.recreate_collection(**single_vector_collection_kwargs)
    local_client.recreate_collection(**multiple_vectors_collection_kwargs)
    remote_client.recreate_collection(**single_vector_collection_kwargs)
    remote_client.recreate_collection(**multiple_vectors_collection_kwargs)

    local_client.upload_collection(
        single_vector_collection_kwargs["collection_name"],
        vectors=np.random.randn(
            VECTOR_NUMBER, single_vector_collection_kwargs["vectors_config"].size
        ),
    )
    local_client.upload_collection(
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
    migrate(local_client, remote_client)
