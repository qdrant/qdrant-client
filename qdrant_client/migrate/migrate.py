from typing import List

from qdrant_client import QdrantClient
from qdrant_client.http import models


def migrate(source_client: QdrantClient, dest_client: QdrantClient, batch_size: int = 100) -> None:
    """Migrate all collections from source client to destination client

    Args:
        source_client (QdrantClient): Source client
        dest_client (QdrantClient): Destination client
        batch_size (int, optional): Batch size for scrolling and uploading vectors. Defaults to 100.
    """
    source_collections = source_client.get_collections().collections
    dest_collections = dest_client.get_collections().collections

    source_collection_names = {collection.name for collection in source_collections}
    dest_collection_names = {collection.name for collection in dest_collections}

    missing_collections = source_collection_names - dest_collection_names
    assert (
        not missing_collections
    ), f"Destination client should have all collections from source client. Missing collections: {missing_collections}"

    _compare_collections(list(source_collection_names), source_client, dest_client)

    print(f"Number of collections to migrate: {len(source_collection_names)}", end="\n\n")
    for collection_name in source_collection_names:
        print(f"Start migrating collection `{collection_name}`")
        _migrate_collection(collection_name, source_client, dest_client, batch_size)
        print(f"Finish migrating collection `{collection_name}`", end="\n\n")


def _compare_collections(
    source_collection_names: List[str],
    source_client: QdrantClient,
    dest_client: QdrantClient,
) -> bool:
    """Compare collections from source client and destination client

    Args:
        source_collection_names (list[str]): List of collection names
        source_client (QdrantClient): Source client
        dest_client (QdrantClient): Destination client

    Returns:
        bool: True if collections have the same vector and distance params
    """
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
            assert len(source_vector_params) == len(
                dest_vector_params
            ), "Mismatched vector params: number of named vectors is not equal"

            for key, source_vector_param in source_vector_params.items():
                dest_vector_param = dest_vector_params[key]
                assert (
                    source_vector_param.size == dest_vector_param.size
                ), f"Vector size is not equal for {key} in {collection_name}"
                assert (
                    source_vector_param.distance == dest_vector_param.distance
                ), f"Distance is not the same for {key} in {collection_name}"

    return True


def _migrate_collection(
    collection_name: str,
    source_client: QdrantClient,
    dest_client: QdrantClient,
    batch_size: int = 100,
) -> None:
    """Migrate collection from source client to destination client

    Args:
        collection_name (str): Collection name
        source_client (QdrantClient): Source client
        dest_client (QdrantClient): Destination client
        batch_size (int, optional): Batch size for scrolling and uploading vectors. Defaults to 100.
    """
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
