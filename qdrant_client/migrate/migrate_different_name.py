from typing import Dict, Optional, Any
from qdrant_client._pydantic_compat import to_dict
from qdrant_client.client_base import QdrantBase
from qdrant_client.http import models
from qdrant_client.http.models import PointStruct
from qdrant_client.migrate.migrate import (
    _has_custom_shards,
    _find_collisions,
    _recreate_payload_schema,
)


def migrate_with_different_name(
    source_client: QdrantBase,
    dest_client: QdrantBase,
    collection_mapping: Dict[str, str],
    recreate_on_collision: bool = False,
    batch_size: int = 100,
) -> None:
    """
    Migrate collections from source client to destination client with optional renaming

    Args:
        source_client (QdrantBase): Source client
        dest_client (QdrantBase): Destination client
        collection_mapping (Dict[str, str]): Mapping of source collection names to destination names
        recreate_on_collision (bool, optional): If True - recreate collection if it exists, otherwise
            raise ValueError.
        batch_size (int, optional): Batch size for scrolling and uploading vectors. Defaults to 100.
    """
    source_collection_names = list(collection_mapping.keys())
    dest_collection_names = list(collection_mapping.values())

    if any(_has_custom_shards(source_client, name) for name in source_collection_names):
        raise ValueError("Migration of collections with custom shards is not supported yet")

    collisions = _find_collisions(dest_client, dest_collection_names)
    absent_dest_collections = set(dest_collection_names) - set(collisions)

    if collisions and not recreate_on_collision:
        raise ValueError(f"Collections already exist in dest_client: {collisions}")

    for src_name, dest_name in collection_mapping.items():
        if dest_name in absent_dest_collections:
            _recreate_collection(source_client, dest_client, src_name, dest_name)
        elif recreate_on_collision:
            dest_client.delete_collection(dest_name)
            _recreate_collection(source_client, dest_client, src_name, dest_name)
        _migrate_collection(source_client, dest_client, src_name, dest_name, batch_size)


def _recreate_collection(
    source_client: QdrantBase,
    dest_client: QdrantBase,
    src_collection_name: str,
    dest_collection_name: str,
) -> None:
    src_collection_info = source_client.get_collection(src_collection_name)
    src_config = src_collection_info.config
    src_payload_schema = src_collection_info.payload_schema

    dest_client.recreate_collection(
        dest_collection_name,
        vectors_config=src_config.params.vectors,
        sparse_vectors_config=src_config.params.sparse_vectors,
        shard_number=src_config.params.shard_number,
        replication_factor=src_config.params.replication_factor,
        write_consistency_factor=src_config.params.write_consistency_factor,
        on_disk_payload=src_config.params.on_disk_payload,
        hnsw_config=models.HnswConfigDiff(**to_dict(src_config.hnsw_config)),
        optimizers_config=models.OptimizersConfigDiff(**to_dict(src_config.optimizer_config)),
        wal_config=models.WalConfigDiff(**to_dict(src_config.wal_config)),
        quantization_config=src_config.quantization_config,
    )

    _recreate_payload_schema(dest_client, dest_collection_name, src_payload_schema)


def _migrate_collection_old(
    source_client: QdrantBase,
    dest_client: QdrantBase,
    src_collection_name: str,
    dest_collection_name: str,
    batch_size: int = 100,
) -> None:
    """Migrate collection from source client to destination client

    Args:
        src_collection_name (str): Source collection name
        dest_collection_name (str): Destination collection name
        source_client (QdrantBase): Source client
        dest_client (QdrantBase): Destination client
        batch_size (int, optional): Batch size for scrolling and uploading vectors. Defaults to 100.
    """
    offset: Optional[Dict[str, Any]] = None
    total_migrated = 0

    while True:
        records, offset = source_client.scroll(
            src_collection_name,
            offset=offset,
            limit=batch_size,
            with_vectors=True,
            with_payload=True,
        )

        if not records:
            break

        points = [
            PointStruct(id=record.id, vector=record.vector, payload=record.payload)
            for record in records
        ]
        dest_client.upsert(dest_collection_name, points=points)

        total_migrated += len(records)
        print(f"Migrated {total_migrated} points so far...")

        if offset is None:
            break

    source_client_vectors_count = source_client.count(src_collection_name).count
    dest_client_vectors_count = dest_client.count(dest_collection_name).count

    print(
        f"Migration completed. Source count: {source_client_vectors_count}, Destination count: {dest_client_vectors_count}"
    )

    assert (
        source_client_vectors_count == dest_client_vectors_count
    ), f"Migration failed, vectors count are not equal: source vector count {source_client_vectors_count}, dest vector count {dest_client_vectors_count}"

    print(
        f"Successfully migrated {total_migrated} points from {src_collection_name} to {dest_collection_name}"
    )


def _migrate_collection(
    source_client: QdrantBase,
    dest_client: QdrantBase,
    src_collection_name: str,
    dest_collection_name: str,
    batch_size: int = 100,
) -> None:
    """Migrate collection from source client to destination client

    Args:
        src_collection_name (str): Source collection name
        dest_collection_name (str): Destination collection name
        source_client (QdrantBase): Source client
        dest_client (QdrantBase): Destination client
        batch_size (int, optional): Batch size for scrolling and uploading vectors. Defaults to 100.
    """
    records, next_offset = source_client.scroll(src_collection_name, limit=2, with_vectors=True)
    dest_client.upload_points(dest_collection_name, records, wait=True)  # type: ignore

    # upload_records has been deprecated due to the usage of models.Record; models.Record has been deprecated as a
    # structure for uploading due to a `shard_key` field, and now is used only as a result structure.
    # since shard_keys are not supported in migration, we can safely type ignore here and use Records for uploading
    while next_offset is not None:
        records, next_offset = source_client.scroll(
            src_collection_name, offset=next_offset, limit=batch_size, with_vectors=True
        )
        dest_client.upload_points(dest_collection_name, records, wait=True)  # type: ignore
    source_client_vectors_count = source_client.count(src_collection_name).count
    dest_client_vectors_count = dest_client.count(dest_collection_name).count
    assert (
        source_client_vectors_count == dest_client_vectors_count
    ), f"Migration failed, vectors count are not equal: source vector count {source_client_vectors_count}, dest vector count {dest_client_vectors_count}"
