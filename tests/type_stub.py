import numpy as np

from qdrant_client import QdrantClient
from qdrant_client import models as rest_models
from qdrant_client.conversions import common_types as types
from qdrant_client.models import (
    CompressionRatio,
    ProductQuantizationConfig,
    ScalarQuantizationConfig,
    ScalarType,
)

qdrant_client = QdrantClient(timeout=30)
qdrant_client.clear_payload("collection", [123])
qdrant_client.count("collection", rest_models.Filter())
qdrant_client.close(grpc_grace=1.5)
qdrant_client.create_full_snapshot()
qdrant_client.create_payload_index("collection", "asd", 3)
qdrant_client.delete("collection", [123])
qdrant_client.delete_collection("collection")
qdrant_client.delete_payload("collection", ["key"], [1])
qdrant_client.delete_payload_index("collection", "field_name")
qdrant_client.delete_snapshot("collection", "sn_name")
qdrant_client.delete_full_snapshot("collection")
qdrant_client.get_collection_aliases("collection")
qdrant_client.get_aliases()
qdrant_client.get_collection("collection")
qdrant_client.collection_exists("collection")
qdrant_client.get_collections()
qdrant_client.list_full_snapshots()
qdrant_client.list_snapshots("collection")
qdrant_client.overwrite_payload("collection", {}, [])
qdrant_client.recover_snapshot("collection", "location", rest_models.SnapshotPriority.REPLICA)
qdrant_client.create_collection(
    "collection",
    types.VectorParams(size=128, distance=rest_models.Distance.COSINE),
    {
        "field": rest_models.SparseVectorParams(
            index=rest_models.SparseIndexParams(
                full_scan_threshold=1000,
                on_disk=False,
            )
        )
    },
    2,
    rest_models.ShardingMethod.AUTO,
    2,
    True,
    True,
    rest_models.HnswConfigDiff(),
    rest_models.OptimizersConfigDiff(),
    rest_models.WalConfigDiff(),
    rest_models.ScalarQuantization(scalar=ScalarQuantizationConfig(type=ScalarType.INT8)),
    5,
    rest_models.StrictModeConfig(),
    {},
)
qdrant_client.recreate_collection(
    "collection",
    types.VectorParams(size=128, distance=rest_models.Distance.COSINE),
    {
        "field": rest_models.SparseVectorParams(
            index=rest_models.SparseIndexParams(
                full_scan_threshold=1000,
                on_disk=False,
            )
        )
    },
    2,
    rest_models.ShardingMethod.AUTO,
    2,
    True,
    True,
    rest_models.HnswConfigDiff(),
    rest_models.OptimizersConfigDiff(),
    rest_models.WalConfigDiff(),
    rest_models.ScalarQuantization(scalar=ScalarQuantizationConfig(type=ScalarType.INT8)),
    None,
    rest_models.StrictModeConfig(),
)
qdrant_client.recreate_collection(
    "collection",
    types.VectorParams(size=128, distance=rest_models.Distance.COSINE),
    {
        "field": rest_models.SparseVectorParams(
            index=rest_models.SparseIndexParams(
                full_scan_threshold=1000,
                on_disk=False,
            )
        )
    },
    2,
    rest_models.ShardingMethod.AUTO,
    2,
    True,
    True,
    rest_models.HnswConfigDiff(),
    rest_models.OptimizersConfigDiff(),
    rest_models.WalConfigDiff(),
    rest_models.ProductQuantization(
        product=ProductQuantizationConfig(compression=CompressionRatio.X32)
    ),
    None,
    rest_models.StrictModeConfig(),
)
qdrant_client.retrieve("collection", [])
qdrant_client.scroll("collection")
qdrant_client.set_payload("collection", {}, [], key=None, wait=True)
qdrant_client.update_collection(
    "collection",
    rest_models.OptimizersConfigDiff(
        deleted_threshold=0.5,
        vacuum_min_vector_number=1000,
        default_segment_number=3,
        max_segment_size=2,
        memmap_threshold=3,
        indexing_threshold=5,
        flush_interval_sec=3000,
        max_optimization_threads=1,
    ),
)
qdrant_client.update_collection_aliases(
    [
        rest_models.CreateAliasOperation(
            create_alias=rest_models.CreateAlias(collection_name="heh", alias_name="hah"),
        )
    ]
)
qdrant_client.upload_points("collection", [])
qdrant_client.upsert("collection", [])
qdrant_client.upload_collection("collection", [[123]])
qdrant_client.update_vectors("collection", [rest_models.PointVectors(id=1, vector=[123])], False)
qdrant_client.delete_vectors("collection", [], [123, 32, 44])
qdrant_client.batch_update_points(
    collection_name="batchcollection",
    update_operations=[
        rest_models.UpsertOperation(
            upsert=rest_models.PointsList(
                points=[rest_models.PointStruct(vector=[0.1, 0.2], id=3, payload={})]
            )
        )
    ],
)
qdrant_client.create_snapshot(collection_name="createsnapshot", wait=False)
qdrant_client.list_shard_snapshots(collection_name="listshardsnapshots", shard_id=3)
qdrant_client.create_shard_snapshot(collection_name="createshardsnapshot", shard_id=3)
qdrant_client.delete_shard_snapshot(
    collection_name="deleteshardsnapshot", shard_id=3, snapshot_name="snapshot_id", wait=False
)
qdrant_client.recover_shard_snapshot(
    collection_name="recovershardsnapshot",
    shard_id=3,
    location="nowhere",
    priority=rest_models.SnapshotPriority.NO_SYNC,
)
qdrant_client.create_shard_key(
    collection_name="qwerty",
    shard_key="new_key",
    shards_number=3,
    replication_factor=2,
    placement=[23],
)
qdrant_client.delete_shard_key(collection_name="zcxzc", shard_key="broken_key")
qdrant_client.migrate(
    dest_client=QdrantClient(),
    collection_names=["collection"],
    batch_size=1,
    recreate_on_collision=False,
)
qdrant_client.query_batch_points(
    collection_name="collection",
    requests=[rest_models.QueryRequest()],
    consistency=None,
    timeout=1,
)
qdrant_client.query_points(
    collection_name="collection",
    query=[0.1, 0.1, 0.1],
    using="",
    prefetch=None,
    query_filter=None,
    search_params=None,
    limit=10,
    offset=None,
    with_payload=True,
    with_vectors=False,
    score_threshold=0.9,
    lookup_from=None,
    consistency=None,
    shard_key_selector=None,
    timeout=1,
)
qdrant_client.facet(
    collection_name="collection",
    key="field",
    facet_filter=rest_models.Filter(),
    exact=True,
    limit=10,
    consistency=None,
    shard_key_selector=None,
    timeout=1,
)
