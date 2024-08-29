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
qdrant_client.get_locks()
qdrant_client.list_full_snapshots()
qdrant_client.list_snapshots("collection")
qdrant_client.lock_storage("reason")
qdrant_client.overwrite_payload("collection", {}, [])
qdrant_client.recommend(
    "collection",
    [],
    [],
    rest_models.Filter(),
    rest_models.SearchParams(),
    10,
    0,
    True,
    True,
    1.0,
    "using",
    rest_models.LookupLocation(collection=""),
    rest_models.RecommendStrategy.AVERAGE_VECTOR,
    1,
)
qdrant_client.recommend_batch(
    "collection",
    [
        rest_models.RecommendRequest(
            positive=[],
            negative=[],
            filter=None,
            params=None,
            limit=10,
            offset=0,
            with_payload=True,
            with_vector=True,
            score_threshold=0.5,
            using=None,
            lookup_from=None,
        )
    ],
)
qdrant_client.discover(
    "collection",
    None,
    [],
    rest_models.Filter(),
    rest_models.SearchParams(),
    10,
    0,
    True,
    True,
    "using",
    rest_models.LookupLocation(collection=""),
    1,
)
qdrant_client.discover_batch(
    "collection",
    [
        rest_models.DiscoverRequest(
            target=None,
            context=[],
            filter=rest_models.Filter(),
            params=rest_models.SearchParams(),
            limit=10,
            offset=0,
            with_vector=True,
            with_payload=True,
            using="using",
            lookup_from=rest_models.LookupLocation(collection=""),
        ),
    ],
)
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
    None,
    5,
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
    5,
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
    5,
)
qdrant_client.retrieve("collection", [])
qdrant_client.scroll("collection")
qdrant_client.search_batch(
    "collection",
    [
        rest_models.SearchRequest(
            vector=[1.0, 0.0, 3.0],
            limit=10,
        )
    ],
)
qdrant_client.set_payload("collection", {}, [], key=None, wait=True)
qdrant_client.unlock_storage()
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
qdrant_client.upload_records("collection", [])
qdrant_client.upload_points("collection", [])
qdrant_client.upsert("collection", [])

qdrant_client.search("collection", [123], with_payload=["str", "another one", "and another one"])
# pyright currently is not happy with np.array and treating it as a "partially unknown type"
qdrant_client.search(
    "collection",
    np.array([123]),  # type: ignore
    with_payload=["str", "another one", "and another one"],
)
qdrant_client.upload_collection("collection", [[123]])
qdrant_client.update_vectors("collection", [rest_models.PointVectors(id=1, vector=[123])], False)
qdrant_client.delete_vectors("collection", [], [123, 32, 44])
qdrant_client.search_groups(
    "collection",
    [123],
    "rand_field",
    rest_models.Filter(
        must=[rest_models.FieldCondition(key="field", match=rest_models.MatchValue(value="123"))]
    ),
    rest_models.SearchParams(hnsw_ef=182),
    2,
    3,
    True,
    True,
    0.2,
)
qdrant_client.recommend_groups(
    "collection",
    "rand_field",
    [14],
    [],
    rest_models.Filter(
        must=[rest_models.FieldCondition(key="field", match=rest_models.MatchValue(value="123"))]
    ),
    rest_models.SearchParams(hnsw_ef=182),
    2,
    3,
    3.0,
    True,
    True,
    "using",
    rest_models.LookupLocation(collection="start"),
    None,
)


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
