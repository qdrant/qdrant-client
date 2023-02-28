import numpy as np

from qdrant_client import QdrantClient
from qdrant_client.conversions import common_types as types
from qdrant_client.http import models as rest_models


qdrant_client = QdrantClient()
qdrant_client.clear_payload("collection", [123])
qdrant_client.count(
    "collection", rest_models.Filter()
)
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
qdrant_client.recover_snapshot(
    "collection", "location", rest_models.SnapshotPriority.REPLICA
)
qdrant_client.recreate_collection(
    "collection",
    types.VectorParams(size=128, distance=rest_models.Distance.COSINE),
    2,
    2,
    True,
    True,
    rest_models.HnswConfigDiff(),
    rest_models.OptimizersConfigDiff(),
    rest_models.WalConfigDiff(),
    None,
    5,
)
print(qdrant_client.rest_uri)
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
qdrant_client.set_payload("collection", {}, [], True)
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
            create_alias=rest_models.CreateAlias(
                collection_name="heh", alias_name="hah"
            ),
        )
    ]
)
qdrant_client.upload_records("collection", [])
qdrant_client.upsert("collection", [])

qdrant_client.search(
    "collection", [123], with_payload=["str", "another one", "and another one"]
)
qdrant_client.search(
    "collection", np.array([123]), with_payload=["str", "another one", "and another one"]
)
qdrant_client.upload_collection("collection", [])
