"""Client for Qdrant Serverless.

Point-level operations (query, upsert, ...) behave like the regular client;
collection management uses the simplified, tenant-facing serverless API.

Usage:

    from qdrant_client.serverless import QdrantServerless, DenseVectorConfig, Distance

    client = QdrantServerless(url="https://...", api_key="...")
    client.create_collection(
        "my-collection",
        dense_vectors=DenseVectorConfig(size=1536, distance=Distance.COSINE),
    )
    client.query_points("my-collection", query=[0.1, 0.2, ...])
"""

from qdrant_client.serverless.async_client import AsyncQdrantServerless
from qdrant_client.serverless.client import QdrantServerless
from qdrant_client.serverless.models import (
    BoolIndex,
    CollectionConfig,
    CollectionInfo,
    CollectionSummary,
    DatetimeIndex,
    DenseVectorConfig,
    Distance,
    FloatIndex,
    GeoIndex,
    IntegerIndex,
    KeywordIndex,
    PayloadIndex,
    PrecisionTier,
    SparseVectorConfig,
    TextIndex,
    TokenizerType,
    UuidIndex,
)

__all__ = [
    "AsyncQdrantServerless",
    "QdrantServerless",
    "BoolIndex",
    "CollectionConfig",
    "CollectionInfo",
    "CollectionSummary",
    "DatetimeIndex",
    "DenseVectorConfig",
    "Distance",
    "FloatIndex",
    "GeoIndex",
    "IntegerIndex",
    "KeywordIndex",
    "PayloadIndex",
    "PrecisionTier",
    "SparseVectorConfig",
    "TextIndex",
    "TokenizerType",
    "UuidIndex",
]
