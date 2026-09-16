"""Client for Qdrant Serverless.

**In development — do not use yet.** This API is experimental and unstable;
it may change without notice and is not ready for production or general use.

Point-level operations (query, upsert, ...) behave like the regular client;
collection management uses the simplified, tenant-facing serverless API.

Usage:

    from qdrant_client.serverless import QdrantServerless
    from qdrant_client.serverless.models import DenseVectorConfig, Distance

    client = QdrantServerless(url="https://...", api_key="...")
    client.create_collection(
        "my-collection",
        dense_vectors=DenseVectorConfig(size=1536, distance=Distance.COSINE),
    )
    client.query_points("my-collection", query=[0.1, 0.2, ...])
"""

from qdrant_client.serverless.async_client import AsyncQdrantServerless
from qdrant_client.serverless.client import QdrantServerless

__all__ = [
    "AsyncQdrantServerless",
    "QdrantServerless",
]
