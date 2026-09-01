import inspect

from qdrant_client.models.serverless import (
    CollectionConfig,
    DenseVectorConfig,
    Distance,
    IntegerIndex,
    KeywordIndex,
    PrecisionTier,
    SparseVectorConfig,
    TextIndex,
    TokenizerType,
)
from qdrant_client.serverless import AsyncQdrantServerless, QdrantServerless
from qdrant_client.serverless.conversions import (
    collection_config_from_grpc,
    collection_config_to_grpc,
)


def test_collection_config_grpc_roundtrip() -> None:
    config = CollectionConfig(
        dense_vectors={
            "": DenseVectorConfig(size=1536, distance=Distance.COSINE),
            "colbert": DenseVectorConfig(
                size=128,
                distance=Distance.DOT,
                multivector=True,
                precision_tier=PrecisionTier.LOW,
            ),
        },
        sparse_vectors={"bm25": SparseVectorConfig(use_idf=True)},
        payload_indexes={
            "user_id": KeywordIndex(),
            "age": IntegerIndex(lookup=True, range=False),
            "description": TextIndex(tokenizer=TokenizerType.WORD, lowercase=False),
        },
    )
    assert collection_config_from_grpc(collection_config_to_grpc(config)) == config


def test_optional_fields_stay_unset() -> None:
    config = CollectionConfig(
        dense_vectors={"": DenseVectorConfig(size=4, distance=Distance.EUCLID)},
        payload_indexes={"age": IntegerIndex(), "text": TextIndex()},
    )
    grpc_config = collection_config_to_grpc(config)
    assert not grpc_config.dense_vectors[""].HasField("precision_tier")
    assert not grpc_config.payload_indexes["age"].integer.HasField("lookup")
    assert not grpc_config.payload_indexes["text"].text.HasField("tokenizer")
    assert collection_config_from_grpc(grpc_config) == config


def test_client_construction_is_offline() -> None:
    client = QdrantServerless(url="https://serverless.example.qdrant.io", api_key="secret")
    assert ("api-key", "secret") in client._remote._grpc_headers
    assert client._remote._grpc_port == 443
    assert client._remote._https
    client.close()


def test_async_client_mirrors_sync_client() -> None:
    sync_methods = {
        name
        for name, _ in inspect.getmembers(QdrantServerless, predicate=inspect.isfunction)
        if not name.startswith("__")
    }
    async_methods = {
        name
        for name, _ in inspect.getmembers(AsyncQdrantServerless, predicate=inspect.isfunction)
        if not name.startswith("__")
    }
    assert sync_methods == async_methods
    for name in async_methods:
        if name.startswith("_"):
            continue
        assert inspect.iscoroutinefunction(getattr(AsyncQdrantServerless, name)), name

    client = AsyncQdrantServerless(url="https://serverless.example.qdrant.io", api_key="secret")
    assert ("api-key", "secret") in client._remote._grpc_headers
    assert client._remote._grpc_port == 443
