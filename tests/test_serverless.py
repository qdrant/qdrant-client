import inspect
from typing import Optional

from qdrant_client.serverless.models import (
    CollectionConfig,
    DenseVectorConfig,
    Distance,
    IntegerIndex,
    KeywordIndex,
    KeywordPrefixParams,
    PrecisionTier,
    Language,
    Snowball,
    SnowballLanguage,
    SnowballParams,
    SparseVectorConfig,
    StopwordsSet,
    TextIndex,
    TokenizerType,
)
from qdrant_client.serverless import AsyncQdrantServerless, QdrantServerless
from qdrant_client.serverless.grpc import serverless_collections_pb2 as pb2
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
            "user_id": KeywordIndex(prefix=KeywordPrefixParams()),
            "age": IntegerIndex(lookup=True, range=False),
            "description": TextIndex(
                tokenizer=TokenizerType.WORD,
                lowercase=False,
                ascii_folding=True,
                stopwords=StopwordsSet(languages=[Language.ENGLISH]),
                stemmer=SnowballParams(type=Snowball.SNOWBALL, language=SnowballLanguage.ENGLISH),
            ),
        },
    )
    assert collection_config_from_grpc(collection_config_to_grpc(config)) == config


def test_optional_fields_stay_unset() -> None:
    config = CollectionConfig(
        dense_vectors={"": DenseVectorConfig(size=4, distance=Distance.EUCLID)},
        payload_indexes={
            "age": IntegerIndex(),
            "user_id": KeywordIndex(),
            "text": TextIndex(),
        },
    )
    grpc_config = collection_config_to_grpc(config)
    assert not grpc_config.dense_vectors[""].HasField("precision_tier")
    assert not grpc_config.payload_indexes["age"].integer.HasField("lookup")
    assert not grpc_config.payload_indexes["user_id"].keyword.HasField("prefix")
    assert not grpc_config.payload_indexes["text"].text.HasField("tokenizer")
    assert not grpc_config.payload_indexes["text"].text.HasField("ascii_folding")
    assert not grpc_config.payload_indexes["text"].text.HasField("stopwords")
    assert not grpc_config.payload_indexes["text"].text.HasField("stemmer")
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


def test_serverless_models_shadow_only_collection_models() -> None:
    """A serverless model replacing a regular one of the same name must be deliberate."""
    from qdrant_client import models as regular
    from qdrant_client.serverless import models as serverless_models

    shadowed = {
        name
        for name, obj in vars(serverless_models).items()
        if not name.startswith("_") and getattr(regular, name, obj) is not obj
    }
    assert shadowed == {
        "CollectionConfig",
        "CollectionInfo",
        "DenseVectorConfig",
        "SparseVectorConfig",
    }
    assert serverless_models.PointStruct is regular.PointStruct


def test_text_index_accepts_single_stopwords_language() -> None:
    config = CollectionConfig(payload_indexes={"t": TextIndex(stopwords=Language.ENGLISH)})
    grpc_config = collection_config_to_grpc(config)
    assert list(grpc_config.payload_indexes["t"].text.stopwords.languages) == ["english"]


def test_text_index_bm25_scoring_and_code_tokenizer() -> None:
    from qdrant_client.serverless.models import TextScoringParams, TextScoringType

    config = CollectionConfig(
        payload_indexes={
            "default": TextIndex(scoring=True),
            "explicit": TextIndex(
                tokenizer=TokenizerType.CODE,
                scoring=TextScoringParams(type=TextScoringType.BM25),
            ),
            "disabled": TextIndex(scoring=False),
        }
    )
    grpc_config = collection_config_to_grpc(config)
    default = grpc_config.payload_indexes["default"].text
    assert default.HasField("scoring") and not default.scoring.HasField("type")
    explicit = grpc_config.payload_indexes["explicit"].text
    assert explicit.tokenizer == pb2.CODE
    assert explicit.scoring.type == pb2.BM25
    # `False` disables scoring: it is sent as an unset field and reads back as unset
    assert not grpc_config.payload_indexes["disabled"].text.HasField("scoring")
    assert collection_config_from_grpc(grpc_config).payload_indexes == {
        "default": TextIndex(scoring=True),
        "explicit": config.payload_indexes["explicit"],
        "disabled": TextIndex(),
    }


def test_text_index_explicit_unspecified_scoring_type_reads_as_default() -> None:
    grpc_config = collection_config_to_grpc(
        CollectionConfig(payload_indexes={"t": TextIndex(scoring=True)})
    )
    text = grpc_config.payload_indexes["t"].text
    text.scoring.type = pb2.TEXT_SCORING_TYPE_UNSPECIFIED
    assert text.scoring.HasField("type")
    assert collection_config_from_grpc(grpc_config).payload_indexes == {
        "t": TextIndex(scoring=True)
    }


def test_get_collection_and_listing_expose_stats() -> None:
    from qdrant_client.serverless.models import CollectionStatus, UpdatesInfo, UpdatesStatus

    full = pb2.GetCollectionResponse(
        exists=True,
        point_count=10,
        status=pb2.YELLOW,
        size_bytes=2048,
        indexed_vectors_count=7,
        updates=pb2.UpdatesInfo(status=pb2.PENDING),
        time=0.5,
    )
    # unset status is COLLECTION_STATUS_UNSPECIFIED: could not be determined
    missing = pb2.GetCollectionResponse(exists=False, time=0.1)

    class FakeCollections:
        def __init__(self) -> None:
            self.responses = [full, missing]

        def GetCollection(
            self, request: pb2.GetCollectionRequest, timeout: Optional[float] = None
        ) -> pb2.GetCollectionResponse:
            return self.responses.pop(0)

        def ListCollections(
            self, request: pb2.ListCollectionsRequest, timeout: Optional[float] = None
        ) -> pb2.ListCollectionsResponse:
            return pb2.ListCollectionsResponse(
                collections=[pb2.CollectionSummary(collection_name="a", size_bytes=4096)],
                time=0.1,
            )

    client = QdrantServerless(url="https://serverless.example.qdrant.io", api_key="secret")
    client._grpc_collections = FakeCollections()  # type: ignore[assignment]

    info = client.get_collection("a")
    assert info.status == CollectionStatus.YELLOW
    assert info.size_bytes == 2048
    assert info.indexed_vectors_count == 7
    assert info.updates == UpdatesInfo(status=UpdatesStatus.PENDING)

    absent = client.get_collection("b")
    assert (absent.status, absent.size_bytes, absent.updates) == (None, None, None)

    summary = client.get_collections().collections[0]
    assert (summary.point_count, summary.size_bytes) == (None, 4096)
    client.close()
