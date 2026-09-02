"""Conversions between serverless pydantic models and the internal gRPC types.

**In development — do not use yet.** Part of the experimental serverless client.

The generated gRPC types are an implementation detail and must not leak into
the public interface.

All model/proto fields are bound by structural pattern matching (never via
`model.field` or ignored with ``*_``) so adding a field forces an update here.
"""

from qdrant_client.serverless import models
from qdrant_client.serverless.grpc import serverless_collections_pb2 as pb2

_DISTANCE_TO_GRPC = {
    models.Distance.COSINE: pb2.COSINE,
    models.Distance.EUCLID: pb2.EUCLID,
    models.Distance.DOT: pb2.DOT,
    models.Distance.MANHATTAN: pb2.MANHATTAN,
}
_DISTANCE_FROM_GRPC = {v: k for k, v in _DISTANCE_TO_GRPC.items()}

_PRECISION_TO_GRPC = {
    models.PrecisionTier.LOW: pb2.LOW,
    models.PrecisionTier.MEDIUM: pb2.MEDIUM,
    models.PrecisionTier.HIGH: pb2.HIGH,
}
_PRECISION_FROM_GRPC = {v: k for k, v in _PRECISION_TO_GRPC.items()}

_TOKENIZER_TO_GRPC = {
    models.TokenizerType.PREFIX: pb2.PREFIX,
    models.TokenizerType.WHITESPACE: pb2.WHITESPACE,
    models.TokenizerType.WORD: pb2.WORD,
    models.TokenizerType.MULTILINGUAL: pb2.MULTILINGUAL,
}
_TOKENIZER_FROM_GRPC = {v: k for k, v in _TOKENIZER_TO_GRPC.items()}


def dense_vector_to_grpc(model: models.DenseVectorConfig) -> pb2.DenseVectorConfig:
    match model:
        case models.DenseVectorConfig(
            size=size,
            distance=distance,
            multivector=multivector,
            precision_tier=precision_tier,
        ):
            result = pb2.DenseVectorConfig(
                size=size,
                distance=_DISTANCE_TO_GRPC[distance],
                multivector=multivector,
            )
            if precision_tier is not None:
                result.precision_tier = _PRECISION_TO_GRPC[precision_tier]
            return result
        case _:
            raise ValueError(f"Unexpected DenseVectorConfig shape: {model!r}")


def dense_vector_from_grpc(grpc_model: pb2.DenseVectorConfig) -> models.DenseVectorConfig:
    size = grpc_model.size
    distance = grpc_model.distance
    multivector = grpc_model.multivector
    precision_tier = (
        _PRECISION_FROM_GRPC[grpc_model.precision_tier]
        if grpc_model.HasField("precision_tier")
        else None
    )
    # Re-bind through the public model constructor so every field is named.
    return models.DenseVectorConfig(
        size=size,
        distance=_DISTANCE_FROM_GRPC[distance],
        multivector=multivector,
        precision_tier=precision_tier,
    )


def sparse_vector_to_grpc(model: models.SparseVectorConfig) -> pb2.SparseVectorConfig:
    match model:
        case models.SparseVectorConfig(use_idf=use_idf, precision_tier=precision_tier):
            result = pb2.SparseVectorConfig(use_idf=use_idf)
            if precision_tier is not None:
                result.precision_tier = _PRECISION_TO_GRPC[precision_tier]
            return result
        case _:
            raise ValueError(f"Unexpected SparseVectorConfig shape: {model!r}")


def sparse_vector_from_grpc(grpc_model: pb2.SparseVectorConfig) -> models.SparseVectorConfig:
    use_idf = grpc_model.use_idf
    precision_tier = (
        _PRECISION_FROM_GRPC[grpc_model.precision_tier]
        if grpc_model.HasField("precision_tier")
        else None
    )
    return models.SparseVectorConfig(use_idf=use_idf, precision_tier=precision_tier)


def payload_index_to_grpc(model: models.PayloadIndex) -> pb2.PayloadIndexConfig:
    result = pb2.PayloadIndexConfig()
    match model:
        case models.KeywordIndex(type=_type):
            result.keyword.SetInParent()
        case models.IntegerIndex(type=_type, lookup=lookup, range=range_):
            result.integer.SetInParent()
            if lookup is not None:
                result.integer.lookup = lookup
            if range_ is not None:
                result.integer.range = range_
        case models.FloatIndex(type=_type):
            result.float.SetInParent()
        case models.UuidIndex(type=_type):
            result.uuid.SetInParent()
        case models.DatetimeIndex(type=_type):
            result.datetime.SetInParent()
        case models.TextIndex(
            type=_type,
            tokenizer=tokenizer,
            lowercase=lowercase,
            phrase_matching=phrase_matching,
            min_token_len=min_token_len,
            max_token_len=max_token_len,
        ):
            result.text.SetInParent()
            if tokenizer is not None:
                result.text.tokenizer = _TOKENIZER_TO_GRPC[tokenizer]
            if lowercase is not None:
                result.text.lowercase = lowercase
            if phrase_matching is not None:
                result.text.phrase_matching = phrase_matching
            if min_token_len is not None:
                result.text.min_token_len = min_token_len
            if max_token_len is not None:
                result.text.max_token_len = max_token_len
        case models.GeoIndex(type=_type):
            result.geo.SetInParent()
        case models.BoolIndex(type=_type):
            result.bool.SetInParent()
        case _:
            raise ValueError(f"Unknown payload index type: {model}")
    return result


def payload_index_from_grpc(grpc_model: pb2.PayloadIndexConfig) -> models.PayloadIndex:
    kind = grpc_model.WhichOneof("index")
    if kind == "keyword":
        _keyword = grpc_model.keyword
        return models.KeywordIndex()
    if kind == "integer":
        integer = grpc_model.integer
        lookup = integer.lookup if integer.HasField("lookup") else None
        range_ = integer.range if integer.HasField("range") else None
        return models.IntegerIndex(lookup=lookup, range=range_)
    if kind == "float":
        _float = grpc_model.float
        return models.FloatIndex()
    if kind == "uuid":
        _uuid = grpc_model.uuid
        return models.UuidIndex()
    if kind == "datetime":
        _datetime = grpc_model.datetime
        return models.DatetimeIndex()
    if kind == "text":
        text = grpc_model.text
        tokenizer = (
            _TOKENIZER_FROM_GRPC[text.tokenizer] if text.HasField("tokenizer") else None
        )
        lowercase = text.lowercase if text.HasField("lowercase") else None
        phrase_matching = (
            text.phrase_matching if text.HasField("phrase_matching") else None
        )
        min_token_len = text.min_token_len if text.HasField("min_token_len") else None
        max_token_len = text.max_token_len if text.HasField("max_token_len") else None
        return models.TextIndex(
            tokenizer=tokenizer,
            lowercase=lowercase,
            phrase_matching=phrase_matching,
            min_token_len=min_token_len,
            max_token_len=max_token_len,
        )
    if kind == "geo":
        _geo = grpc_model.geo
        return models.GeoIndex()
    if kind == "bool":
        _bool = grpc_model.bool
        return models.BoolIndex()
    raise ValueError(f"Unknown payload index type: {kind}")


def collection_config_to_grpc(model: models.CollectionConfig) -> pb2.CollectionConfig:
    match model:
        case models.CollectionConfig(
            dense_vectors=dense_vectors,
            sparse_vectors=sparse_vectors,
            payload_indexes=payload_indexes,
        ):
            result = pb2.CollectionConfig()
            for name, dense in dense_vectors.items():
                result.dense_vectors[name].CopyFrom(dense_vector_to_grpc(dense))
            for name, sparse in sparse_vectors.items():
                result.sparse_vectors[name].CopyFrom(sparse_vector_to_grpc(sparse))
            for field, index in payload_indexes.items():
                result.payload_indexes[field].CopyFrom(payload_index_to_grpc(index))
            return result
        case _:
            raise ValueError(f"Unexpected CollectionConfig shape: {model!r}")


def collection_config_from_grpc(grpc_model: pb2.CollectionConfig) -> models.CollectionConfig:
    dense_vectors = {
        name: dense_vector_from_grpc(dense)
        for name, dense in grpc_model.dense_vectors.items()
    }
    sparse_vectors = {
        name: sparse_vector_from_grpc(sparse)
        for name, sparse in grpc_model.sparse_vectors.items()
    }
    payload_indexes = {
        field: payload_index_from_grpc(index)
        for field, index in grpc_model.payload_indexes.items()
    }
    return models.CollectionConfig(
        dense_vectors=dense_vectors,
        sparse_vectors=sparse_vectors,
        payload_indexes=payload_indexes,
    )
