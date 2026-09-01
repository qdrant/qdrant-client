"""Conversions between serverless pydantic models and the internal gRPC types.

The generated gRPC types are an implementation detail and must not leak into
the public interface.
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
    result = pb2.DenseVectorConfig(
        size=model.size,
        distance=_DISTANCE_TO_GRPC[model.distance],
        multivector=model.multivector,
    )
    if model.precision_tier is not None:
        result.precision_tier = _PRECISION_TO_GRPC[model.precision_tier]
    return result


def dense_vector_from_grpc(grpc_model: pb2.DenseVectorConfig) -> models.DenseVectorConfig:
    return models.DenseVectorConfig(
        size=grpc_model.size,
        distance=_DISTANCE_FROM_GRPC[grpc_model.distance],
        multivector=grpc_model.multivector,
        precision_tier=_PRECISION_FROM_GRPC[grpc_model.precision_tier]
        if grpc_model.HasField("precision_tier")
        else None,
    )


def sparse_vector_to_grpc(model: models.SparseVectorConfig) -> pb2.SparseVectorConfig:
    result = pb2.SparseVectorConfig(use_idf=model.use_idf)
    if model.precision_tier is not None:
        result.precision_tier = _PRECISION_TO_GRPC[model.precision_tier]
    return result


def sparse_vector_from_grpc(grpc_model: pb2.SparseVectorConfig) -> models.SparseVectorConfig:
    return models.SparseVectorConfig(
        use_idf=grpc_model.use_idf,
        precision_tier=_PRECISION_FROM_GRPC[grpc_model.precision_tier]
        if grpc_model.HasField("precision_tier")
        else None,
    )


def payload_index_to_grpc(model: models.PayloadIndex) -> pb2.PayloadIndexConfig:
    result = pb2.PayloadIndexConfig()
    if isinstance(model, models.KeywordIndex):
        result.keyword.SetInParent()
    elif isinstance(model, models.IntegerIndex):
        result.integer.SetInParent()
        if model.lookup is not None:
            result.integer.lookup = model.lookup
        if model.range is not None:
            result.integer.range = model.range
    elif isinstance(model, models.FloatIndex):
        result.float.SetInParent()
    elif isinstance(model, models.UuidIndex):
        result.uuid.SetInParent()
    elif isinstance(model, models.DatetimeIndex):
        result.datetime.SetInParent()
    elif isinstance(model, models.TextIndex):
        result.text.SetInParent()
        if model.tokenizer is not None:
            result.text.tokenizer = _TOKENIZER_TO_GRPC[model.tokenizer]
        if model.lowercase is not None:
            result.text.lowercase = model.lowercase
        if model.phrase_matching is not None:
            result.text.phrase_matching = model.phrase_matching
        if model.min_token_len is not None:
            result.text.min_token_len = model.min_token_len
        if model.max_token_len is not None:
            result.text.max_token_len = model.max_token_len
    elif isinstance(model, models.GeoIndex):
        result.geo.SetInParent()
    elif isinstance(model, models.BoolIndex):
        result.bool.SetInParent()
    else:
        raise ValueError(f"Unknown payload index type: {model}")
    return result


def payload_index_from_grpc(grpc_model: pb2.PayloadIndexConfig) -> models.PayloadIndex:
    kind = grpc_model.WhichOneof("index")
    if kind == "keyword":
        return models.KeywordIndex()
    if kind == "integer":
        integer = grpc_model.integer
        return models.IntegerIndex(
            lookup=integer.lookup if integer.HasField("lookup") else None,
            range=integer.range if integer.HasField("range") else None,
        )
    if kind == "float":
        return models.FloatIndex()
    if kind == "uuid":
        return models.UuidIndex()
    if kind == "datetime":
        return models.DatetimeIndex()
    if kind == "text":
        text = grpc_model.text
        return models.TextIndex(
            tokenizer=_TOKENIZER_FROM_GRPC[text.tokenizer]
            if text.HasField("tokenizer")
            else None,
            lowercase=text.lowercase if text.HasField("lowercase") else None,
            phrase_matching=text.phrase_matching if text.HasField("phrase_matching") else None,
            min_token_len=text.min_token_len if text.HasField("min_token_len") else None,
            max_token_len=text.max_token_len if text.HasField("max_token_len") else None,
        )
    if kind == "geo":
        return models.GeoIndex()
    if kind == "bool":
        return models.BoolIndex()
    raise ValueError(f"Unknown payload index type: {kind}")


def collection_config_to_grpc(model: models.CollectionConfig) -> pb2.CollectionConfig:
    result = pb2.CollectionConfig()
    for name, dense in model.dense_vectors.items():
        result.dense_vectors[name].CopyFrom(dense_vector_to_grpc(dense))
    for name, sparse in model.sparse_vectors.items():
        result.sparse_vectors[name].CopyFrom(sparse_vector_to_grpc(sparse))
    for field, index in model.payload_indexes.items():
        result.payload_indexes[field].CopyFrom(payload_index_to_grpc(index))
    return result


def collection_config_from_grpc(grpc_model: pb2.CollectionConfig) -> models.CollectionConfig:
    return models.CollectionConfig(
        dense_vectors={
            name: dense_vector_from_grpc(dense)
            for name, dense in grpc_model.dense_vectors.items()
        },
        sparse_vectors={
            name: sparse_vector_from_grpc(sparse)
            for name, sparse in grpc_model.sparse_vectors.items()
        },
        payload_indexes={
            field: payload_index_from_grpc(index)
            for field, index in grpc_model.payload_indexes.items()
        },
    )
