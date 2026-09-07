"""Pydantic models for the Qdrant Serverless collection management API.

**In development — do not use yet.** Part of the experimental serverless client;
the API may change without notice.

These mirror the tenant-facing serverless config: unlike the regular client's
collection models, they deliberately expose no storage internals (quantization,
WAL, segments, on_disk placement, ...) - the serverless manager decides those.
"""

from enum import Enum
from typing import Literal, Optional, Union

from pydantic import BaseModel, Field

from qdrant_client.http.models import Distance, TokenizerType

__all__ = [
    "Distance",
    "TokenizerType",
    "PrecisionTier",
    "DenseVectorConfig",
    "SparseVectorConfig",
    "KeywordPrefixParams",
    "KeywordIndex",
    "IntegerIndex",
    "FloatIndex",
    "UuidIndex",
    "DatetimeIndex",
    "StopwordsSet",
    "SnowballParams",
    "StemmingAlgorithm",
    "TextIndex",
    "GeoIndex",
    "BoolIndex",
    "PayloadIndex",
    "CollectionConfig",
    "CollectionInfo",
    "CollectionSummary",
    "CollectionsList",
]


class PrecisionTier(str, Enum):
    """How much vector precision may be traded for cost.

    The manager turns this into a concrete quantization / datatype choice.
    """

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class DenseVectorConfig(BaseModel):
    """Configuration of a single dense (embedding) vector."""

    size: int
    distance: Distance
    multivector: bool = False
    precision_tier: Optional[PrecisionTier] = None


class SparseVectorConfig(BaseModel):
    """Configuration of a single sparse vector."""

    use_idf: bool = False
    precision_tier: Optional[PrecisionTier] = None


class KeywordPrefixParams(BaseModel):
    """Prefix matching options for a keyword index. Presence enables prefix matching."""


class KeywordIndex(BaseModel):
    """Exact match on string values, e.g. `color: "red"`."""

    type: Literal["keyword"] = "keyword"
    prefix: Optional[KeywordPrefixParams] = None


class IntegerIndex(BaseModel):
    """Exact match and/or range filters on integers. Both default to enabled."""

    type: Literal["integer"] = "integer"
    lookup: Optional[bool] = None
    range: Optional[bool] = None


class FloatIndex(BaseModel):
    """Range filters on floating point (and integer) numbers."""

    type: Literal["float"] = "float"


class UuidIndex(BaseModel):
    """Exact match on UUID strings; like keyword but stored compactly."""

    type: Literal["uuid"] = "uuid"


class DatetimeIndex(BaseModel):
    """Range filters on RFC 3339 datetimes."""

    type: Literal["datetime"] = "datetime"


class StopwordsSet(BaseModel):
    """Tokens ignored by a full-text index."""

    languages: list[str] = Field(default_factory=list)
    custom: list[str] = Field(default_factory=list)


class SnowballParams(BaseModel):
    """Snowball stemming for a full-text index."""

    language: str


class StemmingAlgorithm(BaseModel):
    """Stemming algorithm for a full-text index. Unset: no stemming.

    Exactly one of `snowball` or `disabled` should be set.
    """

    snowball: Optional[SnowballParams] = None
    disabled: Optional[bool] = None


class TextIndex(BaseModel):
    """Full-text filtering on string values."""

    type: Literal["text"] = "text"
    tokenizer: Optional[TokenizerType] = None
    lowercase: Optional[bool] = None
    phrase_matching: Optional[bool] = None
    min_token_len: Optional[int] = None
    max_token_len: Optional[int] = None
    ascii_folding: Optional[bool] = None
    stopwords: Optional[StopwordsSet] = None
    stemmer: Optional[StemmingAlgorithm] = None


class GeoIndex(BaseModel):
    """Geo radius / bounding box / polygon filters on `{lon, lat}` values."""

    type: Literal["geo"] = "geo"


class BoolIndex(BaseModel):
    """Exact match on booleans."""

    type: Literal["bool"] = "bool"


PayloadIndex = Union[
    KeywordIndex,
    IntegerIndex,
    FloatIndex,
    UuidIndex,
    DatetimeIndex,
    TextIndex,
    GeoIndex,
    BoolIndex,
]


class CollectionConfig(BaseModel):
    """The tenant-facing collection config.

    Vector maps are keyed by vector name; the empty name "" is the unnamed
    default vector. Payload indexes are keyed by payload field name
    (JSON path, e.g. `user_id` or `meta.tags`).
    """

    dense_vectors: dict[str, DenseVectorConfig] = Field(default_factory=dict)
    sparse_vectors: dict[str, SparseVectorConfig] = Field(default_factory=dict)
    payload_indexes: dict[str, PayloadIndex] = Field(default_factory=dict)


class CollectionInfo(BaseModel):
    """A collection's configuration and stats, as returned by `get_collection`.

    `point_count` is eventually consistent and absent until stats have been
    written for the collection.
    """

    exists: bool
    config: Optional[CollectionConfig] = None
    point_count: Optional[int] = None


class CollectionSummary(BaseModel):
    """One collection in a `get_collections` listing."""

    collection_name: str
    point_count: Optional[int] = None


class CollectionsList(BaseModel):
    """A page of collections returned by `get_collections`."""

    collections: list[CollectionSummary]
    next_offset_token: Optional[str] = None
