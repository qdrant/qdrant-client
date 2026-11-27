"""Pydantic models for the Qdrant Serverless collection management API.

**In development — do not use yet.** Part of the experimental serverless client;
the API may change without notice.

These mirror the tenant-facing serverless config: unlike the regular client's
collection models, they deliberately expose no storage internals (quantization,
WAL, segments, on_disk placement, ...) - the serverless manager decides those.

All regular models (`PointStruct`, `Filter`, ...) are re-exported here, so
`qdrant_client.serverless.models` is the single import for serverless code.
Serverless collection models below replace the regular ones with the same name
(`CollectionConfig`, `CollectionInfo`, `DenseVectorConfig`, `SparseVectorConfig`).
`tests/test_serverless.py` checks that no regular model used by serverless
operations is shadowed.
"""

from qdrant_client.models import *  # noqa: F403  # re-export; definitions below take precedence

from enum import Enum
from typing import Literal, Optional, Union

from pydantic import BaseModel, Field

# explicit imports for names used here and in `conversions`: mypy cannot see through the star import
from qdrant_client.http.models import (
    CollectionStatus,
    DisabledStemmerParams,
    Distance,
    Language,
    NoStemmer,
    Snowball,
    SnowballLanguage,
    SnowballParams,
    StemmingAlgorithm,
    StopwordsInterface,
    StopwordsSet,
    TextScoringInterface,
    TextScoringParams,
    TextScoringType,
    TokenizerType,
)


class PrecisionTier(str, Enum):
    """How much vector precision may be traded for cost.

    The manager turns this into a concrete quantization / datatype choice.
    """

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    FULL = "full"


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
    nested: Optional[str] = None
    """Index every object of this array as a separate entry, e.g. `offers` for the
    field `offers[].price`. The field must be inside the array's elements. Default: disabled."""


class IntegerIndex(BaseModel):
    """Exact match and/or range filters on integers. Both default to enabled."""

    type: Literal["integer"] = "integer"
    lookup: Optional[bool] = None
    range: Optional[bool] = None
    nested: Optional[str] = None
    """Index every object of this array as a separate entry, e.g. `offers` for the
    field `offers[].price`. The field must be inside the array's elements. Default: disabled."""


class FloatIndex(BaseModel):
    """Range filters on floating point (and integer) numbers."""

    type: Literal["float"] = "float"
    nested: Optional[str] = None
    """Index every object of this array as a separate entry, e.g. `offers` for the
    field `offers[].price`. The field must be inside the array's elements. Default: disabled."""


class UuidIndex(BaseModel):
    """Exact match on UUID strings; like keyword but stored compactly."""

    type: Literal["uuid"] = "uuid"
    nested: Optional[str] = None
    """Index every object of this array as a separate entry, e.g. `offers` for the
    field `offers[].id`. The field must be inside the array's elements. Default: disabled."""


class DatetimeIndex(BaseModel):
    """Range filters on RFC 3339 datetimes."""

    type: Literal["datetime"] = "datetime"
    nested: Optional[str] = None
    """Index every object of this array as a separate entry, e.g. `offers` for the
    field `offers[].updated_at`. The field must be inside the array's elements. Default: disabled."""


class TextIndex(BaseModel):
    """Full-text filtering on string values."""

    type: Literal["text"] = "text"
    tokenizer: Optional[TokenizerType] = None
    lowercase: Optional[bool] = None
    phrase_matching: Optional[bool] = None
    min_token_len: Optional[int] = None
    max_token_len: Optional[int] = None
    ascii_folding: Optional[bool] = None
    stopwords: Optional[StopwordsInterface] = None
    stemmer: Optional[StemmingAlgorithm] = None
    scoring: Optional[TextScoringInterface] = None
    """Enable ranking points by BM25 over this field: `True` for the defaults or
    `TextScoringParams(type=TextScoringType.BM25)`. `False` or unset: disabled.
    Changing it rebuilds the index."""
    nested: Optional[str] = None
    """Index every object of this array as a separate entry, e.g. `offers` for the
    field `offers[].title`. Every query on the field treats each element as its own
    document: BM25 scores elements separately. Default: disabled."""


class GeoIndex(BaseModel):
    """Geo radius / bounding box / polygon filters on `{lon, lat}` values."""

    type: Literal["geo"] = "geo"
    nested: Optional[str] = None
    """Index every object of this array as a separate entry, e.g. `offers` for the
    field `offers[].location`. The field must be inside the array's elements. Default: disabled."""


class BoolIndex(BaseModel):
    """Exact match on booleans."""

    type: Literal["bool"] = "bool"
    nested: Optional[str] = None
    """Index every object of this array as a separate entry, e.g. `offers` for the
    field `offers[].in_stock`. The field must be inside the array's elements. Default: disabled."""


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


class CreateCollectionResult(BaseModel):
    """Result of `create_collection`."""

    collection_name: str
    result: str
    time: float


class DeleteCollectionResult(BaseModel):
    """Result of `delete_collection`."""

    deleted: bool
    time: float


class UpdatesStatus(str, Enum):
    """Whether acknowledged writes have been applied."""

    APPLIED = "applied"
    """Every acknowledged write is applied and visible to search."""
    PENDING = "pending"
    """Some acknowledged writes are still queued and not yet visible to search."""


class UpdatesInfo(BaseModel):
    """Progress of acknowledged writes to a collection."""

    status: UpdatesStatus


class CollectionInfo(BaseModel):
    """A collection's configuration and stats, as returned by `get_collection`.

    `point_count` is eventually consistent and absent until stats have been
    written for the collection. `status` is `green` or `yellow` (optimization
    pending or running), or `None` if it could not be determined.
    `size_bytes` is the estimated stored size the collection size limit applies
    to. `indexed_vectors_count` is approximate. `updates` is absent when the
    progress of acknowledged writes could not be determined.
    """

    exists: bool
    config: Optional[CollectionConfig] = None
    point_count: Optional[int] = None
    status: Optional[CollectionStatus] = None
    size_bytes: Optional[int] = None
    indexed_vectors_count: Optional[int] = None
    updates: Optional[UpdatesInfo] = None
    time: float


class CollectionSummary(BaseModel):
    """One collection in a `get_collections` listing."""

    collection_name: str
    point_count: Optional[int] = None
    size_bytes: Optional[int] = None


class CollectionsList(BaseModel):
    """A page of collections returned by `get_collections`."""

    collections: list[CollectionSummary]
    next_offset_token: Optional[str] = None
    time: float
