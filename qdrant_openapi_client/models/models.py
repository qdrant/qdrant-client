from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field


class AliasOperationsAnyOf(BaseModel):
    """
    Create alternative name for a collection. Collection will be available under both names for search, retrieve,
    """

    create_alias: "AliasOperationsAnyOfCreateAlias" = Field(
        ...,
        description="Create alternative name for a collection. Collection will be available under both names for search, retrieve,",
    )


class AliasOperationsAnyOf1(BaseModel):
    """
    Delete alias if exists
    """

    delete_alias: "AliasOperationsAnyOf1DeleteAlias" = Field(..., description="Delete alias if exists")


class AliasOperationsAnyOf1DeleteAlias(BaseModel):
    alias_name: str = Field(..., description="")


class AliasOperationsAnyOf2(BaseModel):
    """
    Change alias to a new one
    """

    rename_alias: "AliasOperationsAnyOf2RenameAlias" = Field(..., description="Change alias to a new one")


class AliasOperationsAnyOf2RenameAlias(BaseModel):
    new_alias_name: str = Field(..., description="")
    old_alias_name: str = Field(..., description="")


class AliasOperationsAnyOfCreateAlias(BaseModel):
    alias_name: str = Field(..., description="")
    collection_name: str = Field(..., description="")


class CollectionConfig(BaseModel):
    hnsw_config: "HnswConfig" = Field(..., description="")
    optimizer_config: "OptimizersConfig" = Field(..., description="")
    params: "CollectionParams" = Field(..., description="")
    wal_config: "WalConfig" = Field(..., description="")


class CollectionDescription(BaseModel):
    name: str = Field(..., description="")


class CollectionInfo(BaseModel):
    """
    Current statistics and configuration of the collection.
    """

    config: "CollectionConfig" = Field(..., description="Current statistics and configuration of the collection.")
    disk_data_size: int = Field(..., description="Disk space, used by collection")
    payload_schema: Dict[str, "PayloadSchemaInfo"] = Field(..., description="Types of stored payload")
    ram_data_size: int = Field(..., description="RAM used by collection")
    segments_count: int = Field(..., description="Number of segments in collection")
    status: "CollectionStatus" = Field(..., description="Current statistics and configuration of the collection.")
    vectors_count: int = Field(..., description="Number of vectors in collection")


class CollectionParams(BaseModel):
    distance: "Distance" = Field(..., description="")
    vector_size: int = Field(..., description="Size of a vectors used")


class CollectionStatus(str, Enum):
    GREEN = "green"
    YELLOW = "yellow"
    RED = "red"


class CollectionsResponse(BaseModel):
    collections: List["CollectionDescription"] = Field(..., description="")


class Distance(str, Enum):
    COSINE = "Cosine"
    EUCLID = "Euclid"
    DOT = "Dot"


class ErrorResponse(BaseModel):
    time: Optional[float] = Field(None, description="Time spent to process this request")
    status: Optional["ErrorResponseStatus"] = Field(None, description="")
    result: Optional[Any] = Field(None, description="")


class ErrorResponseStatus(BaseModel):
    error: Optional[str] = Field(None, description="Description of the occurred error.")


class FieldCondition(BaseModel):
    geo_bounding_box: Optional["GeoBoundingBox"] = Field(
        None, description="Check if points geo location lies in a given area"
    )
    geo_radius: Optional["GeoRadius"] = Field(None, description="Check if geo point is within a given radius")
    key: str = Field(..., description="")
    match: Optional["Match"] = Field(None, description="Check if point has field with a given value")
    range: Optional["Range"] = Field(None, description="Check if points value lies in a given range")


class FieldIndexOperationsAnyOf(BaseModel):
    """
    Create index for payload field
    """

    create_index: str = Field(..., description="Create index for payload field")


class FieldIndexOperationsAnyOf1(BaseModel):
    """
    Delete index for the field
    """

    delete_index: str = Field(..., description="Delete index for the field")


class Filter(BaseModel):
    must: Optional[List["Condition"]] = Field(None, description="All conditions must match")
    must_not: Optional[List["Condition"]] = Field(None, description="All conditions must NOT match")
    should: Optional[List["Condition"]] = Field(None, description="At least one of thous conditions should match")


class GeoBoundingBox(BaseModel):
    bottom_right: "GeoPoint" = Field(..., description="")
    top_left: "GeoPoint" = Field(..., description="")


class GeoPoint(BaseModel):
    lat: float = Field(..., description="")
    lon: float = Field(..., description="")


class GeoRadius(BaseModel):
    center: "GeoPoint" = Field(..., description="")
    radius: float = Field(..., description="Radius of the area in meters")


class HasIdCondition(BaseModel):
    has_id: List[int] = Field(..., description="")


class HnswConfig(BaseModel):
    ef_construct: int = Field(
        ...,
        description="Number of neighbours to consider during the index building. Larger the value - more accurate the search, more time required to build index.",
    )
    full_scan_threshold: int = Field(
        ...,
        description="Minimal amount of points for additional payload-based indexing. If payload chunk is smaller than `full_scan_threshold` additional indexing won&#x27;t be used - in this case full-scan search should be preferred by query planner and additional indexing is not required.",
    )
    m: int = Field(
        ...,
        description="Number of edges per node in the index graph. Larger the value - more accurate the search, more space required.",
    )


class HnswConfigDiff(BaseModel):
    ef_construct: Optional[int] = Field(
        None,
        description="Number of neighbours to consider during the index building. Larger the value - more accurate the search, more time required to build index.",
    )
    full_scan_threshold: Optional[int] = Field(
        None,
        description="Minimal amount of points for additional payload-based indexing. If payload chunk is smaller than `full_scan_threshold` additional indexing won&#x27;t be used - in this case full-scan search should be preferred by query planner and additional indexing is not required.",
    )
    m: Optional[int] = Field(
        None,
        description="Number of edges per node in the index graph. Larger the value - more accurate the search, more space required.",
    )


class InlineResponse200(BaseModel):
    time: Optional[float] = Field(None, description="Time spent to process this request")
    status: Literal[
        "ok",
    ] = Field(None, description="")
    result: Optional["CollectionsResponse"] = Field(None, description="")


class InlineResponse2001(BaseModel):
    time: Optional[float] = Field(None, description="Time spent to process this request")
    status: Literal[
        "ok",
    ] = Field(None, description="")
    result: Optional[bool] = Field(None, description="")


class InlineResponse2002(BaseModel):
    time: Optional[float] = Field(None, description="Time spent to process this request")
    status: Literal[
        "ok",
    ] = Field(None, description="")
    result: Optional["CollectionInfo"] = Field(None, description="")


class InlineResponse2003(BaseModel):
    time: Optional[float] = Field(None, description="Time spent to process this request")
    status: Literal[
        "ok",
    ] = Field(None, description="")
    result: Optional["UpdateResult"] = Field(None, description="")


class InlineResponse2004(BaseModel):
    time: Optional[float] = Field(None, description="Time spent to process this request")
    status: Literal[
        "ok",
    ] = Field(None, description="")
    result: Optional["Record"] = Field(None, description="")


class InlineResponse2005(BaseModel):
    time: Optional[float] = Field(None, description="Time spent to process this request")
    status: Literal[
        "ok",
    ] = Field(None, description="")
    result: Optional[List["Record"]] = Field(None, description="")


class InlineResponse2006(BaseModel):
    time: Optional[float] = Field(None, description="Time spent to process this request")
    status: Literal[
        "ok",
    ] = Field(None, description="")
    result: Optional["ScrollResult"] = Field(None, description="")


class InlineResponse2007(BaseModel):
    time: Optional[float] = Field(None, description="Time spent to process this request")
    status: Literal[
        "ok",
    ] = Field(None, description="")
    result: Optional[List["ScoredPoint"]] = Field(None, description="")


class Match(BaseModel):
    integer: Optional[int] = Field(None, description="Integer value to match")
    keyword: Optional[str] = Field(None, description="Keyword value to match")


class OptimizersConfig(BaseModel):
    deleted_threshold: float = Field(
        ...,
        description="The minimal fraction of deleted vectors in a segment, required to perform segment optimization",
    )
    flush_interval_sec: int = Field(..., description="Minimum interval between forced flushes.")
    indexing_threshold: int = Field(
        ...,
        description="Maximum number of vectors allowed for plain index. Default value based on https://github.com/google-research/google-research/blob/master/scann/docs/algorithms.md",
    )
    max_segment_number: int = Field(
        ..., description="If the number of segments exceeds this value, the optimizer will merge the smallest segments."
    )
    memmap_threshold: int = Field(
        ...,
        description="Maximum number of vectors to store in-memory per segment. Segments larger than this threshold will be stored as read-only memmaped file.",
    )
    payload_indexing_threshold: int = Field(
        ...,
        description="Starting from this amount of vectors per-segment the engine will start building index for payload.",
    )
    vacuum_min_vector_number: int = Field(
        ..., description="The minimal number of vectors in a segment, required to perform segment optimization"
    )


class OptimizersConfigDiff(BaseModel):
    deleted_threshold: Optional[float] = Field(
        None,
        description="The minimal fraction of deleted vectors in a segment, required to perform segment optimization",
    )
    flush_interval_sec: Optional[int] = Field(None, description="Minimum interval between forced flushes.")
    indexing_threshold: Optional[int] = Field(
        None,
        description="Maximum number of vectors allowed for plain index. Default value based on https://github.com/google-research/google-research/blob/master/scann/docs/algorithms.md",
    )
    max_segment_number: Optional[int] = Field(
        None,
        description="If the number of segments exceeds this value, the optimizer will merge the smallest segments.",
    )
    memmap_threshold: Optional[int] = Field(
        None,
        description="Maximum number of vectors to store in-memory per segment. Segments larger than this threshold will be stored as read-only memmaped file.",
    )
    payload_indexing_threshold: Optional[int] = Field(
        None,
        description="Starting from this amount of vectors per-segment the engine will start building index for payload.",
    )
    vacuum_min_vector_number: Optional[int] = Field(
        None, description="The minimal number of vectors in a segment, required to perform segment optimization"
    )


class PayloadInterfaceStrictAnyOf(BaseModel):
    type: Literal[
        "keyword",
    ] = Field(..., description="")
    value: "PayloadVariantForString" = Field(..., description="")


class PayloadInterfaceStrictAnyOf1(BaseModel):
    type: Literal[
        "integer",
    ] = Field(..., description="")
    value: "PayloadVariantForInt64" = Field(..., description="")


class PayloadInterfaceStrictAnyOf2(BaseModel):
    type: Literal[
        "float",
    ] = Field(..., description="")
    value: "PayloadVariantForDouble" = Field(..., description="")


class PayloadInterfaceStrictAnyOf3(BaseModel):
    type: Literal[
        "geo",
    ] = Field(..., description="")
    value: "PayloadVariantForGeoPoint" = Field(..., description="")


class PayloadOpsAnyOf(BaseModel):
    """
    Set payload value, overrides if it is already exists
    """

    set_payload: "PayloadOpsAnyOfSetPayload" = Field(
        ..., description="Set payload value, overrides if it is already exists"
    )


class PayloadOpsAnyOf1(BaseModel):
    """
    Deletes specified payload values if they are assigned
    """

    delete_payload: "PayloadOpsAnyOf1DeletePayload" = Field(
        ..., description="Deletes specified payload values if they are assigned"
    )


class PayloadOpsAnyOf1DeletePayload(BaseModel):
    keys: List[str] = Field(..., description="")
    points: List[int] = Field(..., description="Deletes values from each point in this list")


class PayloadOpsAnyOf2(BaseModel):
    """
    Drops all Payload values associated with given points.
    """

    clear_payload: "PayloadOpsAnyOf2ClearPayload" = Field(
        ..., description="Drops all Payload values associated with given points."
    )


class PayloadOpsAnyOf2ClearPayload(BaseModel):
    points: List[int] = Field(..., description="")


class PayloadOpsAnyOfSetPayload(BaseModel):
    payload: Dict[str, "PayloadInterface"] = Field(..., description="")
    points: List[int] = Field(..., description="Assigns payload to each point in this list")


class PayloadSchemaInfo(BaseModel):
    data_type: "PayloadSchemaType" = Field(..., description="")
    indexed: bool = Field(..., description="")


class PayloadSchemaTypeAnyOf(BaseModel):
    type: Literal[
        "keyword",
    ] = Field(..., description="")


class PayloadSchemaTypeAnyOf1(BaseModel):
    type: Literal[
        "integer",
    ] = Field(..., description="")


class PayloadSchemaTypeAnyOf2(BaseModel):
    type: Literal[
        "float",
    ] = Field(..., description="")


class PayloadSchemaTypeAnyOf3(BaseModel):
    type: Literal[
        "geo",
    ] = Field(..., description="")


class PayloadTypeAnyOf(BaseModel):
    type: Literal[
        "keyword",
    ] = Field(..., description="")
    value: List[str] = Field(..., description="")


class PayloadTypeAnyOf1(BaseModel):
    type: Literal[
        "integer",
    ] = Field(..., description="")
    value: List[int] = Field(..., description="")


class PayloadTypeAnyOf2(BaseModel):
    type: Literal[
        "float",
    ] = Field(..., description="")
    value: List[float] = Field(..., description="")


class PayloadTypeAnyOf3(BaseModel):
    type: Literal[
        "geo",
    ] = Field(..., description="")
    value: List["GeoPoint"] = Field(..., description="")


class PointInsertOperationsAnyOf(BaseModel):
    """
    Inset points from a batch.
    """

    batch: "PointInsertOperationsAnyOfBatch" = Field(..., description="Inset points from a batch.")


class PointInsertOperationsAnyOf1(BaseModel):
    """
    Insert points from a list
    """

    points: List["PointStruct"] = Field(..., description="Insert points from a list")


class PointInsertOperationsAnyOfBatch(BaseModel):
    ids: List[int] = Field(..., description="")
    payloads: Optional[List[Dict[str, "PayloadInterface"]]] = Field(None, description="")
    vectors: List[List[float]] = Field(..., description="")


class PointOperationsAnyOf(BaseModel):
    """
    Insert or update points
    """

    upsert_points: "PointInsertOperations" = Field(..., description="Insert or update points")


class PointOperationsAnyOf1(BaseModel):
    """
    Delete point if exists
    """

    delete_points: "PointOperationsAnyOf1DeletePoints" = Field(..., description="Delete point if exists")


class PointOperationsAnyOf1DeletePoints(BaseModel):
    ids: List[int] = Field(..., description="")


class PointRequest(BaseModel):
    ids: List[int] = Field(..., description="")


class PointStruct(BaseModel):
    id: int = Field(..., description="Point id")
    payload: Optional[Dict[str, "PayloadInterface"]] = Field(None, description="Payload values (optional)")
    vector: List[float] = Field(..., description="Vector")


class Range(BaseModel):
    gt: Optional[float] = Field(None, description="point.key &gt; range.gt")
    gte: Optional[float] = Field(None, description="point.key &gt;= range.gte")
    lt: Optional[float] = Field(None, description="point.key &lt; range.lt")
    lte: Optional[float] = Field(None, description="point.key &lt;= range.lte")


class RecommendRequest(BaseModel):
    """
    Search request
    """

    filter: Optional["Filter"] = Field(None, description="Look only for points which satisfies this conditions")
    negative: List[int] = Field(..., description="Try to avoid vectors like this")
    params: Optional["SearchParams"] = Field(None, description="Additional search params")
    positive: List[int] = Field(..., description="Look for vectors closest to those")
    top: int = Field(..., description="Max number of result to return")


class Record(BaseModel):
    """
    Point data
    """

    id: int = Field(..., description="Id of the point")
    payload: Optional[Dict[str, "PayloadType"]] = Field(None, description="Payload - values assigned to the point")
    vector: Optional[List[float]] = Field(None, description="Vector of the point")


class ScoredPoint(BaseModel):
    id: int = Field(..., description="Point id")
    score: float = Field(..., description="Points vector distance to the query vector")


class ScrollRequest(BaseModel):
    """
    Scroll request - paginate over all points which matches given condition
    """

    filter: Optional["Filter"] = Field(
        None, description="Look only for points which satisfies this conditions. If not provided - all points."
    )
    limit: Optional[int] = Field(None, description="Page size. Default: 10")
    offset: Optional[int] = Field(None, description="Start ID to read points from. Default: 0")
    with_payload: Optional[bool] = Field(None, description="Return point payload with the result. Default: true")
    with_vector: Optional[bool] = Field(None, description="Return point vector with the result. Default: false")


class ScrollResult(BaseModel):
    """
    Result of the points read request. Contains
    """

    next_page_offset: Optional[int] = Field(
        None, description="Offset which should be used to retrieve a next page result"
    )
    points: List["Record"] = Field(..., description="List of retrieved points")


class SearchParams(BaseModel):
    """
    Additional parameters of the search
    """

    hnsw_ef: Optional[int] = Field(
        None,
        description="Params relevant to HNSW index /// Size of the beam in a beam-search. Larger the value - more accurate the result, more time required for search.",
    )


class SearchRequest(BaseModel):
    """
    Search request
    """

    filter: Optional["Filter"] = Field(None, description="Look only for points which satisfies this conditions")
    params: Optional["SearchParams"] = Field(None, description="Additional search params")
    top: int = Field(..., description="Max number of result to return")
    vector: List[float] = Field(..., description="Look for vectors closest to this")


class StorageOperationsAnyOf(BaseModel):
    """
    Create new collection and (optionally) specify index params
    """

    create_collection: "StorageOperationsAnyOfCreateCollection" = Field(
        ..., description="Create new collection and (optionally) specify index params"
    )


class StorageOperationsAnyOf1(BaseModel):
    """
    Update parameters of the existing collection
    """

    update_collection: "StorageOperationsAnyOf1UpdateCollection" = Field(
        ..., description="Update parameters of the existing collection"
    )


class StorageOperationsAnyOf1UpdateCollection(BaseModel):
    name: str = Field(..., description="")
    optimizers_config: Optional["OptimizersConfigDiff"] = Field(
        None,
        description="Custom params for Optimizers.  If none - values from service configuration file are used. This operation is blocking, it will only proceed ones all current optimizations are complete",
    )


class StorageOperationsAnyOf2(BaseModel):
    """
    Delete collection with given name
    """

    delete_collection: str = Field(..., description="Delete collection with given name")


class StorageOperationsAnyOf3(BaseModel):
    """
    Perform changes of collection aliases. Alias changes are atomic, meaning that no collection modifications can happen between alias operations.
    """

    change_aliases: "StorageOperationsAnyOf3ChangeAliases" = Field(
        ...,
        description="Perform changes of collection aliases. Alias changes are atomic, meaning that no collection modifications can happen between alias operations.",
    )


class StorageOperationsAnyOf3ChangeAliases(BaseModel):
    actions: List["AliasOperations"] = Field(..., description="")


class StorageOperationsAnyOfCreateCollection(BaseModel):
    distance: "Distance" = Field(..., description="")
    hnsw_config: Optional["HnswConfigDiff"] = Field(
        None, description="Custom params for HNSW index. If none - values from service configuration file are used."
    )
    name: str = Field(..., description="")
    optimizers_config: Optional["OptimizersConfigDiff"] = Field(
        None, description="Custom params for Optimizers.  If none - values from service configuration file are used."
    )
    vector_size: int = Field(..., description="")
    wal_config: Optional["WalConfigDiff"] = Field(
        None, description="Custom params for WAL. If none - values from service configuration file are used."
    )


class UpdateResult(BaseModel):
    operation_id: int = Field(..., description="Sequential number of the operation")
    status: "UpdateStatus" = Field(..., description="")


class UpdateStatus(str, Enum):
    ACKNOWLEDGED = "acknowledged"
    COMPLETED = "completed"


class WalConfig(BaseModel):
    wal_capacity_mb: int = Field(..., description="Size of a single WAL segment in MB")
    wal_segments_ahead: int = Field(..., description="Number of WAL segments to create ahead of actually used ones")


class WalConfigDiff(BaseModel):
    wal_capacity_mb: Optional[int] = Field(None, description="Size of a single WAL segment in MB")
    wal_segments_ahead: Optional[int] = Field(
        None, description="Number of WAL segments to create ahead of actually used ones"
    )


AliasOperations = Union[
    AliasOperationsAnyOf,
    AliasOperationsAnyOf1,
    AliasOperationsAnyOf2,
]
Condition = Union[
    FieldCondition,
    Filter,
    HasIdCondition,
]
FieldIndexOperations = Union[
    FieldIndexOperationsAnyOf,
    FieldIndexOperationsAnyOf1,
]
PayloadInterfaceStrict = Union[
    PayloadInterfaceStrictAnyOf,
    PayloadInterfaceStrictAnyOf1,
    PayloadInterfaceStrictAnyOf2,
    PayloadInterfaceStrictAnyOf3,
]
PayloadOps = Union[
    PayloadOpsAnyOf,
    PayloadOpsAnyOf1,
    PayloadOpsAnyOf2,
]
PayloadSchemaType = Union[
    PayloadSchemaTypeAnyOf,
    PayloadSchemaTypeAnyOf1,
    PayloadSchemaTypeAnyOf2,
    PayloadSchemaTypeAnyOf3,
]
PayloadType = Union[
    PayloadTypeAnyOf,
    PayloadTypeAnyOf1,
    PayloadTypeAnyOf2,
    PayloadTypeAnyOf3,
]
PayloadVariantForDouble = Union[
    List[float],
    float,
]
PayloadVariantForGeoPoint = Union[
    GeoPoint,
    List[GeoPoint],
]
PayloadVariantForInt64 = Union[
    List[int],
    int,
]
PayloadVariantForString = Union[
    List[str],
    str,
]
PointInsertOperations = Union[
    PointInsertOperationsAnyOf,
    PointInsertOperationsAnyOf1,
]
PointOperations = Union[
    PointOperationsAnyOf,
    PointOperationsAnyOf1,
]
StorageOperations = Union[
    StorageOperationsAnyOf,
    StorageOperationsAnyOf1,
    StorageOperationsAnyOf2,
    StorageOperationsAnyOf3,
]
CollectionUpdateOperations = Union[
    FieldIndexOperations,
    PayloadOps,
    PointOperations,
]
PayloadInterface = Union[
    PayloadInterfaceStrict,
    PayloadVariantForDouble,
    PayloadVariantForInt64,
    PayloadVariantForString,
]
