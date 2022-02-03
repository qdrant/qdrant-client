from enum import Enum
from typing import Any, Dict, List, Optional, Union

try:
    from typing import Literal
except ImportError:
    # Python 3.7 backport
    from typing_extensions import Literal

from pydantic import BaseModel, Field
from pydantic.types import StrictFloat, StrictInt, StrictStr


class Batch(BaseModel):
    ids: List["ExtendedPointId"] = Field(..., description="")
    payloads: Optional[List[Dict[str, "PayloadInterface"]]] = Field(None, description="")
    vectors: List[List[float]] = Field(..., description="")


class ChangeAliasesOperation(BaseModel):
    """
    Operation for performing changes of collection aliases. Alias changes are atomic, meaning that no collection modifications can happen between alias operations.
    """

    actions: List["AliasOperations"] = Field(
        ...,
        description="Operation for performing changes of collection aliases. Alias changes are atomic, meaning that no collection modifications can happen between alias operations.",
    )


class CollectionConfig(BaseModel):
    hnsw_config: "HnswConfig" = Field(..., description="")
    optimizer_config: "OptimizersConfig" = Field(..., description="")
    params: "CollectionParams" = Field(..., description="")
    wal_config: "WalConfig" = Field(..., description="")


class CollectionDescription(BaseModel):
    name: str = Field(..., description="")


class CollectionInfo(BaseModel):
    """
    Current statistics and configuration of the collection
    """

    config: "CollectionConfig" = Field(..., description="Current statistics and configuration of the collection")
    disk_data_size: int = Field(..., description="Disk space, used by collection")
    payload_schema: Dict[str, "PayloadSchemaInfo"] = Field(..., description="Types of stored payload")
    ram_data_size: int = Field(..., description="RAM used by collection")
    segments_count: int = Field(..., description="Number of segments in collection")
    status: "CollectionStatus" = Field(..., description="Current statistics and configuration of the collection")
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


class CreateAlias(BaseModel):
    """
    Create alternative name for a collection. Collection will be available under both names for search, retrieve,
    """

    alias_name: str = Field(
        ...,
        description="Create alternative name for a collection. Collection will be available under both names for search, retrieve,",
    )
    collection_name: str = Field(
        ...,
        description="Create alternative name for a collection. Collection will be available under both names for search, retrieve,",
    )


class CreateAliasOperation(BaseModel):
    create_alias: "CreateAlias" = Field(..., description="")


class CreateCollection(BaseModel):
    """
    Operation for creating new collection and (optionally) specify index params
    """

    distance: "Distance" = Field(
        ..., description="Operation for creating new collection and (optionally) specify index params"
    )
    hnsw_config: Optional["HnswConfigDiff"] = Field(
        None, description="Custom params for HNSW index. If none - values from service configuration file are used."
    )
    optimizers_config: Optional["OptimizersConfigDiff"] = Field(
        None, description="Custom params for Optimizers.  If none - values from service configuration file are used."
    )
    vector_size: int = Field(
        ..., description="Operation for creating new collection and (optionally) specify index params"
    )
    wal_config: Optional["WalConfigDiff"] = Field(
        None, description="Custom params for WAL. If none - values from service configuration file are used."
    )


class CreateCollectionOperation(BaseModel):
    """
    Operation for creating new collection and (optionally) specify index params
    """

    distance: "Distance" = Field(
        ..., description="Operation for creating new collection and (optionally) specify index params"
    )
    hnsw_config: Optional["HnswConfigDiff"] = Field(
        None, description="Custom params for HNSW index. If none - values from service configuration file are used."
    )
    name: str = Field(..., description="Operation for creating new collection and (optionally) specify index params")
    optimizers_config: Optional["OptimizersConfigDiff"] = Field(
        None, description="Custom params for Optimizers.  If none - values from service configuration file are used."
    )
    vector_size: int = Field(
        ..., description="Operation for creating new collection and (optionally) specify index params"
    )
    wal_config: Optional["WalConfigDiff"] = Field(
        None, description="Custom params for WAL. If none - values from service configuration file are used."
    )


class CreateFieldIndex(BaseModel):
    field_name: str = Field(..., description="")


class DeleteAlias(BaseModel):
    """
    Delete alias if exists
    """

    alias_name: str = Field(..., description="Delete alias if exists")


class DeleteAliasOperation(BaseModel):
    """
    Delete alias if exists
    """

    delete_alias: "DeleteAlias" = Field(..., description="Delete alias if exists")


class DeletePayload(BaseModel):
    keys: List[str] = Field(..., description="")
    points: List["ExtendedPointId"] = Field(..., description="Deletes values from each point in this list")


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
    """
    All possible payload filtering conditions
    """

    geo_bounding_box: Optional["GeoBoundingBox"] = Field(
        None, description="Check if points geo location lies in a given area"
    )
    geo_radius: Optional["GeoRadius"] = Field(None, description="Check if geo point is within a given radius")
    key: str = Field(..., description="All possible payload filtering conditions")
    match: Optional["Match"] = Field(None, description="Check if point has field with a given value")
    range: Optional["Range"] = Field(None, description="Check if points value lies in a given range")


class FieldIndexOperationsOneOf(BaseModel):
    """
    Create index for payload field
    """

    create_index: str = Field(..., description="Create index for payload field")


class FieldIndexOperationsOneOf1(BaseModel):
    """
    Delete index for the field
    """

    delete_index: str = Field(..., description="Delete index for the field")


class Filter(BaseModel):
    must: Optional[List["Condition"]] = Field(None, description="All conditions must match")
    must_not: Optional[List["Condition"]] = Field(None, description="All conditions must NOT match")
    should: Optional[List["Condition"]] = Field(None, description="At least one of thous conditions should match")


class FilterSelector(BaseModel):
    filter: "Filter" = Field(..., description="")


class GeoBoundingBox(BaseModel):
    """
    Geo filter request  Matches coordinates inside the rectangle, described by coordinates of lop-left and bottom-right edges
    """

    bottom_right: "GeoPoint" = Field(
        ...,
        description="Geo filter request  Matches coordinates inside the rectangle, described by coordinates of lop-left and bottom-right edges",
    )
    top_left: "GeoPoint" = Field(
        ...,
        description="Geo filter request  Matches coordinates inside the rectangle, described by coordinates of lop-left and bottom-right edges",
    )


class GeoPoint(BaseModel):
    """
    Geo point payload schema
    """

    lat: float = Field(..., description="Geo point payload schema")
    lon: float = Field(..., description="Geo point payload schema")


class GeoRadius(BaseModel):
    """
    Geo filter request  Matches coordinates inside the circle of `radius` and center with coordinates `center`
    """

    center: "GeoPoint" = Field(
        ...,
        description="Geo filter request  Matches coordinates inside the circle of `radius` and center with coordinates `center`",
    )
    radius: float = Field(..., description="Radius of the area in meters")


class HasIdCondition(BaseModel):
    """
    ID-based filtering condition
    """

    has_id: List["ExtendedPointId"] = Field(..., description="ID-based filtering condition")


class HnswConfig(BaseModel):
    """
    Config of HNSW index
    """

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
    """
    Match filter request
    """

    integer: Optional[int] = Field(None, description="Integer value to match")
    keyword: Optional[str] = Field(None, description="Keyword value to match")


class OptimizersConfig(BaseModel):
    default_segment_number: int = Field(
        ...,
        description="Target amount of segments optimizer will try to keep. Real amount of segments may vary depending on multiple parameters: - Amount of stored points - Current write RPS  It is recommended to select default number of segments as a factor of the number of search threads, so that each segment would be handled evenly by one of the threads",
    )
    deleted_threshold: float = Field(
        ...,
        description="The minimal fraction of deleted vectors in a segment, required to perform segment optimization",
    )
    flush_interval_sec: int = Field(..., description="Minimum interval between forced flushes.")
    indexing_threshold: int = Field(
        ...,
        description="Maximum number of vectors allowed for plain index. Default value based on https://github.com/google-research/google-research/blob/master/scann/docs/algorithms.md",
    )
    max_optimization_threads: int = Field(..., description="Maximum available threads for optimization workers")
    max_segment_size: int = Field(
        ...,
        description="Do not create segments larger this number of points. Large segments might require disproportionately long indexation times, therefore it makes sense to limit the size of segments.  If indexation speed have more priority for your - make this parameter lower. If search speed is more important - make this parameter higher.",
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
    default_segment_number: Optional[int] = Field(
        None,
        description="Target amount of segments optimizer will try to keep. Real amount of segments may vary depending on multiple parameters: - Amount of stored points - Current write RPS  It is recommended to select default number of segments as a factor of the number of search threads, so that each segment would be handled evenly by one of the threads",
    )
    deleted_threshold: Optional[float] = Field(
        None,
        description="The minimal fraction of deleted vectors in a segment, required to perform segment optimization",
    )
    flush_interval_sec: Optional[int] = Field(None, description="Minimum interval between forced flushes.")
    indexing_threshold: Optional[int] = Field(
        None,
        description="Maximum number of vectors allowed for plain index. Default value based on https://github.com/google-research/google-research/blob/master/scann/docs/algorithms.md",
    )
    max_optimization_threads: Optional[int] = Field(
        None, description="Maximum available threads for optimization workers"
    )
    max_segment_size: Optional[int] = Field(
        None,
        description="Do not create segments larger this number of points. Large segments might require disproportionately long indexation times, therefore it makes sense to limit the size of segments.  If indexation speed have more priority for your - make this parameter lower. If search speed is more important - make this parameter higher.",
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


class PayloadInterfaceStrictOneOf(BaseModel):
    type: Literal[
        "keyword",
    ] = Field(..., description="")
    value: "PayloadVariantForString" = Field(..., description="")


class PayloadInterfaceStrictOneOf1(BaseModel):
    type: Literal[
        "integer",
    ] = Field(..., description="")
    value: "PayloadVariantForInt64" = Field(..., description="")


class PayloadInterfaceStrictOneOf2(BaseModel):
    type: Literal[
        "float",
    ] = Field(..., description="")
    value: "PayloadVariantForDouble" = Field(..., description="")


class PayloadInterfaceStrictOneOf3(BaseModel):
    type: Literal[
        "geo",
    ] = Field(..., description="")
    value: "PayloadVariantForGeoPoint" = Field(..., description="")


class PayloadOpsOneOf(BaseModel):
    """
    Set payload value, overrides if it is already exists
    """

    set_payload: "SetPayload" = Field(..., description="Set payload value, overrides if it is already exists")


class PayloadOpsOneOf1(BaseModel):
    """
    Deletes specified payload values if they are assigned
    """

    delete_payload: "DeletePayload" = Field(..., description="Deletes specified payload values if they are assigned")


class PayloadOpsOneOf2(BaseModel):
    """
    Drops all Payload values associated with given points.
    """

    clear_payload: "PayloadOpsOneOf2ClearPayload" = Field(
        ..., description="Drops all Payload values associated with given points."
    )


class PayloadOpsOneOf2ClearPayload(BaseModel):
    points: List["ExtendedPointId"] = Field(..., description="")


class PayloadSchemaInfo(BaseModel):
    """
    Payload field type &amp; index information
    """

    data_type: "PayloadSchemaType" = Field(..., description="Payload field type &amp; index information")
    indexed: bool = Field(..., description="Payload field type &amp; index information")


class PayloadSchemaTypeOneOf(BaseModel):
    type: Literal[
        "keyword",
    ] = Field(..., description="")


class PayloadSchemaTypeOneOf1(BaseModel):
    type: Literal[
        "integer",
    ] = Field(..., description="")


class PayloadSchemaTypeOneOf2(BaseModel):
    type: Literal[
        "float",
    ] = Field(..., description="")


class PayloadSchemaTypeOneOf3(BaseModel):
    type: Literal[
        "geo",
    ] = Field(..., description="")


class PayloadSelector(BaseModel):
    """
    Specifies how to treat payload selector
    """

    exclude: List[str] = Field(..., description="Post-exclude return payload key type")
    include: List[str] = Field(..., description="Include return payload key type")


class PayloadTypeOneOf(BaseModel):
    type: Literal[
        "keyword",
    ] = Field(..., description="")
    value: List[str] = Field(..., description="")


class PayloadTypeOneOf1(BaseModel):
    type: Literal[
        "integer",
    ] = Field(..., description="")
    value: List[int] = Field(..., description="")


class PayloadTypeOneOf2(BaseModel):
    type: Literal[
        "float",
    ] = Field(..., description="")
    value: List[float] = Field(..., description="")


class PayloadTypeOneOf3(BaseModel):
    type: Literal[
        "geo",
    ] = Field(..., description="")
    value: List["GeoPoint"] = Field(..., description="")


class PointIdsList(BaseModel):
    points: List["ExtendedPointId"] = Field(..., description="")


class PointOperationsOneOf(BaseModel):
    """
    Insert or update points
    """

    upsert_points: "PointInsertOperations" = Field(..., description="Insert or update points")


class PointOperationsOneOf1(BaseModel):
    """
    Delete point if exists
    """

    delete_points: "PointOperationsOneOf1DeletePoints" = Field(..., description="Delete point if exists")


class PointOperationsOneOf1DeletePoints(BaseModel):
    ids: List["ExtendedPointId"] = Field(..., description="")


class PointOperationsOneOf2(BaseModel):
    """
    Delete points by given filter criteria
    """

    delete_points_by_filter: "Filter" = Field(..., description="Delete points by given filter criteria")


class PointRequest(BaseModel):
    ids: List["ExtendedPointId"] = Field(..., description="")
    with_payload: Optional["WithPayloadInterface"] = Field(None, description="")
    with_vector: Optional[bool] = Field(None, description="")


class PointStruct(BaseModel):
    id: "ExtendedPointId" = Field(..., description="")
    payload: Optional[Dict[str, "PayloadInterface"]] = Field(None, description="Payload values (optional)")
    vector: List[float] = Field(..., description="Vector")


class PointsBatch(BaseModel):
    batch: "Batch" = Field(..., description="")


class PointsList(BaseModel):
    points: List["PointStruct"] = Field(..., description="")


class Range(BaseModel):
    """
    Range filter request
    """

    gt: Optional[float] = Field(None, description="point.key &gt; range.gt")
    gte: Optional[float] = Field(None, description="point.key &gt;= range.gte")
    lt: Optional[float] = Field(None, description="point.key &lt; range.lt")
    lte: Optional[float] = Field(None, description="point.key &lt;= range.lte")


class RecommendRequest(BaseModel):
    """
    Recommendation request. Provides positive and negative examples of the vectors, which are already stored in the collection.  Service should look for the points which are closer to positive examples and at the same time further to negative examples. The concrete way of how to compare negative and positive distances is up to implementation in `segment` crate.
    """

    filter: Optional["Filter"] = Field(None, description="Look only for points which satisfies this conditions")
    negative: List["ExtendedPointId"] = Field(..., description="Try to avoid vectors like this")
    params: Optional["SearchParams"] = Field(None, description="Additional search params")
    positive: List["ExtendedPointId"] = Field(..., description="Look for vectors closest to those")
    top: int = Field(..., description="Max number of result to return")


class Record(BaseModel):
    """
    Point data
    """

    id: "ExtendedPointId" = Field(..., description="Point data")
    payload: Optional[Dict[str, "PayloadType"]] = Field(None, description="Payload - values assigned to the point")
    vector: Optional[List[float]] = Field(None, description="Vector of the point")


class RenameAlias(BaseModel):
    """
    Change alias to a new one
    """

    new_alias_name: str = Field(..., description="Change alias to a new one")
    old_alias_name: str = Field(..., description="Change alias to a new one")


class RenameAliasOperation(BaseModel):
    """
    Change alias to a new one
    """

    rename_alias: "RenameAlias" = Field(..., description="Change alias to a new one")


class ScoredPoint(BaseModel):
    """
    Search result
    """

    id: "ExtendedPointId" = Field(..., description="Search result")
    payload: Optional[Dict[str, "PayloadType"]] = Field(None, description="Payload - values assigned to the point")
    score: float = Field(..., description="Points vector distance to the query vector")
    vector: Optional[List[float]] = Field(None, description="Vector of the point")
    version: int = Field(..., description="Point version")


class ScrollRequest(BaseModel):
    """
    Scroll request - paginate over all points which matches given condition
    """

    filter: Optional["Filter"] = Field(
        None, description="Look only for points which satisfies this conditions. If not provided - all points."
    )
    limit: Optional[int] = Field(None, description="Page size. Default: 10")
    offset: Optional["ExtendedPointId"] = Field(None, description="Start ID to read points from. Default: 0")
    with_payload: Optional["WithPayloadInterface"] = Field(
        None, description="Return point payload with the result. Default: True"
    )
    with_vector: Optional[bool] = Field(None, description="Return point vector with the result. Default: false")


class ScrollResult(BaseModel):
    """
    Result of the points read request
    """

    next_page_offset: Optional["ExtendedPointId"] = Field(
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
    Search request. Holds all conditions and parameters for the search of most similar points by vector similarity given the filtering restrictions.
    """

    filter: Optional["Filter"] = Field(None, description="Look only for points which satisfies this conditions")
    params: Optional["SearchParams"] = Field(None, description="Additional search params")
    top: int = Field(..., description="Max number of result to return")
    vector: List[float] = Field(..., description="Look for vectors closest to this")
    with_payload: Optional["WithPayloadInterface"] = Field(None, description="Payload interface")
    with_vector: Optional[bool] = Field(None, description="Return point vector with the result. Default: false")


class SetPayload(BaseModel):
    payload: Dict[str, "PayloadInterface"] = Field(..., description="")
    points: List["ExtendedPointId"] = Field(..., description="Assigns payload to each point in this list")


class StorageOperationsOneOf(BaseModel):
    create_collection: "CreateCollectionOperation" = Field(..., description="")


class StorageOperationsOneOf1(BaseModel):
    update_collection: "UpdateCollectionOperation" = Field(..., description="")


class StorageOperationsOneOf2(BaseModel):
    delete_collection: str = Field(..., description="Operation for deleting collection with given name")


class StorageOperationsOneOf3(BaseModel):
    change_aliases: "ChangeAliasesOperation" = Field(..., description="")


class UpdateCollection(BaseModel):
    """
    Operation for updating parameters of the existing collection
    """

    optimizers_config: Optional["OptimizersConfigDiff"] = Field(
        None,
        description="Custom params for Optimizers.  If none - values from service configuration file are used. This operation is blocking, it will only proceed ones all current optimizations are complete",
    )


class UpdateCollectionOperation(BaseModel):
    """
    Operation for updating parameters of the existing collection
    """

    name: str = Field(..., description="Operation for updating parameters of the existing collection")
    optimizers_config: Optional["OptimizersConfigDiff"] = Field(
        None,
        description="Custom params for Optimizers.  If none - values from service configuration file are used. This operation is blocking, it will only proceed ones all current optimizations are complete",
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
    CreateAliasOperation,
    DeleteAliasOperation,
    RenameAliasOperation,
]
Condition = Union[
    FieldCondition,
    HasIdCondition,
    Filter,
]
ExtendedPointId = Union[
    StrictInt,
    StrictStr,
]
FieldIndexOperations = Union[
    FieldIndexOperationsOneOf,
    FieldIndexOperationsOneOf1,
]
PayloadInterfaceStrict = Union[
    PayloadInterfaceStrictOneOf,
    PayloadInterfaceStrictOneOf1,
    PayloadInterfaceStrictOneOf2,
    PayloadInterfaceStrictOneOf3,
]
PayloadOps = Union[
    PayloadOpsOneOf,
    PayloadOpsOneOf1,
    PayloadOpsOneOf2,
]
PayloadSchemaType = Union[
    PayloadSchemaTypeOneOf,
    PayloadSchemaTypeOneOf1,
    PayloadSchemaTypeOneOf2,
    PayloadSchemaTypeOneOf3,
]
PayloadType = Union[
    PayloadTypeOneOf,
    PayloadTypeOneOf1,
    PayloadTypeOneOf2,
    PayloadTypeOneOf3,
]
PayloadVariantForDouble = Union[
    List[StrictFloat],
    StrictFloat,
]
PayloadVariantForGeoPoint = Union[
    GeoPoint,
    List[GeoPoint],
]
PayloadVariantForInt64 = Union[
    List[StrictInt],
    StrictInt,
]
PayloadVariantForString = Union[
    List[StrictStr],
    StrictStr,
]
PointInsertOperations = Union[
    PointsBatch,
    PointsList,
]
PointOperations = Union[
    PointOperationsOneOf,
    PointOperationsOneOf1,
    PointOperationsOneOf2,
]
PointsSelector = Union[
    PointIdsList,
    FilterSelector,
]
StorageOperations = Union[
    StorageOperationsOneOf,
    StorageOperationsOneOf1,
    StorageOperationsOneOf2,
    StorageOperationsOneOf3,
]
WithPayloadInterface = Union[
    PayloadSelector,
    List[StrictStr],
    bool,
]
CollectionUpdateOperations = Union[
    PointOperations,
    PayloadOps,
    FieldIndexOperations,
]
PayloadInterface = Union[
    PayloadVariantForString,
    PayloadVariantForInt64,
    PayloadVariantForDouble,
    PayloadInterfaceStrict,
]
