from enum import Enum
from typing import Any, Dict, List, Optional, Union

try:
    from typing import Literal
except ImportError:
    # Python 3.7 backport
    from typing_extensions import Literal

from pydantic import BaseModel, Field
from pydantic.types import StrictBool, StrictInt, StrictStr


class Batch(BaseModel):
    ids: List["ExtendedPointId"] = Field(..., description="")
    vectors: List[List[float]] = Field(..., description="")
    payloads: Optional[List["Payload"]] = Field(None, description="")


class ChangeAliasesOperation(BaseModel):
    """
    Operation for performing changes of collection aliases. Alias changes are atomic, meaning that no collection modifications can happen between alias operations.
    """

    actions: List["AliasOperations"] = Field(
        ...,
        description="Operation for performing changes of collection aliases. Alias changes are atomic, meaning that no collection modifications can happen between alias operations.",
    )


class ClusterStatusOneOf(BaseModel):
    status: Literal[
        "disabled",
    ] = Field(..., description="")


class ClusterStatusOneOf1(BaseModel):
    """
    Description of enabled cluster
    """

    status: Literal[
        "enabled",
    ] = Field(..., description="Description of enabled cluster")
    peer_id: int = Field(..., description="ID of this peer")
    peers: Dict[str, "PeerInfo"] = Field(..., description="Peers composition of the cluster with main information")
    raft_info: "RaftInfo" = Field(..., description="Description of enabled cluster")


class CollectionConfig(BaseModel):
    params: "CollectionParams" = Field(..., description="")
    hnsw_config: "HnswConfig" = Field(..., description="")
    optimizer_config: "OptimizersConfig" = Field(..., description="")
    wal_config: "WalConfig" = Field(..., description="")


class CollectionDescription(BaseModel):
    name: str = Field(..., description="")


class CollectionInfo(BaseModel):
    """
    Current statistics and configuration of the collection
    """

    status: "CollectionStatus" = Field(..., description="Current statistics and configuration of the collection")
    optimizer_status: "OptimizersStatus" = Field(
        ..., description="Current statistics and configuration of the collection"
    )
    vectors_count: int = Field(..., description="Number of vectors in collection")
    segments_count: int = Field(..., description="Number of segments in collection")
    disk_data_size: int = Field(..., description="Disk space, used by collection")
    ram_data_size: int = Field(..., description="RAM used by collection")
    config: "CollectionConfig" = Field(..., description="Current statistics and configuration of the collection")
    payload_schema: Dict[str, "PayloadIndexInfo"] = Field(..., description="Types of stored payload")


class CollectionParams(BaseModel):
    vector_size: int = Field(..., description="Size of a vectors used")
    distance: "Distance" = Field(..., description="")
    shard_number: Optional[int] = Field(1, description="Number of shards the collection has")
    on_disk_payload: Optional[bool] = Field(
        False,
        description="If true - point&#x27;s payload will not be stored in memory. It will be read from the disk every time it is requested. This setting saves RAM by (slightly) increasing the response time. Note: those payload values that are involved in filtering and are indexed - remain in RAM.",
    )


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

    collection_name: str = Field(
        ...,
        description="Create alternative name for a collection. Collection will be available under both names for search, retrieve,",
    )
    alias_name: str = Field(
        ...,
        description="Create alternative name for a collection. Collection will be available under both names for search, retrieve,",
    )


class CreateAliasOperation(BaseModel):
    create_alias: "CreateAlias" = Field(..., description="")


class CreateCollection(BaseModel):
    """
    Operation for creating new collection and (optionally) specify index params
    """

    vector_size: int = Field(
        ..., description="Operation for creating new collection and (optionally) specify index params"
    )
    distance: "Distance" = Field(
        ..., description="Operation for creating new collection and (optionally) specify index params"
    )
    shard_number: Optional[int] = Field(
        None,
        description="Number of shards in collection. Default is 1 for standalone, otherwise equal to the number of nodes Minimum is 1",
    )
    on_disk_payload: Optional[bool] = Field(
        None,
        description="If true - point&#x27;s payload will not be stored in memory. It will be read from the disk every time it is requested. This setting saves RAM by (slightly) increasing the response time. Note: those payload values that are involved in filtering and are indexed - remain in RAM.",
    )
    hnsw_config: Optional["HnswConfigDiff"] = Field(
        None, description="Custom params for HNSW index. If none - values from service configuration file are used."
    )
    wal_config: Optional["WalConfigDiff"] = Field(
        None, description="Custom params for WAL. If none - values from service configuration file are used."
    )
    optimizers_config: Optional["OptimizersConfigDiff"] = Field(
        None, description="Custom params for Optimizers.  If none - values from service configuration file are used."
    )


class CreateFieldIndex(BaseModel):
    field_name: str = Field(..., description="")
    field_type: Optional["PayloadSchemaType"] = Field(None, description="")


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

    key: str = Field(..., description="Payload key")
    match: Optional["Match"] = Field(None, description="Check if point has field with a given value")
    range: Optional["Range"] = Field(None, description="Check if points value lies in a given range")
    geo_bounding_box: Optional["GeoBoundingBox"] = Field(
        None, description="Check if points geo location lies in a given area"
    )
    geo_radius: Optional["GeoRadius"] = Field(None, description="Check if geo point is within a given radius")
    values_count: Optional["ValuesCount"] = Field(None, description="Check number of values of the field")


class Filter(BaseModel):
    should: Optional[List["Condition"]] = Field(None, description="At least one of those conditions should match")
    must: Optional[List["Condition"]] = Field(None, description="All conditions must match")
    must_not: Optional[List["Condition"]] = Field(None, description="All conditions must NOT match")


class FilterSelector(BaseModel):
    filter: "Filter" = Field(..., description="")


class GeoBoundingBox(BaseModel):
    """
    Geo filter request  Matches coordinates inside the rectangle, described by coordinates of lop-left and bottom-right edges
    """

    top_left: "GeoPoint" = Field(
        ...,
        description="Geo filter request  Matches coordinates inside the rectangle, described by coordinates of lop-left and bottom-right edges",
    )
    bottom_right: "GeoPoint" = Field(
        ...,
        description="Geo filter request  Matches coordinates inside the rectangle, described by coordinates of lop-left and bottom-right edges",
    )


class GeoPoint(BaseModel):
    """
    Geo point payload schema
    """

    lon: float = Field(..., description="Geo point payload schema")
    lat: float = Field(..., description="Geo point payload schema")


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

    m: int = Field(
        ...,
        description="Number of edges per node in the index graph. Larger the value - more accurate the search, more space required.",
    )
    ef_construct: int = Field(
        ...,
        description="Number of neighbours to consider during the index building. Larger the value - more accurate the search, more time required to build index.",
    )
    full_scan_threshold: int = Field(
        ...,
        description="Minimal size (in KiloBytes) of vectors for additional payload-based indexing. If payload chunk is smaller than `full_scan_threshold_kb` additional indexing won&#x27;t be used - in this case full-scan search should be preferred by query planner and additional indexing is not required. Note: 1Kb = 1 vector of size 256",
    )


class HnswConfigDiff(BaseModel):
    m: Optional[int] = Field(
        None,
        description="Number of edges per node in the index graph. Larger the value - more accurate the search, more space required.",
    )
    ef_construct: Optional[int] = Field(
        None,
        description="Number of neighbours to consider during the index building. Larger the value - more accurate the search, more time required to build index.",
    )
    full_scan_threshold: Optional[int] = Field(
        None,
        description="Minimal size (in KiloBytes) of vectors for additional payload-based indexing. If payload chunk is smaller than `full_scan_threshold_kb` additional indexing won&#x27;t be used - in this case full-scan search should be preferred by query planner and additional indexing is not required. Note: 1Kb = 1 vector of size 256",
    )


class InlineResponse200(BaseModel):
    time: Optional[float] = Field(None, description="Time spent to process this request")
    status: Literal[
        "ok",
    ] = Field(None, description="")
    result: Optional["ClusterStatus"] = Field(None, description="")


class InlineResponse2001(BaseModel):
    time: Optional[float] = Field(None, description="Time spent to process this request")
    status: Literal[
        "ok",
    ] = Field(None, description="")
    result: Optional["CollectionsResponse"] = Field(None, description="")


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
    result: Optional[bool] = Field(None, description="")


class InlineResponse2004(BaseModel):
    time: Optional[float] = Field(None, description="Time spent to process this request")
    status: Literal[
        "ok",
    ] = Field(None, description="")
    result: Optional["UpdateResult"] = Field(None, description="")


class InlineResponse2005(BaseModel):
    time: Optional[float] = Field(None, description="Time spent to process this request")
    status: Literal[
        "ok",
    ] = Field(None, description="")
    result: Optional["Record"] = Field(None, description="")


class InlineResponse2006(BaseModel):
    time: Optional[float] = Field(None, description="Time spent to process this request")
    status: Literal[
        "ok",
    ] = Field(None, description="")
    result: Optional[List["Record"]] = Field(None, description="")


class InlineResponse2007(BaseModel):
    time: Optional[float] = Field(None, description="Time spent to process this request")
    status: Literal[
        "ok",
    ] = Field(None, description="")
    result: Optional["ScrollResult"] = Field(None, description="")


class InlineResponse2008(BaseModel):
    time: Optional[float] = Field(None, description="Time spent to process this request")
    status: Literal[
        "ok",
    ] = Field(None, description="")
    result: Optional[List["ScoredPoint"]] = Field(None, description="")


class IsEmptyCondition(BaseModel):
    """
    Select points with empty payload for a specified field
    """

    is_empty: "PayloadField" = Field(..., description="Select points with empty payload for a specified field")


class MatchInteger(BaseModel):
    """
    Match filter request (deprecated)
    """

    integer: int = Field(..., description="Integer value to match")


class MatchKeyword(BaseModel):
    """
    Match by keyword (deprecated)
    """

    keyword: str = Field(..., description="Keyword value to match")


class MatchValue(BaseModel):
    value: "ValueVariants" = Field(..., description="")


class OptimizersConfig(BaseModel):
    deleted_threshold: float = Field(
        ...,
        description="The minimal fraction of deleted vectors in a segment, required to perform segment optimization",
    )
    vacuum_min_vector_number: int = Field(
        ..., description="The minimal number of vectors in a segment, required to perform segment optimization"
    )
    default_segment_number: int = Field(
        ...,
        description="Target amount of segments optimizer will try to keep. Real amount of segments may vary depending on multiple parameters: - Amount of stored points - Current write RPS  It is recommended to select default number of segments as a factor of the number of search threads, so that each segment would be handled evenly by one of the threads If `default_segment_number = 0`, will be automatically selected by the number of available CPUs",
    )
    max_segment_size: int = Field(
        ...,
        description="Do not create segments larger this size (in KiloBytes). Large segments might require disproportionately long indexation times, therefore it makes sense to limit the size of segments.  If indexation speed have more priority for your - make this parameter lower. If search speed is more important - make this parameter higher. Note: 1Kb = 1 vector of size 256",
    )
    memmap_threshold: int = Field(
        ...,
        description="Maximum size (in KiloBytes) of vectors to store in-memory per segment. Segments larger than this threshold will be stored as read-only memmaped file. To enable memmap storage, lower the threshold Note: 1Kb = 1 vector of size 256",
    )
    indexing_threshold: int = Field(
        ...,
        description="Maximum size (in KiloBytes) of vectors allowed for plain index. Default value based on https://github.com/google-research/google-research/blob/master/scann/docs/algorithms.md Note: 1Kb = 1 vector of size 256",
    )
    flush_interval_sec: int = Field(..., description="Minimum interval between forced flushes.")
    max_optimization_threads: int = Field(..., description="Maximum available threads for optimization workers")


class OptimizersConfigDiff(BaseModel):
    deleted_threshold: Optional[float] = Field(
        None,
        description="The minimal fraction of deleted vectors in a segment, required to perform segment optimization",
    )
    vacuum_min_vector_number: Optional[int] = Field(
        None, description="The minimal number of vectors in a segment, required to perform segment optimization"
    )
    default_segment_number: Optional[int] = Field(
        None,
        description="Target amount of segments optimizer will try to keep. Real amount of segments may vary depending on multiple parameters: - Amount of stored points - Current write RPS  It is recommended to select default number of segments as a factor of the number of search threads, so that each segment would be handled evenly by one of the threads If `default_segment_number = 0`, will be automatically selected by the number of available CPUs",
    )
    max_segment_size: Optional[int] = Field(
        None,
        description="Do not create segments larger this size (in KiloBytes). Large segments might require disproportionately long indexation times, therefore it makes sense to limit the size of segments.  If indexation speed have more priority for your - make this parameter lower. If search speed is more important - make this parameter higher. Note: 1Kb = 1 vector of size 256",
    )
    memmap_threshold: Optional[int] = Field(
        None,
        description="Maximum size (in KiloBytes) of vectors to store in-memory per segment. Segments larger than this threshold will be stored as read-only memmaped file. To enable memmap storage, lower the threshold Note: 1Kb = 1 vector of size 256",
    )
    indexing_threshold: Optional[int] = Field(
        None,
        description="Maximum size (in KiloBytes) of vectors allowed for plain index. Default value based on https://github.com/google-research/google-research/blob/master/scann/docs/algorithms.md Note: 1Kb = 1 vector of size 256",
    )
    flush_interval_sec: Optional[int] = Field(None, description="Minimum interval between forced flushes.")
    max_optimization_threads: Optional[int] = Field(
        None, description="Maximum available threads for optimization workers"
    )


class OptimizersStatusOneOf(str, Enum):
    OK = "ok"


class OptimizersStatusOneOf1(BaseModel):
    """
    Something wrong happened with optimizers
    """

    error: str = Field(..., description="Something wrong happened with optimizers")


Payload = dict


class PayloadField(BaseModel):
    """
    Payload field
    """

    key: str = Field(..., description="Payload field name")


class PayloadIndexInfo(BaseModel):
    """
    Payload field type &amp; index information
    """

    data_type: "PayloadSchemaType" = Field(..., description="Payload field type &amp; index information")


class PayloadSchemaType(str, Enum):
    KEYWORD = "keyword"
    INTEGER = "integer"
    FLOAT = "float"
    GEO = "geo"


class PayloadSelectorExclude(BaseModel):
    exclude: List[str] = Field(..., description="Exclude this fields from returning payload")


class PayloadSelectorInclude(BaseModel):
    include: List[str] = Field(..., description="Only include this payload keys")


class PeerInfo(BaseModel):
    """
    Information of a peer in the cluster
    """

    uri: str = Field(..., description="Information of a peer in the cluster")


class PointIdsList(BaseModel):
    points: List["ExtendedPointId"] = Field(..., description="")


class PointRequest(BaseModel):
    ids: List["ExtendedPointId"] = Field(..., description="Look for points with ids")
    with_payload: Optional["WithPayloadInterface"] = Field(
        None, description="Select which payload to return with the response. Default: All"
    )
    with_vector: Optional[bool] = Field(False, description="Whether to return the point vector with the result?")


class PointStruct(BaseModel):
    id: "ExtendedPointId" = Field(..., description="")
    vector: List[float] = Field(..., description="Vector")
    payload: Optional["Payload"] = Field(None, description="Payload values (optional)")


class PointsBatch(BaseModel):
    batch: "Batch" = Field(..., description="")


class PointsList(BaseModel):
    points: List["PointStruct"] = Field(..., description="")


class RaftInfo(BaseModel):
    """
    Summary information about the current raft state
    """

    term: int = Field(
        ...,
        description="Raft divides time into terms of arbitrary length, each beginning with an election. If a candidate wins the election, it remains the leader for the rest of the term. The term number increases monotonically. Each server stores the current term number which is also exchanged in every communication.",
    )
    commit: int = Field(
        ..., description="The index of the latest committed (finalized) operation that this peer is aware of."
    )
    pending_operations: int = Field(
        ..., description="Number of consensus operations pending to be applied on this peer"
    )
    leader: Optional[int] = Field(None, description="Leader of the current term")
    role: Optional["StateRole"] = Field(None, description="Role of this peer in the current term")


class Range(BaseModel):
    """
    Range filter request
    """

    lt: Optional[float] = Field(None, description="point.key &lt; range.lt")
    gt: Optional[float] = Field(None, description="point.key &gt; range.gt")
    gte: Optional[float] = Field(None, description="point.key &gt;= range.gte")
    lte: Optional[float] = Field(None, description="point.key &lt;= range.lte")


class RecommendRequest(BaseModel):
    """
    Recommendation request. Provides positive and negative examples of the vectors, which are already stored in the collection.  Service should look for the points which are closer to positive examples and at the same time further to negative examples. The concrete way of how to compare negative and positive distances is up to implementation in `segment` crate.
    """

    positive: List["ExtendedPointId"] = Field(..., description="Look for vectors closest to those")
    negative: List["ExtendedPointId"] = Field(..., description="Try to avoid vectors like this")
    filter: Optional["Filter"] = Field(None, description="Look only for points which satisfies this conditions")
    params: Optional["SearchParams"] = Field(None, description="Additional search params")
    top: int = Field(..., description="Max number of result to return")
    with_payload: Optional["WithPayloadInterface"] = Field(
        None, description="Select which payload to return with the response. Default: None"
    )
    with_vector: Optional[bool] = Field(False, description="Whether to return the point vector with the result?")
    score_threshold: Optional[float] = Field(
        None,
        description="Define a minimal score threshold for the result. If defined, less similar results will not be returned. Score of the returned result might be higher or smaller than the threshold depending on the Distance function used. E.g. for cosine similarity only higher scores will be returned.",
    )


class Record(BaseModel):
    """
    Point data
    """

    id: "ExtendedPointId" = Field(..., description="Point data")
    payload: Optional["Payload"] = Field(None, description="Payload - values assigned to the point")
    vector: Optional[List[float]] = Field(None, description="Vector of the point")


class RenameAlias(BaseModel):
    """
    Change alias to a new one
    """

    old_alias_name: str = Field(..., description="Change alias to a new one")
    new_alias_name: str = Field(..., description="Change alias to a new one")


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
    version: int = Field(..., description="Point version")
    score: float = Field(..., description="Points vector distance to the query vector")
    payload: Optional["Payload"] = Field(None, description="Payload - values assigned to the point")
    vector: Optional[List[float]] = Field(None, description="Vector of the point")


class ScrollRequest(BaseModel):
    """
    Scroll request - paginate over all points which matches given condition
    """

    offset: Optional["ExtendedPointId"] = Field(None, description="Start ID to read points from.")
    limit: Optional[int] = Field(None, description="Page size. Default: 10")
    filter: Optional["Filter"] = Field(
        None, description="Look only for points which satisfies this conditions. If not provided - all points."
    )
    with_payload: Optional["WithPayloadInterface"] = Field(
        None, description="Select which payload to return with the response. Default: All"
    )
    with_vector: Optional[bool] = Field(False, description="Whether to return the point vector with the result?")


class ScrollResult(BaseModel):
    """
    Result of the points read request
    """

    points: List["Record"] = Field(..., description="List of retrieved points")
    next_page_offset: Optional["ExtendedPointId"] = Field(
        None, description="Offset which should be used to retrieve a next page result"
    )


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

    vector: List[float] = Field(..., description="Look for vectors closest to this")
    filter: Optional["Filter"] = Field(None, description="Look only for points which satisfies this conditions")
    params: Optional["SearchParams"] = Field(None, description="Additional search params")
    top: int = Field(..., description="Max number of result to return")
    with_payload: Optional["WithPayloadInterface"] = Field(
        None, description="Select which payload to return with the response. Default: None"
    )
    with_vector: Optional[bool] = Field(False, description="Whether to return the point vector with the result?")
    score_threshold: Optional[float] = Field(
        None,
        description="Define a minimal score threshold for the result. If defined, less similar results will not be returned. Score of the returned result might be higher or smaller than the threshold depending on the Distance function used. E.g. for cosine similarity only higher scores will be returned.",
    )


class SetPayload(BaseModel):
    payload: "Payload" = Field(..., description="")
    points: List["ExtendedPointId"] = Field(..., description="Assigns payload to each point in this list")


class StateRole(str, Enum):
    FOLLOWER = "Follower"
    CANDIDATE = "Candidate"
    LEADER = "Leader"
    PRECANDIDATE = "PreCandidate"


class UpdateCollection(BaseModel):
    """
    Operation for updating parameters of the existing collection
    """

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


class ValuesCount(BaseModel):
    """
    Values count filter request
    """

    lt: Optional[int] = Field(None, description="point.key.length() &lt; values_count.lt")
    gt: Optional[int] = Field(None, description="point.key.length() &gt; values_count.gt")
    gte: Optional[int] = Field(None, description="point.key.length() &gt;= values_count.gte")
    lte: Optional[int] = Field(None, description="point.key.length() &lt;= values_count.lte")


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
ClusterStatus = Union[
    ClusterStatusOneOf,
    ClusterStatusOneOf1,
]
Condition = Union[
    FieldCondition,
    IsEmptyCondition,
    HasIdCondition,
    Filter,
]
ExtendedPointId = Union[
    StrictInt,
    StrictStr,
]
Match = Union[
    MatchValue,
    MatchKeyword,
    MatchInteger,
]
OptimizersStatus = Union[
    OptimizersStatusOneOf,
    OptimizersStatusOneOf1,
]
PayloadSelector = Union[
    PayloadSelectorInclude,
    PayloadSelectorExclude,
]
PointInsertOperations = Union[
    PointsBatch,
    PointsList,
]
PointsSelector = Union[
    PointIdsList,
    FilterSelector,
]
ValueVariants = Union[
    StrictBool,
    StrictInt,
    StrictStr,
]
WithPayloadInterface = Union[
    PayloadSelector,
    List[StrictStr],
    StrictBool,
]
