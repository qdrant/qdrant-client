from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union

try:
    from typing import Literal
except ImportError:
    # Python 3.7 backport
    from typing_extensions import Literal

from pydantic import BaseModel, Field
from pydantic.types import StrictBool, StrictFloat, StrictInt, StrictStr

Payload = Dict[str, Any]
VectorsConfigDiff = Dict[str, "VectorParamsDiff"]


class AbortTransferOperation(BaseModel, extra="forbid"):
    abort_transfer: "MoveShard" = Field(..., description="")


class AliasDescription(BaseModel, extra="forbid"):
    alias_name: str = Field(..., description="")
    collection_name: str = Field(..., description="")


class AppBuildTelemetry(BaseModel, extra="forbid"):
    name: str = Field(..., description="")
    version: str = Field(..., description="")
    features: Optional["AppFeaturesTelemetry"] = Field(default=None, description="")
    system: Optional["RunningEnvironmentTelemetry"] = Field(default=None, description="")
    startup: datetime = Field(..., description="")


class AppFeaturesTelemetry(BaseModel, extra="forbid"):
    debug: bool = Field(..., description="")
    web_feature: bool = Field(..., description="")
    service_debug_feature: bool = Field(..., description="")
    recovery_mode: bool = Field(..., description="")


class Batch(BaseModel, extra="forbid"):
    ids: List["ExtendedPointId"] = Field(..., description="")
    vectors: "BatchVectorStruct" = Field(..., description="")
    payloads: Optional[List["Payload"]] = Field(default=None, description="")


class ChangeAliasesOperation(BaseModel, extra="forbid"):
    """
    Operation for performing changes of collection aliases. Alias changes are atomic, meaning that no collection modifications can happen between alias operations.
    """

    actions: List["AliasOperations"] = Field(
        ...,
        description="Operation for performing changes of collection aliases. Alias changes are atomic, meaning that no collection modifications can happen between alias operations.",
    )


class ClusterConfigTelemetry(BaseModel, extra="forbid"):
    grpc_timeout_ms: int = Field(..., description="")
    p2p: "P2pConfigTelemetry" = Field(..., description="")
    consensus: "ConsensusConfigTelemetry" = Field(..., description="")


class ClusterStatusOneOf(BaseModel, extra="forbid"):
    status: Literal[
        "disabled",
    ] = Field(..., description="")


class ClusterStatusOneOf1(BaseModel, extra="forbid"):
    """
    Description of enabled cluster
    """

    status: Literal[
        "enabled",
    ] = Field(..., description="Description of enabled cluster")
    peer_id: int = Field(..., description="ID of this peer")
    peers: Dict[str, "PeerInfo"] = Field(..., description="Peers composition of the cluster with main information")
    raft_info: "RaftInfo" = Field(..., description="Description of enabled cluster")
    consensus_thread_status: "ConsensusThreadStatus" = Field(..., description="Description of enabled cluster")
    message_send_failures: Dict[str, "MessageSendErrors"] = Field(
        ...,
        description="Consequent failures of message send operations in consensus by peer address. On the first success to send to that peer - entry is removed from this hashmap.",
    )


class ClusterStatusTelemetry(BaseModel, extra="forbid"):
    number_of_peers: int = Field(..., description="")
    term: int = Field(..., description="")
    commit: int = Field(..., description="")
    pending_operations: int = Field(..., description="")
    role: Optional["StateRole"] = Field(default=None, description="")
    is_voter: bool = Field(..., description="")
    peer_id: Optional[int] = Field(default=None, description="")
    consensus_thread_status: "ConsensusThreadStatus" = Field(..., description="")


class ClusterTelemetry(BaseModel, extra="forbid"):
    enabled: bool = Field(..., description="")
    status: Optional["ClusterStatusTelemetry"] = Field(default=None, description="")
    config: Optional["ClusterConfigTelemetry"] = Field(default=None, description="")


class CollectionClusterInfo(BaseModel, extra="forbid"):
    """
    Current clustering distribution for the collection
    """

    peer_id: int = Field(..., description="ID of this peer")
    shard_count: int = Field(..., description="Total number of shards")
    local_shards: List["LocalShardInfo"] = Field(..., description="Local shards")
    remote_shards: List["RemoteShardInfo"] = Field(..., description="Remote shards")
    shard_transfers: List["ShardTransferInfo"] = Field(..., description="Shard transfers")


class CollectionConfig(BaseModel, extra="forbid"):
    params: "CollectionParams" = Field(..., description="")
    hnsw_config: "HnswConfig" = Field(..., description="")
    optimizer_config: "OptimizersConfig" = Field(..., description="")
    wal_config: "WalConfig" = Field(..., description="")
    quantization_config: Optional["QuantizationConfig"] = Field(default=None, description="")


class CollectionDescription(BaseModel, extra="forbid"):
    name: str = Field(..., description="")


class CollectionInfo(BaseModel, extra="forbid"):
    """
    Current statistics and configuration of the collection
    """

    status: "CollectionStatus" = Field(..., description="Current statistics and configuration of the collection")
    optimizer_status: "OptimizersStatus" = Field(
        ..., description="Current statistics and configuration of the collection"
    )
    vectors_count: int = Field(
        ...,
        description="Number of vectors in collection All vectors in collection are available for querying Calculated as `points_count x vectors_per_point` Where `vectors_per_point` is a number of named vectors in schema",
    )
    indexed_vectors_count: int = Field(
        ...,
        description="Number of indexed vectors in the collection. Indexed vectors in large segments are faster to query, as it is stored in vector index (HNSW)",
    )
    points_count: int = Field(
        ..., description="Number of points (vectors + payloads) in collection Each point could be accessed by unique id"
    )
    segments_count: int = Field(
        ..., description="Number of segments in collection. Each segment has independent vector as payload indexes"
    )
    config: "CollectionConfig" = Field(..., description="Current statistics and configuration of the collection")
    payload_schema: Dict[str, "PayloadIndexInfo"] = Field(..., description="Types of stored payload")


class CollectionParams(BaseModel, extra="forbid"):
    vectors: "VectorsConfig" = Field(..., description="")
    shard_number: Optional[int] = Field(default=1, description="Number of shards the collection has")
    replication_factor: Optional[int] = Field(default=1, description="Number of replicas for each shard")
    write_consistency_factor: Optional[int] = Field(
        default=1,
        description="Defines how many replicas should apply the operation for us to consider it successful. Increasing this number will make the collection more resilient to inconsistencies, but will also make it fail if not enough replicas are available. Does not have any performance impact.",
    )
    on_disk_payload: Optional[bool] = Field(
        default=False,
        description="If true - point&#x27;s payload will not be stored in memory. It will be read from the disk every time it is requested. This setting saves RAM by (slightly) increasing the response time. Note: those payload values that are involved in filtering and are indexed - remain in RAM.",
    )


class CollectionParamsDiff(BaseModel, extra="forbid"):
    replication_factor: Optional[int] = Field(default=None, description="Number of replicas for each shard")
    write_consistency_factor: Optional[int] = Field(
        default=None, description="Minimal number successful responses from replicas to consider operation successful"
    )
    on_disk_payload: Optional[bool] = Field(
        default=None,
        description="If true - point&#x27;s payload will not be stored in memory. It will be read from the disk every time it is requested. This setting saves RAM by (slightly) increasing the response time. Note: those payload values that are involved in filtering and are indexed - remain in RAM.",
    )


class CollectionStatus(str, Enum):
    GREEN = "green"
    YELLOW = "yellow"
    RED = "red"


class CollectionTelemetry(BaseModel, extra="forbid"):
    id: str = Field(..., description="")
    init_time_ms: int = Field(..., description="")
    config: "CollectionConfig" = Field(..., description="")
    shards: List["ReplicaSetTelemetry"] = Field(..., description="")
    transfers: List["ShardTransferInfo"] = Field(..., description="")


class CollectionsAggregatedTelemetry(BaseModel, extra="forbid"):
    vectors: int = Field(..., description="")
    optimizers_status: "OptimizersStatus" = Field(..., description="")
    params: "CollectionParams" = Field(..., description="")


class CollectionsAliasesResponse(BaseModel, extra="forbid"):
    aliases: List["AliasDescription"] = Field(..., description="")


class CollectionsResponse(BaseModel, extra="forbid"):
    collections: List["CollectionDescription"] = Field(..., description="")


class CollectionsTelemetry(BaseModel, extra="forbid"):
    number_of_collections: int = Field(..., description="")
    collections: Optional[List["CollectionTelemetryEnum"]] = Field(default=None, description="")


class CompressionRatio(str, Enum):
    X4 = "x4"
    X8 = "x8"
    X16 = "x16"
    X32 = "x32"
    X64 = "x64"


class ConsensusConfigTelemetry(BaseModel, extra="forbid"):
    max_message_queue_size: int = Field(..., description="")
    tick_period_ms: int = Field(..., description="")
    bootstrap_timeout_sec: int = Field(..., description="")


class ConsensusThreadStatusOneOf(BaseModel, extra="forbid"):
    consensus_thread_status: Literal[
        "working",
    ] = Field(..., description="")
    last_update: datetime = Field(..., description="")


class ConsensusThreadStatusOneOf1(BaseModel, extra="forbid"):
    consensus_thread_status: Literal[
        "stopped",
    ] = Field(..., description="")


class ConsensusThreadStatusOneOf2(BaseModel, extra="forbid"):
    consensus_thread_status: Literal[
        "stopped_with_err",
    ] = Field(..., description="")
    err: str = Field(..., description="")


class CountRequest(BaseModel, extra="forbid"):
    """
    Count Request Counts the number of points which satisfy the given filter. If filter is not provided, the count of all points in the collection will be returned.
    """

    filter: Optional["Filter"] = Field(default=None, description="Look only for points which satisfies this conditions")
    exact: Optional[bool] = Field(
        default=True,
        description="If true, count exact number of points. If false, count approximate number of points faster. Approximate count might be unreliable during the indexing process. Default: true",
    )


class CountResult(BaseModel, extra="forbid"):
    count: int = Field(..., description="Number of points which satisfy the conditions")


class CreateAlias(BaseModel, extra="forbid"):
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


class CreateAliasOperation(BaseModel, extra="forbid"):
    create_alias: "CreateAlias" = Field(..., description="")


class CreateCollection(BaseModel, extra="forbid"):
    """
    Operation for creating new collection and (optionally) specify index params
    """

    vectors: "VectorsConfig" = Field(
        ..., description="Operation for creating new collection and (optionally) specify index params"
    )
    shard_number: Optional[int] = Field(
        default=None,
        description="Number of shards in collection. Default is 1 for standalone, otherwise equal to the number of nodes Minimum is 1",
    )
    replication_factor: Optional[int] = Field(
        default=None, description="Number of shards replicas. Default is 1 Minimum is 1"
    )
    write_consistency_factor: Optional[int] = Field(
        default=None,
        description="Defines how many replicas should apply the operation for us to consider it successful. Increasing this number will make the collection more resilient to inconsistencies, but will also make it fail if not enough replicas are available. Does not have any performance impact.",
    )
    on_disk_payload: Optional[bool] = Field(
        default=None,
        description="If true - point&#x27;s payload will not be stored in memory. It will be read from the disk every time it is requested. This setting saves RAM by (slightly) increasing the response time. Note: those payload values that are involved in filtering and are indexed - remain in RAM.",
    )
    hnsw_config: Optional["HnswConfigDiff"] = Field(
        default=None,
        description="Custom params for HNSW index. If none - values from service configuration file are used.",
    )
    wal_config: Optional["WalConfigDiff"] = Field(
        default=None, description="Custom params for WAL. If none - values from service configuration file are used."
    )
    optimizers_config: Optional["OptimizersConfigDiff"] = Field(
        default=None,
        description="Custom params for Optimizers.  If none - values from service configuration file are used.",
    )
    init_from: Optional["InitFrom"] = Field(default=None, description="Specify other collection to copy data from.")
    quantization_config: Optional["QuantizationConfig"] = Field(
        default=None, description="Quantization parameters. If none - quantization is disabled."
    )


class CreateFieldIndex(BaseModel, extra="forbid"):
    field_name: str = Field(..., description="")
    field_schema: Optional["PayloadFieldSchema"] = Field(default=None, description="")


class DeleteAlias(BaseModel, extra="forbid"):
    """
    Delete alias if exists
    """

    alias_name: str = Field(..., description="Delete alias if exists")


class DeleteAliasOperation(BaseModel, extra="forbid"):
    """
    Delete alias if exists
    """

    delete_alias: "DeleteAlias" = Field(..., description="Delete alias if exists")


class DeletePayload(BaseModel, extra="forbid"):
    keys: List[str] = Field(..., description="List of payload keys to remove from payload")
    points: Optional[List["ExtendedPointId"]] = Field(
        default=None, description="Deletes values from each point in this list"
    )
    filter: Optional["Filter"] = Field(
        default=None, description="Deletes values from points that satisfy this filter condition"
    )


class DeleteVectors(BaseModel, extra="forbid"):
    points: Optional[List["ExtendedPointId"]] = Field(
        default=None, description="Deletes values from each point in this list"
    )
    filter: Optional["Filter"] = Field(
        default=None, description="Deletes values from points that satisfy this filter condition"
    )
    vector: List[str] = Field(..., description="Vector names")


class Disabled(str, Enum):
    DISABLED = "Disabled"


class Distance(str, Enum):
    COSINE = "Cosine"
    EUCLID = "Euclid"
    DOT = "Dot"


class DropReplicaOperation(BaseModel, extra="forbid"):
    drop_replica: "Replica" = Field(..., description="")


class ErrorResponse(BaseModel, extra="forbid"):
    time: Optional[float] = Field(default=None, description="Time spent to process this request")
    status: Optional["ErrorResponseStatus"] = Field(default=None, description="")
    result: Optional[Any] = Field(default=None, description="")


class ErrorResponseStatus(BaseModel, extra="forbid"):
    error: Optional[str] = Field(default=None, description="Description of the occurred error.")


class FieldCondition(BaseModel, extra="forbid"):
    """
    All possible payload filtering conditions
    """

    key: str = Field(..., description="Payload key")
    match: Optional["Match"] = Field(default=None, description="Check if point has field with a given value")
    range: Optional["Range"] = Field(default=None, description="Check if points value lies in a given range")
    geo_bounding_box: Optional["GeoBoundingBox"] = Field(
        default=None, description="Check if points geo location lies in a given area"
    )
    geo_radius: Optional["GeoRadius"] = Field(default=None, description="Check if geo point is within a given radius")
    values_count: Optional["ValuesCount"] = Field(default=None, description="Check number of values of the field")


class Filter(BaseModel, extra="forbid"):
    should: Optional[List["Condition"]] = Field(
        default=None, description="At least one of those conditions should match"
    )
    must: Optional[List["Condition"]] = Field(default=None, description="All conditions must match")
    must_not: Optional[List["Condition"]] = Field(default=None, description="All conditions must NOT match")


class FilterSelector(BaseModel, extra="forbid"):
    filter: "Filter" = Field(..., description="")


class GeoBoundingBox(BaseModel, extra="forbid"):
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


class GeoPoint(BaseModel, extra="forbid"):
    """
    Geo point payload schema
    """

    lon: float = Field(..., description="Geo point payload schema")
    lat: float = Field(..., description="Geo point payload schema")


class GeoRadius(BaseModel, extra="forbid"):
    """
    Geo filter request  Matches coordinates inside the circle of `radius` and center with coordinates `center`
    """

    center: "GeoPoint" = Field(
        ...,
        description="Geo filter request  Matches coordinates inside the circle of `radius` and center with coordinates `center`",
    )
    radius: float = Field(..., description="Radius of the area in meters")


class GroupsResult(BaseModel, extra="forbid"):
    groups: List["PointGroup"] = Field(..., description="")


class GrpcTelemetry(BaseModel, extra="forbid"):
    responses: Dict[str, "OperationDurationStatistics"] = Field(..., description="")


class HasIdCondition(BaseModel, extra="forbid"):
    """
    ID-based filtering condition
    """

    has_id: List["ExtendedPointId"] = Field(..., description="ID-based filtering condition")


class HnswConfig(BaseModel, extra="forbid"):
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
    max_indexing_threads: Optional[int] = Field(
        default=0, description="Number of parallel threads used for background index building. If 0 - auto selection."
    )
    on_disk: Optional[bool] = Field(
        default=None,
        description="Store HNSW index on disk. If set to false, index will be stored in RAM. Default: false",
    )
    payload_m: Optional[int] = Field(
        default=None,
        description="Custom M param for hnsw graph built for payload index. If not set, default M will be used.",
    )


class HnswConfigDiff(BaseModel, extra="forbid"):
    m: Optional[int] = Field(
        default=None,
        description="Number of edges per node in the index graph. Larger the value - more accurate the search, more space required.",
    )
    ef_construct: Optional[int] = Field(
        default=None,
        description="Number of neighbours to consider during the index building. Larger the value - more accurate the search, more time required to build the index.",
    )
    full_scan_threshold: Optional[int] = Field(
        default=None,
        description="Minimal size (in kilobytes) of vectors for additional payload-based indexing. If payload chunk is smaller than `full_scan_threshold_kb` additional indexing won&#x27;t be used - in this case full-scan search should be preferred by query planner and additional indexing is not required. Note: 1Kb = 1 vector of size 256",
    )
    max_indexing_threads: Optional[int] = Field(
        default=None,
        description="Number of parallel threads used for background index building. If 0 - auto selection.",
    )
    on_disk: Optional[bool] = Field(
        default=None,
        description="Store HNSW index on disk. If set to false, the index will be stored in RAM. Default: false",
    )
    payload_m: Optional[int] = Field(
        default=None,
        description="Custom M param for additional payload-aware HNSW links. If not set, default M will be used.",
    )


class IndexesOneOf(BaseModel, extra="forbid"):
    """
    Do not use any index, scan whole vector collection during search. Guarantee 100% precision, but may be time consuming on large collections.
    """

    type: Literal["plain",] = Field(
        ...,
        description="Do not use any index, scan whole vector collection during search. Guarantee 100% precision, but may be time consuming on large collections.",
    )
    options: Any = Field(
        ...,
        description="Do not use any index, scan whole vector collection during search. Guarantee 100% precision, but may be time consuming on large collections.",
    )


class IndexesOneOf1(BaseModel, extra="forbid"):
    """
    Use filterable HNSW index for approximate search. Is very fast even on a very huge collections, but require additional space to store index and additional time to build it.
    """

    type: Literal["hnsw",] = Field(
        ...,
        description="Use filterable HNSW index for approximate search. Is very fast even on a very huge collections, but require additional space to store index and additional time to build it.",
    )
    options: "HnswConfig" = Field(
        ...,
        description="Use filterable HNSW index for approximate search. Is very fast even on a very huge collections, but require additional space to store index and additional time to build it.",
    )


class InitFrom(BaseModel, extra="forbid"):
    """
    Operation for creating new collection and (optionally) specify index params
    """

    collection: str = Field(
        ..., description="Operation for creating new collection and (optionally) specify index params"
    )


class InlineResponse200(BaseModel, extra="forbid"):
    time: Optional[float] = Field(default=None, description="Time spent to process this request")
    status: Literal[
        "ok",
    ] = Field(None, description="")
    result: Optional["TelemetryData"] = Field(default=None, description="")


class InlineResponse2001(BaseModel, extra="forbid"):
    time: Optional[float] = Field(default=None, description="Time spent to process this request")
    status: Literal[
        "ok",
    ] = Field(None, description="")
    result: Optional["LocksOption"] = Field(default=None, description="")


class InlineResponse20010(BaseModel, extra="forbid"):
    time: Optional[float] = Field(default=None, description="Time spent to process this request")
    status: Literal[
        "ok",
    ] = Field(None, description="")
    result: Optional["SnapshotDescription"] = Field(default=None, description="")


class InlineResponse20011(BaseModel, extra="forbid"):
    time: Optional[float] = Field(default=None, description="Time spent to process this request")
    status: Literal[
        "ok",
    ] = Field(None, description="")
    result: Optional["Record"] = Field(default=None, description="")


class InlineResponse20012(BaseModel, extra="forbid"):
    time: Optional[float] = Field(default=None, description="Time spent to process this request")
    status: Literal[
        "ok",
    ] = Field(None, description="")
    result: Optional[List["Record"]] = Field(default=None, description="")


class InlineResponse20013(BaseModel, extra="forbid"):
    time: Optional[float] = Field(default=None, description="Time spent to process this request")
    status: Literal[
        "ok",
    ] = Field(None, description="")
    result: Optional["ScrollResult"] = Field(default=None, description="")


class InlineResponse20014(BaseModel, extra="forbid"):
    time: Optional[float] = Field(default=None, description="Time spent to process this request")
    status: Literal[
        "ok",
    ] = Field(None, description="")
    result: Optional[List["ScoredPoint"]] = Field(default=None, description="")


class InlineResponse20015(BaseModel, extra="forbid"):
    time: Optional[float] = Field(default=None, description="Time spent to process this request")
    status: Literal[
        "ok",
    ] = Field(None, description="")
    result: Optional[List[List["ScoredPoint"]]] = Field(default=None, description="")


class InlineResponse20016(BaseModel, extra="forbid"):
    time: Optional[float] = Field(default=None, description="Time spent to process this request")
    status: Literal[
        "ok",
    ] = Field(None, description="")
    result: Optional["GroupsResult"] = Field(default=None, description="")


class InlineResponse20017(BaseModel, extra="forbid"):
    time: Optional[float] = Field(default=None, description="Time spent to process this request")
    status: Literal[
        "ok",
    ] = Field(None, description="")
    result: Optional["CountResult"] = Field(default=None, description="")


class InlineResponse2002(BaseModel, extra="forbid"):
    time: Optional[float] = Field(default=None, description="Time spent to process this request")
    status: Literal[
        "ok",
    ] = Field(None, description="")
    result: Optional["ClusterStatus"] = Field(default=None, description="")


class InlineResponse2003(BaseModel, extra="forbid"):
    time: Optional[float] = Field(default=None, description="Time spent to process this request")
    status: Literal[
        "ok",
    ] = Field(None, description="")
    result: Optional[bool] = Field(default=None, description="")


class InlineResponse2004(BaseModel, extra="forbid"):
    time: Optional[float] = Field(default=None, description="Time spent to process this request")
    status: Literal[
        "ok",
    ] = Field(None, description="")
    result: Optional["CollectionsResponse"] = Field(default=None, description="")


class InlineResponse2005(BaseModel, extra="forbid"):
    time: Optional[float] = Field(default=None, description="Time spent to process this request")
    status: Literal[
        "ok",
    ] = Field(None, description="")
    result: Optional["CollectionInfo"] = Field(default=None, description="")


class InlineResponse2006(BaseModel, extra="forbid"):
    time: Optional[float] = Field(default=None, description="Time spent to process this request")
    status: Literal[
        "ok",
    ] = Field(None, description="")
    result: Optional["UpdateResult"] = Field(default=None, description="")


class InlineResponse2007(BaseModel, extra="forbid"):
    time: Optional[float] = Field(default=None, description="Time spent to process this request")
    status: Literal[
        "ok",
    ] = Field(None, description="")
    result: Optional["CollectionClusterInfo"] = Field(default=None, description="")


class InlineResponse2008(BaseModel, extra="forbid"):
    time: Optional[float] = Field(default=None, description="Time spent to process this request")
    status: Literal[
        "ok",
    ] = Field(None, description="")
    result: Optional["CollectionsAliasesResponse"] = Field(default=None, description="")


class InlineResponse2009(BaseModel, extra="forbid"):
    time: Optional[float] = Field(default=None, description="Time spent to process this request")
    status: Literal[
        "ok",
    ] = Field(None, description="")
    result: Optional[List["SnapshotDescription"]] = Field(default=None, description="")


class InlineResponse202(BaseModel, extra="forbid"):
    time: Optional[float] = Field(default=None, description="Time spent to process this request")
    status: Literal[
        "accepted",
    ] = Field(None, description="")


class IsEmptyCondition(BaseModel, extra="forbid"):
    """
    Select points with empty payload for a specified field
    """

    is_empty: "PayloadField" = Field(..., description="Select points with empty payload for a specified field")


class IsNullCondition(BaseModel, extra="forbid"):
    """
    Select points with null payload for a specified field
    """

    is_null: "PayloadField" = Field(..., description="Select points with null payload for a specified field")


class LocalShardInfo(BaseModel, extra="forbid"):
    shard_id: int = Field(..., description="Local shard id")
    points_count: int = Field(..., description="Number of points in the shard")
    state: "ReplicaState" = Field(..., description="")


class LocalShardTelemetry(BaseModel, extra="forbid"):
    variant_name: Optional[str] = Field(default=None, description="")
    segments: List["SegmentTelemetry"] = Field(..., description="")
    optimizations: "OptimizerTelemetry" = Field(..., description="")


class LocksOption(BaseModel, extra="forbid"):
    error_message: Optional[str] = Field(default=None, description="")
    write: bool = Field(..., description="")


class LookupLocation(BaseModel, extra="forbid"):
    """
    Defines a location to use for looking up the vector. Specifies collection and vector field name.
    """

    collection: str = Field(..., description="Name of the collection used for lookup")
    vector: Optional[str] = Field(
        default=None,
        description="Optional name of the vector field within the collection. If not provided, the default vector field will be used.",
    )


class MatchAny(BaseModel, extra="forbid"):
    """
    Exact match on any of the given values
    """

    any: "AnyVariants" = Field(..., description="Exact match on any of the given values")


class MatchExcept(BaseModel, extra="forbid"):
    """
    Should have at least one value not matching the any given values
    """

    except_: "AnyVariants" = Field(
        ..., description="Should have at least one value not matching the any given values", alias="except"
    )


class MatchText(BaseModel, extra="forbid"):
    """
    Full-text match of the strings.
    """

    text: str = Field(..., description="Full-text match of the strings.")


class MatchValue(BaseModel, extra="forbid"):
    """
    Exact match of the given value
    """

    value: "ValueVariants" = Field(..., description="Exact match of the given value")


class MessageSendErrors(BaseModel, extra="forbid"):
    """
    Message send failures for a particular peer
    """

    count: int = Field(..., description="Message send failures for a particular peer")
    latest_error: Optional[str] = Field(default=None, description="Message send failures for a particular peer")


class MoveShard(BaseModel, extra="forbid"):
    shard_id: int = Field(..., description="")
    to_peer_id: int = Field(..., description="")
    from_peer_id: int = Field(..., description="")


class MoveShardOperation(BaseModel, extra="forbid"):
    move_shard: "MoveShard" = Field(..., description="")


class NamedVector(BaseModel, extra="forbid"):
    """
    Vector data with name
    """

    name: str = Field(..., description="Name of vector data")
    vector: List[float] = Field(..., description="Vector data")


class Nested(BaseModel, extra="forbid"):
    """
    Select points with payload for a specified nested field
    """

    key: str = Field(..., description="Select points with payload for a specified nested field")
    filter: "Filter" = Field(..., description="Select points with payload for a specified nested field")


class NestedCondition(BaseModel, extra="forbid"):
    nested: "Nested" = Field(..., description="")


class OperationDurationStatistics(BaseModel, extra="forbid"):
    count: int = Field(..., description="")
    fail_count: Optional[int] = Field(default=None, description="")
    avg_duration_micros: Optional[float] = Field(default=None, description="")
    min_duration_micros: Optional[float] = Field(default=None, description="")
    max_duration_micros: Optional[float] = Field(default=None, description="")
    last_responded: Optional[datetime] = Field(default=None, description="")


class OptimizerTelemetry(BaseModel, extra="forbid"):
    status: "OptimizersStatus" = Field(..., description="")
    optimizations: "OperationDurationStatistics" = Field(..., description="")


class OptimizersConfig(BaseModel, extra="forbid"):
    deleted_threshold: float = Field(
        ...,
        description="The minimal fraction of deleted vectors in a segment, required to perform segment optimization",
    )
    vacuum_min_vector_number: int = Field(
        ..., description="The minimal number of vectors in a segment, required to perform segment optimization"
    )
    default_segment_number: int = Field(
        ...,
        description="Target amount of segments optimizer will try to keep. Real amount of segments may vary depending on multiple parameters: - Amount of stored points - Current write RPS  It is recommended to select default number of segments as a factor of the number of search threads, so that each segment would be handled evenly by one of the threads. If `default_segment_number = 0`, will be automatically selected by the number of available CPUs.",
    )
    max_segment_size: Optional[int] = Field(
        default=None,
        description="Do not create segments larger this size (in kilobytes). Large segments might require disproportionately long indexation times, therefore it makes sense to limit the size of segments.  If indexing speed is more important - make this parameter lower. If search speed is more important - make this parameter higher. Note: 1Kb = 1 vector of size 256 If not set, will be automatically selected considering the number of available CPUs.",
    )
    memmap_threshold: Optional[int] = Field(
        default=None,
        description="Maximum size (in kilobytes) of vectors to store in-memory per segment. Segments larger than this threshold will be stored as read-only memmaped file.  Memmap storage is disabled by default, to enable it, set this threshold to a reasonable value.  To disable memmap storage, set this to `0`. Internally it will use the largest threshold possible.  Note: 1Kb = 1 vector of size 256",
    )
    indexing_threshold: Optional[int] = Field(
        default=None,
        description="Maximum size (in kilobytes) of vectors allowed for plain index, exceeding this threshold will enable vector indexing  Default value is 20,000, based on &lt;https://github.com/google-research/google-research/blob/master/scann/docs/algorithms.md&gt;.  To disable vector indexing, set to `0`.  Note: 1kB = 1 vector of size 256.",
    )
    flush_interval_sec: int = Field(..., description="Minimum interval between forced flushes.")
    max_optimization_threads: int = Field(..., description="Maximum available threads for optimization workers")


class OptimizersConfigDiff(BaseModel, extra="forbid"):
    deleted_threshold: Optional[float] = Field(
        default=None,
        description="The minimal fraction of deleted vectors in a segment, required to perform segment optimization",
    )
    vacuum_min_vector_number: Optional[int] = Field(
        default=None, description="The minimal number of vectors in a segment, required to perform segment optimization"
    )
    default_segment_number: Optional[int] = Field(
        default=None,
        description="Target amount of segments optimizer will try to keep. Real amount of segments may vary depending on multiple parameters: - Amount of stored points - Current write RPS  It is recommended to select default number of segments as a factor of the number of search threads, so that each segment would be handled evenly by one of the threads If `default_segment_number = 0`, will be automatically selected by the number of available CPUs",
    )
    max_segment_size: Optional[int] = Field(
        default=None,
        description="Do not create segments larger this size (in kilobytes). Large segments might require disproportionately long indexation times, therefore it makes sense to limit the size of segments.  If indexation speed have more priority for your - make this parameter lower. If search speed is more important - make this parameter higher. Note: 1Kb = 1 vector of size 256",
    )
    memmap_threshold: Optional[int] = Field(
        default=None,
        description="Maximum size (in kilobytes) of vectors to store in-memory per segment. Segments larger than this threshold will be stored as read-only memmaped file.  Memmap storage is disabled by default, to enable it, set this threshold to a reasonable value.  To disable memmap storage, set this to `0`.  Note: 1Kb = 1 vector of size 256",
    )
    indexing_threshold: Optional[int] = Field(
        default=None,
        description="Maximum size (in kilobytes) of vectors allowed for plain index, exceeding this threshold will enable vector indexing  Default value is 20,000, based on &lt;https://github.com/google-research/google-research/blob/master/scann/docs/algorithms.md&gt;.  To disable vector indexing, set to `0`.  Note: 1kB = 1 vector of size 256.",
    )
    flush_interval_sec: Optional[int] = Field(default=None, description="Minimum interval between forced flushes.")
    max_optimization_threads: Optional[int] = Field(
        default=None, description="Maximum available threads for optimization workers"
    )


class OptimizersStatusOneOf(str, Enum):
    OK = "ok"


class OptimizersStatusOneOf1(BaseModel, extra="forbid"):
    """
    Something wrong happened with optimizers
    """

    error: str = Field(..., description="Something wrong happened with optimizers")


class P2pConfigTelemetry(BaseModel, extra="forbid"):
    connection_pool_size: int = Field(..., description="")


class PayloadField(BaseModel, extra="forbid"):
    """
    Payload field
    """

    key: str = Field(..., description="Payload field name")


class PayloadIndexInfo(BaseModel, extra="forbid"):
    """
    Display payload field type &amp; index information
    """

    data_type: "PayloadSchemaType" = Field(..., description="Display payload field type &amp; index information")
    params: Optional["PayloadSchemaParams"] = Field(
        default=None, description="Display payload field type &amp; index information"
    )
    points: int = Field(..., description="Number of points indexed with this index")


class PayloadIndexTelemetry(BaseModel, extra="forbid"):
    field_name: Optional[str] = Field(default=None, description="")
    points_values_count: int = Field(..., description="")
    points_count: int = Field(..., description="")
    histogram_bucket_size: Optional[int] = Field(default=None, description="")


class PayloadSchemaType(str, Enum):
    KEYWORD = "keyword"
    INTEGER = "integer"
    FLOAT = "float"
    GEO = "geo"
    TEXT = "text"
    BOOL = "bool"


class PayloadSelectorExclude(BaseModel, extra="forbid"):
    exclude: List[str] = Field(..., description="Exclude this fields from returning payload")


class PayloadSelectorInclude(BaseModel, extra="forbid"):
    include: List[str] = Field(..., description="Only include this payload keys")


class PayloadStorageTypeOneOf(BaseModel, extra="forbid"):
    type: Literal[
        "in_memory",
    ] = Field(..., description="")


class PayloadStorageTypeOneOf1(BaseModel, extra="forbid"):
    type: Literal[
        "on_disk",
    ] = Field(..., description="")


class PeerInfo(BaseModel, extra="forbid"):
    """
    Information of a peer in the cluster
    """

    uri: str = Field(..., description="Information of a peer in the cluster")


class PointGroup(BaseModel, extra="forbid"):
    hits: List["ScoredPoint"] = Field(..., description="Scored points that have the same value of the group_by key")
    id: "GroupId" = Field(..., description="")
    lookup: Optional["Record"] = Field(default=None, description="Record that has been looked up using the group id")


class PointIdsList(BaseModel, extra="forbid"):
    points: List["ExtendedPointId"] = Field(..., description="")


class PointRequest(BaseModel, extra="forbid"):
    ids: List["ExtendedPointId"] = Field(..., description="Look for points with ids")
    with_payload: Optional["WithPayloadInterface"] = Field(
        default=None, description="Select which payload to return with the response. Default: All"
    )
    with_vector: Optional["WithVector"] = Field(default=None, description="")


class PointStruct(BaseModel, extra="forbid"):
    id: "ExtendedPointId" = Field(..., description="")
    vector: "VectorStruct" = Field(..., description="")
    payload: Optional["Payload"] = Field(default=None, description="Payload values (optional)")


class PointVectors(BaseModel, extra="forbid"):
    id: "ExtendedPointId" = Field(..., description="")
    vector: "VectorStruct" = Field(..., description="")


class PointsBatch(BaseModel, extra="forbid"):
    batch: "Batch" = Field(..., description="")


class PointsList(BaseModel, extra="forbid"):
    points: List["PointStruct"] = Field(..., description="")


class ProductQuantization(BaseModel, extra="forbid"):
    product: "ProductQuantizationConfig" = Field(..., description="")


class ProductQuantizationConfig(BaseModel, extra="forbid"):
    compression: "CompressionRatio" = Field(..., description="")
    always_ram: Optional[bool] = Field(default=None, description="")


class QuantizationSearchParams(BaseModel, extra="forbid"):
    """
    Additional parameters of the search
    """

    ignore: Optional[bool] = Field(
        default=False, description="If true, quantized vectors are ignored. Default is false."
    )
    rescore: Optional[bool] = Field(
        default=False,
        description="If true, use original vectors to re-score top-k results. Might require more time in case if original vectors are stored on disk. Default is false.",
    )
    oversampling: Optional[float] = Field(
        default=None,
        description="Oversampling factor for quantization. Default is 1.0.  Defines how many extra vectors should be pre-selected using quantized index, and then re-scored using original vectors.  For example, if `oversampling` is 2.4 and `limit` is 100, then 240 vectors will be pre-selected using quantized index, and then top-100 will be returned after re-scoring.",
    )


class RaftInfo(BaseModel, extra="forbid"):
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
    leader: Optional[int] = Field(default=None, description="Leader of the current term")
    role: Optional["StateRole"] = Field(default=None, description="Role of this peer in the current term")
    is_voter: bool = Field(..., description="Is this peer a voter or a learner")


class Range(BaseModel, extra="forbid"):
    """
    Range filter request
    """

    lt: Optional[float] = Field(default=None, description="point.key &lt; range.lt")
    gt: Optional[float] = Field(default=None, description="point.key &gt; range.gt")
    gte: Optional[float] = Field(default=None, description="point.key &gt;= range.gte")
    lte: Optional[float] = Field(default=None, description="point.key &lt;= range.lte")


class ReadConsistencyType(str, Enum):
    MAJORITY = "majority"
    QUORUM = "quorum"
    ALL = "all"


class RecommendGroupsRequest(BaseModel, extra="forbid"):
    positive: List["ExtendedPointId"] = Field(..., description="Look for vectors closest to those")
    negative: Optional[List["ExtendedPointId"]] = Field(default=[], description="Try to avoid vectors like this")
    filter: Optional["Filter"] = Field(default=None, description="Look only for points which satisfies this conditions")
    params: Optional["SearchParams"] = Field(default=None, description="Additional search params")
    with_payload: Optional["WithPayloadInterface"] = Field(
        default=None, description="Select which payload to return with the response. Default: None"
    )
    with_vector: Optional["WithVector"] = Field(
        default=None, description="Whether to return the point vector with the result?"
    )
    score_threshold: Optional[float] = Field(
        default=None,
        description="Define a minimal score threshold for the result. If defined, less similar results will not be returned. Score of the returned result might be higher or smaller than the threshold depending on the Distance function used. E.g. for cosine similarity only higher scores will be returned.",
    )
    using: Optional["UsingVector"] = Field(
        default=None,
        description="Define which vector to use for recommendation, if not specified - try to use default vector",
    )
    lookup_from: Optional["LookupLocation"] = Field(
        default=None,
        description="The location used to lookup vectors. If not specified - use current collection. Note: the other collection should have the same vector size as the current collection",
    )
    group_by: str = Field(
        ...,
        description="Payload field to group by, must be a string or number field. If the field contains more than 1 value, all values will be used for grouping. One point can be in multiple groups.",
    )
    group_size: int = Field(..., description="Maximum amount of points to return per group")
    limit: int = Field(..., description="Maximum amount of groups to return")
    with_lookup: Optional["WithLookupInterface"] = Field(
        default=None, description="Look for points in another collection using the group ids"
    )


class RecommendRequest(BaseModel, extra="forbid"):
    """
    Recommendation request. Provides positive and negative examples of the vectors, which are already stored in the collection.  Service should look for the points which are closer to positive examples and at the same time further to negative examples. The concrete way of how to compare negative and positive distances is up to implementation in `segment` crate.
    """

    positive: List["ExtendedPointId"] = Field(..., description="Look for vectors closest to those")
    negative: Optional[List["ExtendedPointId"]] = Field(default=[], description="Try to avoid vectors like this")
    filter: Optional["Filter"] = Field(default=None, description="Look only for points which satisfies this conditions")
    params: Optional["SearchParams"] = Field(default=None, description="Additional search params")
    limit: int = Field(..., description="Max number of result to return")
    offset: Optional[int] = Field(
        default=0,
        description="Offset of the first result to return. May be used to paginate results. Note: large offset values may cause performance issues.",
    )
    with_payload: Optional["WithPayloadInterface"] = Field(
        default=None, description="Select which payload to return with the response. Default: None"
    )
    with_vector: Optional["WithVector"] = Field(
        default=None, description="Whether to return the point vector with the result?"
    )
    score_threshold: Optional[float] = Field(
        default=None,
        description="Define a minimal score threshold for the result. If defined, less similar results will not be returned. Score of the returned result might be higher or smaller than the threshold depending on the Distance function used. E.g. for cosine similarity only higher scores will be returned.",
    )
    using: Optional["UsingVector"] = Field(
        default=None,
        description="Define which vector to use for recommendation, if not specified - try to use default vector",
    )
    lookup_from: Optional["LookupLocation"] = Field(
        default=None,
        description="The location used to lookup vectors. If not specified - use current collection. Note: the other collection should have the same vector size as the current collection",
    )


class RecommendRequestBatch(BaseModel, extra="forbid"):
    searches: List["RecommendRequest"] = Field(..., description="")


class Record(BaseModel, extra="forbid"):
    """
    Point data
    """

    id: "ExtendedPointId" = Field(..., description="Point data")
    payload: Optional["Payload"] = Field(default=None, description="Payload - values assigned to the point")
    vector: Optional["VectorStruct"] = Field(default=None, description="Vector of the point")


class RemoteShardInfo(BaseModel, extra="forbid"):
    shard_id: int = Field(..., description="Remote shard id")
    peer_id: int = Field(..., description="Remote peer id")
    state: "ReplicaState" = Field(..., description="")


class RemoteShardTelemetry(BaseModel, extra="forbid"):
    shard_id: int = Field(..., description="")
    peer_id: Optional[int] = Field(default=None, description="")
    searches: "OperationDurationStatistics" = Field(..., description="")
    updates: "OperationDurationStatistics" = Field(..., description="")


class RenameAlias(BaseModel, extra="forbid"):
    """
    Change alias to a new one
    """

    old_alias_name: str = Field(..., description="Change alias to a new one")
    new_alias_name: str = Field(..., description="Change alias to a new one")


class RenameAliasOperation(BaseModel, extra="forbid"):
    """
    Change alias to a new one
    """

    rename_alias: "RenameAlias" = Field(..., description="Change alias to a new one")


class Replica(BaseModel, extra="forbid"):
    shard_id: int = Field(..., description="")
    peer_id: int = Field(..., description="")


class ReplicaSetTelemetry(BaseModel, extra="forbid"):
    id: int = Field(..., description="")
    local: Optional["LocalShardTelemetry"] = Field(default=None, description="")
    remote: List["RemoteShardTelemetry"] = Field(..., description="")
    replicate_states: Dict[str, "ReplicaState"] = Field(..., description="")


class ReplicaState(str, Enum):
    ACTIVE = "Active"
    DEAD = "Dead"
    PARTIAL = "Partial"
    INITIALIZING = "Initializing"
    LISTENER = "Listener"


class ReplicateShardOperation(BaseModel, extra="forbid"):
    replicate_shard: "MoveShard" = Field(..., description="")


class RequestsTelemetry(BaseModel, extra="forbid"):
    rest: "WebApiTelemetry" = Field(..., description="")
    grpc: "GrpcTelemetry" = Field(..., description="")


class RunningEnvironmentTelemetry(BaseModel, extra="forbid"):
    distribution: Optional[str] = Field(default=None, description="")
    distribution_version: Optional[str] = Field(default=None, description="")
    is_docker: bool = Field(..., description="")
    cores: Optional[int] = Field(default=None, description="")
    ram_size: Optional[int] = Field(default=None, description="")
    disk_size: Optional[int] = Field(default=None, description="")
    cpu_flags: str = Field(..., description="")


class ScalarQuantization(BaseModel, extra="forbid"):
    scalar: "ScalarQuantizationConfig" = Field(..., description="")


class ScalarQuantizationConfig(BaseModel, extra="forbid"):
    type: "ScalarType" = Field(..., description="")
    quantile: Optional[float] = Field(
        default=None,
        description="Quantile for quantization. Expected value range in [0.5, 1.0]. If not set - use the whole range of values",
    )
    always_ram: Optional[bool] = Field(
        default=None,
        description="If true - quantized vectors always will be stored in RAM, ignoring the config of main storage",
    )


class ScalarType(str, Enum):
    INT8 = "int8"


class ScoredPoint(BaseModel, extra="forbid"):
    """
    Search result
    """

    id: "ExtendedPointId" = Field(..., description="Search result")
    version: int = Field(..., description="Point version")
    score: float = Field(..., description="Points vector distance to the query vector")
    payload: Optional["Payload"] = Field(default=None, description="Payload - values assigned to the point")
    vector: Optional["VectorStruct"] = Field(default=None, description="Vector of the point")


class ScrollRequest(BaseModel, extra="forbid"):
    """
    Scroll request - paginate over all points which matches given condition
    """

    offset: Optional["ExtendedPointId"] = Field(default=None, description="Start ID to read points from.")
    limit: Optional[int] = Field(default=None, description="Page size. Default: 10")
    filter: Optional["Filter"] = Field(
        default=None, description="Look only for points which satisfies this conditions. If not provided - all points."
    )
    with_payload: Optional["WithPayloadInterface"] = Field(
        default=None, description="Select which payload to return with the response. Default: All"
    )
    with_vector: Optional["WithVector"] = Field(
        default=None, description="Scroll request - paginate over all points which matches given condition"
    )


class ScrollResult(BaseModel, extra="forbid"):
    """
    Result of the points read request
    """

    points: List["Record"] = Field(..., description="List of retrieved points")
    next_page_offset: Optional["ExtendedPointId"] = Field(
        default=None, description="Offset which should be used to retrieve a next page result"
    )


class SearchGroupsRequest(BaseModel, extra="forbid"):
    vector: "NamedVectorStruct" = Field(..., description="")
    filter: Optional["Filter"] = Field(default=None, description="Look only for points which satisfies this conditions")
    params: Optional["SearchParams"] = Field(default=None, description="Additional search params")
    with_payload: Optional["WithPayloadInterface"] = Field(
        default=None, description="Select which payload to return with the response. Default: None"
    )
    with_vector: Optional["WithVector"] = Field(
        default=None, description="Whether to return the point vector with the result?"
    )
    score_threshold: Optional[float] = Field(
        default=None,
        description="Define a minimal score threshold for the result. If defined, less similar results will not be returned. Score of the returned result might be higher or smaller than the threshold depending on the Distance function used. E.g. for cosine similarity only higher scores will be returned.",
    )
    group_by: str = Field(
        ...,
        description="Payload field to group by, must be a string or number field. If the field contains more than 1 value, all values will be used for grouping. One point can be in multiple groups.",
    )
    group_size: int = Field(..., description="Maximum amount of points to return per group")
    limit: int = Field(..., description="Maximum amount of groups to return")
    with_lookup: Optional["WithLookupInterface"] = Field(
        default=None, description="Look for points in another collection using the group ids"
    )


class SearchParams(BaseModel, extra="forbid"):
    """
    Additional parameters of the search
    """

    hnsw_ef: Optional[int] = Field(
        default=None,
        description="Params relevant to HNSW index Size of the beam in a beam-search. Larger the value - more accurate the result, more time required for search.",
    )
    exact: Optional[bool] = Field(
        default=False,
        description="Search without approximation. If set to true, search may run long but with exact results.",
    )
    quantization: Optional["QuantizationSearchParams"] = Field(default=None, description="Quantization params")


class SearchRequest(BaseModel, extra="forbid"):
    """
    Search request. Holds all conditions and parameters for the search of most similar points by vector similarity given the filtering restrictions.
    """

    vector: "NamedVectorStruct" = Field(
        ...,
        description="Search request. Holds all conditions and parameters for the search of most similar points by vector similarity given the filtering restrictions.",
    )
    filter: Optional["Filter"] = Field(default=None, description="Look only for points which satisfies this conditions")
    params: Optional["SearchParams"] = Field(default=None, description="Additional search params")
    limit: int = Field(..., description="Max number of result to return")
    offset: Optional[int] = Field(
        default=0,
        description="Offset of the first result to return. May be used to paginate results. Note: large offset values may cause performance issues.",
    )
    with_payload: Optional["WithPayloadInterface"] = Field(
        default=None, description="Select which payload to return with the response. Default: None"
    )
    with_vector: Optional["WithVector"] = Field(
        default=None, description="Whether to return the point vector with the result?"
    )
    score_threshold: Optional[float] = Field(
        default=None,
        description="Define a minimal score threshold for the result. If defined, less similar results will not be returned. Score of the returned result might be higher or smaller than the threshold depending on the Distance function used. E.g. for cosine similarity only higher scores will be returned.",
    )


class SearchRequestBatch(BaseModel, extra="forbid"):
    searches: List["SearchRequest"] = Field(..., description="")


class SegmentConfig(BaseModel, extra="forbid"):
    vector_data: Dict[str, "VectorDataConfig"] = Field(..., description="")
    payload_storage_type: "PayloadStorageType" = Field(..., description="")


class SegmentInfo(BaseModel, extra="forbid"):
    """
    Aggregated information about segment
    """

    segment_type: "SegmentType" = Field(..., description="Aggregated information about segment")
    num_vectors: int = Field(..., description="Aggregated information about segment")
    num_points: int = Field(..., description="Aggregated information about segment")
    num_deleted_vectors: int = Field(..., description="Aggregated information about segment")
    ram_usage_bytes: int = Field(..., description="Aggregated information about segment")
    disk_usage_bytes: int = Field(..., description="Aggregated information about segment")
    is_appendable: bool = Field(..., description="Aggregated information about segment")
    index_schema: Dict[str, "PayloadIndexInfo"] = Field(..., description="Aggregated information about segment")


class SegmentTelemetry(BaseModel, extra="forbid"):
    info: "SegmentInfo" = Field(..., description="")
    config: "SegmentConfig" = Field(..., description="")
    vector_index_searches: List["VectorIndexSearchesTelemetry"] = Field(..., description="")
    payload_field_indices: List["PayloadIndexTelemetry"] = Field(..., description="")


class SegmentType(str, Enum):
    PLAIN = "plain"
    INDEXED = "indexed"
    SPECIAL = "special"


class SetPayload(BaseModel, extra="forbid"):
    payload: "Payload" = Field(..., description="")
    points: Optional[List["ExtendedPointId"]] = Field(
        default=None, description="Assigns payload to each point in this list"
    )
    filter: Optional["Filter"] = Field(
        default=None, description="Assigns payload to each point that satisfy this filter condition"
    )


class ShardTransferInfo(BaseModel, extra="forbid"):
    shard_id: int = Field(..., description="")
    from_: int = Field(..., description="", alias="from")
    to: int = Field(..., description="")
    sync: bool = Field(
        ...,
        description="If `true` transfer is a synchronization of a replicas If `false` transfer is a moving of a shard from one peer to another",
    )


class SnapshotDescription(BaseModel, extra="forbid"):
    name: str = Field(..., description="")
    creation_time: Optional[str] = Field(default=None, description="")
    size: int = Field(..., description="")


class SnapshotPriority(str, Enum):
    SNAPSHOT = "snapshot"
    REPLICA = "replica"


class SnapshotRecover(BaseModel, extra="forbid"):
    location: str = Field(
        ...,
        description="Examples: - URL `http://localhost:8080/collections/my_collection/snapshots/my_snapshot` - Local path `file:///qdrant/snapshots/test_collection-2022-08-04-10-49-10.snapshot`",
    )
    priority: Optional["SnapshotPriority"] = Field(
        default=None,
        description="Defines which data should be used as a source of truth if there are other replicas in the cluster. If set to `Snapshot`, the snapshot will be used as a source of truth, and the current state will be overwritten. If set to `Replica`, the current state will be used as a source of truth, and after recovery if will be synchronized with the snapshot.",
    )


class StateRole(str, Enum):
    FOLLOWER = "Follower"
    CANDIDATE = "Candidate"
    LEADER = "Leader"
    PRECANDIDATE = "PreCandidate"


class TelemetryData(BaseModel, extra="forbid"):
    id: str = Field(..., description="")
    app: "AppBuildTelemetry" = Field(..., description="")
    collections: "CollectionsTelemetry" = Field(..., description="")
    cluster: "ClusterTelemetry" = Field(..., description="")
    requests: "RequestsTelemetry" = Field(..., description="")


class TextIndexParams(BaseModel, extra="forbid"):
    type: "TextIndexType" = Field(..., description="")
    tokenizer: Optional["TokenizerType"] = Field(default=None, description="")
    min_token_len: Optional[int] = Field(default=None, description="")
    max_token_len: Optional[int] = Field(default=None, description="")
    lowercase: Optional[bool] = Field(default=None, description="If true, lowercase all tokens. Default: true")


class TextIndexType(str, Enum):
    TEXT = "text"


class TokenizerType(str, Enum):
    PREFIX = "prefix"
    WHITESPACE = "whitespace"
    WORD = "word"
    MULTILINGUAL = "multilingual"


class UpdateCollection(BaseModel, extra="forbid"):
    """
    Operation for updating parameters of the existing collection
    """

    vectors: Optional["VectorsConfigDiff"] = Field(
        default=None,
        description="Vector data parameters to update. It is possible to provide one config for single vector mode and list of configs for multiple vectors mode.",
    )
    optimizers_config: Optional["OptimizersConfigDiff"] = Field(
        default=None,
        description="Custom params for Optimizers.  If none - it is left unchanged. This operation is blocking, it will only proceed once all current optimizations are complete",
    )
    params: Optional["CollectionParamsDiff"] = Field(
        default=None, description="Collection base params. If none - it is left unchanged."
    )
    hnsw_config: Optional["HnswConfigDiff"] = Field(
        default=None, description="HNSW parameters to update for the collection index. If none - it is left unchanged."
    )
    quantization_config: Optional["QuantizationConfigDiff"] = Field(
        default=None, description="Quantization parameters to update. If none - it is left unchanged."
    )


class UpdateResult(BaseModel, extra="forbid"):
    operation_id: int = Field(..., description="Sequential number of the operation")
    status: "UpdateStatus" = Field(..., description="")


class UpdateStatus(str, Enum):
    ACKNOWLEDGED = "acknowledged"
    COMPLETED = "completed"


class UpdateVectors(BaseModel, extra="forbid"):
    points: List["PointVectors"] = Field(..., description="Points with named vectors")


class ValuesCount(BaseModel, extra="forbid"):
    """
    Values count filter request
    """

    lt: Optional[int] = Field(default=None, description="point.key.length() &lt; values_count.lt")
    gt: Optional[int] = Field(default=None, description="point.key.length() &gt; values_count.gt")
    gte: Optional[int] = Field(default=None, description="point.key.length() &gt;= values_count.gte")
    lte: Optional[int] = Field(default=None, description="point.key.length() &lt;= values_count.lte")


class VectorDataConfig(BaseModel, extra="forbid"):
    """
    Config of single vector data storage
    """

    size: int = Field(..., description="Size/dimensionality of the vectors used")
    distance: "Distance" = Field(..., description="Config of single vector data storage")
    storage_type: "VectorStorageType" = Field(..., description="Config of single vector data storage")
    index: "Indexes" = Field(..., description="Config of single vector data storage")
    quantization_config: Optional["QuantizationConfig"] = Field(
        default=None, description="Vector specific quantization config that overrides collection config"
    )


class VectorIndexSearchesTelemetry(BaseModel, extra="forbid"):
    index_name: Optional[str] = Field(default=None, description="")
    unfiltered_plain: "OperationDurationStatistics" = Field(..., description="")
    unfiltered_hnsw: "OperationDurationStatistics" = Field(..., description="")
    filtered_plain: "OperationDurationStatistics" = Field(..., description="")
    filtered_small_cardinality: "OperationDurationStatistics" = Field(..., description="")
    filtered_large_cardinality: "OperationDurationStatistics" = Field(..., description="")
    filtered_exact: "OperationDurationStatistics" = Field(..., description="")
    unfiltered_exact: "OperationDurationStatistics" = Field(..., description="")


class VectorParams(BaseModel, extra="forbid"):
    """
    Params of single vector data storage
    """

    size: int = Field(..., description="Size of a vectors used")
    distance: "Distance" = Field(..., description="Params of single vector data storage")
    hnsw_config: Optional["HnswConfigDiff"] = Field(
        default=None,
        description="Custom params for HNSW index. If none - values from collection configuration are used.",
    )
    quantization_config: Optional["QuantizationConfig"] = Field(
        default=None,
        description="Custom params for quantization. If none - values from collection configuration are used.",
    )
    on_disk: Optional[bool] = Field(
        default=None,
        description="If true, vectors are served from disk, improving RAM usage at the cost of latency Default: false",
    )


class VectorParamsDiff(BaseModel, extra="forbid"):
    hnsw_config: Optional["HnswConfigDiff"] = Field(
        default=None, description="Update params for HNSW index. If empty object - it will be unset."
    )
    quantization_config: Optional["QuantizationConfigDiff"] = Field(
        default=None, description="Update params for quantization. If none - it is left unchanged."
    )
    on_disk: Optional[bool] = Field(
        default=None, description="If true, vectors are served from disk, improving RAM usage at the cost of latency"
    )


class VectorStorageTypeOneOf(str, Enum):
    MEMORY = "Memory"


class VectorStorageTypeOneOf1(str, Enum):
    MMAP = "Mmap"


class VectorStorageTypeOneOf2(str, Enum):
    CHUNKEDMMAP = "ChunkedMmap"


class WalConfig(BaseModel, extra="forbid"):
    wal_capacity_mb: int = Field(..., description="Size of a single WAL segment in MB")
    wal_segments_ahead: int = Field(..., description="Number of WAL segments to create ahead of actually used ones")


class WalConfigDiff(BaseModel, extra="forbid"):
    wal_capacity_mb: Optional[int] = Field(default=None, description="Size of a single WAL segment in MB")
    wal_segments_ahead: Optional[int] = Field(
        default=None, description="Number of WAL segments to create ahead of actually used ones"
    )


class WebApiTelemetry(BaseModel, extra="forbid"):
    responses: Dict[str, Dict[str, "OperationDurationStatistics"]] = Field(..., description="")


class WithLookup(BaseModel, extra="forbid"):
    collection: str = Field(..., description="Name of the collection to use for points lookup")
    with_payload: Optional["WithPayloadInterface"] = Field(
        default=None, description="Options for specifying which payload to include (or not)"
    )
    with_vectors: Optional["WithVector"] = Field(
        default=None, description="Options for specifying which vectors to include (or not)"
    )


class WriteOrdering(str, Enum):
    WEAK = "weak"
    MEDIUM = "medium"
    STRONG = "strong"


AliasOperations = Union[
    CreateAliasOperation,
    DeleteAliasOperation,
    RenameAliasOperation,
]
AnyVariants = Union[
    List[StrictInt],
    List[StrictStr],
]
BatchVectorStruct = Union[
    List[List[StrictFloat]],
    Dict[StrictStr, List[List[StrictFloat]]],
]
ClusterOperations = Union[
    MoveShardOperation,
    ReplicateShardOperation,
    AbortTransferOperation,
    DropReplicaOperation,
]
ClusterStatus = Union[
    ClusterStatusOneOf,
    ClusterStatusOneOf1,
]
CollectionTelemetryEnum = Union[
    CollectionTelemetry,
    CollectionsAggregatedTelemetry,
]
Condition = Union[
    FieldCondition,
    IsEmptyCondition,
    IsNullCondition,
    HasIdCondition,
    NestedCondition,
    Filter,
]
ConsensusThreadStatus = Union[
    ConsensusThreadStatusOneOf,
    ConsensusThreadStatusOneOf1,
    ConsensusThreadStatusOneOf2,
]
ExtendedPointId = Union[
    StrictInt,
    StrictStr,
]
GroupId = Union[
    StrictInt,
    StrictStr,
]
Indexes = Union[
    IndexesOneOf,
    IndexesOneOf1,
]
Match = Union[
    MatchValue,
    MatchText,
    MatchAny,
    MatchExcept,
]
NamedVectorStruct = Union[
    NamedVector,
    List[StrictFloat],
]
OptimizersStatus = Union[
    OptimizersStatusOneOf,
    OptimizersStatusOneOf1,
]
PayloadSchemaParams = Union[
    TextIndexParams,
]
PayloadSelector = Union[
    PayloadSelectorInclude,
    PayloadSelectorExclude,
]
PayloadStorageType = Union[
    PayloadStorageTypeOneOf,
    PayloadStorageTypeOneOf1,
]
PointInsertOperations = Union[
    PointsBatch,
    PointsList,
]
PointsSelector = Union[
    PointIdsList,
    FilterSelector,
]
QuantizationConfig = Union[
    ScalarQuantization,
    ProductQuantization,
]
QuantizationConfigDiff = Union[
    ScalarQuantization,
    ProductQuantization,
    Disabled,
]
ReadConsistency = Union[
    ReadConsistencyType,
    StrictInt,
]
UsingVector = Union[
    StrictStr,
]
ValueVariants = Union[
    StrictBool,
    StrictInt,
    StrictStr,
]
VectorStorageType = Union[
    VectorStorageTypeOneOf,
    VectorStorageTypeOneOf1,
    VectorStorageTypeOneOf2,
]
VectorStruct = Union[
    List[StrictFloat],
    Dict[StrictStr, List[StrictFloat]],
]
VectorsConfig = Union[
    VectorParams,
    Dict[StrictStr, VectorParams],
]
WithLookupInterface = Union[
    WithLookup,
    StrictStr,
]
WithVector = Union[
    List[StrictStr],
    StrictBool,
]
PayloadFieldSchema = Union[
    PayloadSchemaType,
    PayloadSchemaParams,
]
WithPayloadInterface = Union[
    PayloadSelector,
    List[StrictStr],
    StrictBool,
]
