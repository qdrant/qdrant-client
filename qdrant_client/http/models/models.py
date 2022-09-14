from datetime import datetime
from enum import Enum
from typing import List, Union

try:
    pass
except ImportError:
    # Python 3.7 backport
    pass

from pydantic import BaseModel, Field
from pydantic.types import StrictBool, StrictFloat, StrictInt, StrictStr

AbortTransferOperation = dict
AppBuildTelemetry = dict
Batch = dict
ChangeAliasesOperation = dict
ClusterConfigTelemetry = dict
ClusterStatus200Response = dict
ClusterStatusOneOf = dict
ClusterStatusOneOf1 = dict
CollectionClusterInfo = dict
CollectionClusterInfo200Response = dict
CollectionConfig = dict
CollectionDescription = dict
CollectionInfo = dict
CollectionParams = dict


class CollectionStatus(str, Enum):
    GREEN = "green"
    YELLOW = "yellow"
    RED = "red"


CollectionTelemetry = dict
CollectionsResponse = dict
ConfigsTelemetry = dict
ConsensusConfigTelemetry = dict
ConsensusThreadStatusOneOf = dict
ConsensusThreadStatusOneOf1 = dict
ConsensusThreadStatusOneOf2 = dict
CountPoints200Response = dict
CountRequest = dict
CountResult = dict
CreateAlias = dict
CreateAliasOperation = dict
CreateCollection = dict
CreateFieldIndex = dict
CreateFieldIndex200Response = dict
CreateSnapshot200Response = dict
DeleteAlias = dict
DeleteAliasOperation = dict
DeletePayload = dict


class Distance(str, Enum):
    COSINE = "Cosine"
    EUCLID = "Euclid"
    DOT = "Dot"


Duration = dict
ErrorResponse = dict
ErrorResponseStatus = dict
FieldCondition = dict
Filter = dict
FilterSelector = dict
GeoBoundingBox = dict
GeoPoint = dict
GeoRadius = dict
GetCollection200Response = dict
GetCollections200Response = dict
GetPoint200Response = dict
GetPoints200Response = dict
HasIdCondition = dict
HnswConfig = dict
HnswConfigDiff = dict
IndexesOneOf = dict
IndexesOneOf1 = dict
IsEmptyCondition = dict
ListSnapshots200Response = dict
LocalShardInfo = dict
MatchText = dict
MatchValue = dict
MoveShard = dict
MoveShardOperation = dict
NamedVector = dict
OptimizerTelemetryOneOf = dict
OptimizerTelemetryOneOf1 = dict
OptimizerTelemetryOneOf2 = dict
OptimizerTelemetryOneOfIndexing = dict
OptimizersConfig = dict
OptimizersConfigDiff = dict


class OptimizersStatusOneOf(str, Enum):
    OK = "ok"


OptimizersStatusOneOf1 = dict
P2pConfigTelemetry = dict
Payload = dict
PayloadField = dict
PayloadIndexInfo = dict
PayloadIndexTelemetry = dict
PayloadSchemaParamsOneOf = dict


class PayloadSchemaType(str, Enum):
    KEYWORD = "keyword"
    INTEGER = "integer"
    FLOAT = "float"
    GEO = "geo"
    TEXT = "text"


PayloadSelectorExclude = dict
PayloadSelectorInclude = dict
PayloadStorageTypeOneOf = dict
PayloadStorageTypeOneOf1 = dict
PeerInfo = dict
PointIdsList = dict
PointRequest = dict
PointStruct = dict


class PointsBatch(BaseModel):
    batch: "Batch" = Field(..., description="")


PointsList = dict
RaftInfo = dict
Range = dict
RecommendRequest = dict
RecommendRequestBatch = dict
Record = dict
RemoteShardInfo = dict
RemovePeer200Response = dict
RenameAlias = dict
RenameAliasOperation = dict
RunningEnvironmentTelemetry = dict
ScoredPoint = dict
ScrollPoints200Response = dict
ScrollRequest = dict
ScrollResult = dict
SearchBatchPoints200Response = dict
SearchParams = dict
SearchPoints200Response = dict
SearchRequest = dict
SearchRequestBatch = dict
SegmentConfig = dict
SegmentInfo = dict
SegmentTelemetry = dict


class SegmentType(str, Enum):
    PLAIN = "plain"
    INDEXED = "indexed"
    SPECIAL = "special"


ServiceConfigTelemetry = dict
SetPayload = dict
ShardTelemetryOneOf = dict
ShardTelemetryOneOf1 = dict
ShardTelemetryOneOf1Local = dict
ShardTelemetryOneOf2 = dict
ShardTelemetryOneOf3 = dict
ShardTelemetryOneOfRemote = dict
ShardTransferInfo = dict
SnapshotDescription = dict


class StateRole(str, Enum):
    FOLLOWER = "Follower"
    CANDIDATE = "Candidate"
    LEADER = "Leader"
    PRECANDIDATE = "PreCandidate"


StorageTypeOneOf = dict
StorageTypeOneOf1 = dict
Telemetry200Response = dict
TelemetryData = dict
TelemetryOperationStatistics = dict


class TokenizerType(str, Enum):
    PREFIX = "prefix"
    WHITESPACE = "whitespace"
    WORD = "word"


UpdateCollection = dict
UpdateResult = dict


class UpdateStatus(str, Enum):
    ACKNOWLEDGED = "acknowledged"
    COMPLETED = "completed"


ValuesCount = dict
VectorDataConfig = dict
VectorIndexTelemetry = dict
VectorParams = dict
WalConfig = dict
WalConfigDiff = dict
WebApiTelemetry = dict
AliasOperations = Union[
    CreateAliasOperation,
    DeleteAliasOperation,
    RenameAliasOperation,
]
BatchPayloadsInner = Union[
    Payload,
    bool,
    date,
    datetime,
    dict,
    float,
    int,
    list,
    str,
    none_type,
]
BatchVectorStruct = Union[
    List[[float]],
    {str: ([[float]],)},
]
ClusterOperations = Union[
    MoveShardOperation,
    AbortTransferOperation,
]
ClusterStatus = Union[
    ClusterStatusOneOf,
    ClusterStatusOneOf1,
]
CollectionParamsDistance = Union[
    Distance,
    bool,
    date,
    datetime,
    dict,
    float,
    int,
    list,
    str,
    none_type,
]
Condition = Union[
    FieldCondition,
    IsEmptyCondition,
    HasIdCondition,
    Filter,
]
ConsensusThreadStatus = Union[
    ConsensusThreadStatusOneOf,
    ConsensusThreadStatusOneOf1,
    ConsensusThreadStatusOneOf2,
]
CreateCollectionDistance = Union[
    Distance,
    bool,
    date,
    datetime,
    dict,
    float,
    int,
    list,
    str,
    none_type,
]
CreateCollectionHnswConfig = Union[
    HnswConfigDiff,
    bool,
    date,
    datetime,
    dict,
    float,
    int,
    list,
    str,
    none_type,
]
CreateCollectionOptimizersConfig = Union[
    OptimizersConfigDiff,
    bool,
    date,
    datetime,
    dict,
    float,
    int,
    list,
    str,
    none_type,
]
CreateCollectionWalConfig = Union[
    WalConfigDiff,
    bool,
    date,
    datetime,
    dict,
    float,
    int,
    list,
    str,
    none_type,
]
ExtendedPointId = Union[
    StrictInt,
    StrictStr,
]
FieldConditionGeoBoundingBox = Union[
    GeoBoundingBox,
    bool,
    date,
    datetime,
    dict,
    float,
    int,
    list,
    str,
    none_type,
]
FieldConditionGeoRadius = Union[
    GeoRadius,
    bool,
    date,
    datetime,
    dict,
    float,
    int,
    list,
    str,
    none_type,
]
FieldConditionRange = Union[
    Range,
    bool,
    date,
    datetime,
    dict,
    float,
    int,
    list,
    str,
    none_type,
]
FieldConditionValuesCount = Union[
    ValuesCount,
    bool,
    date,
    datetime,
    dict,
    float,
    int,
    list,
    str,
    none_type,
]
Indexes = Union[
    IndexesOneOf,
    IndexesOneOf1,
]
Match = Union[
    MatchValue,
    MatchText,
]
NamedVectorStruct = Union[
    NamedVector,
    List[StrictFloat],
]
OptimizerTelemetry = Union[
    OptimizerTelemetryOneOf,
    OptimizerTelemetryOneOf1,
    OptimizerTelemetryOneOf2,
]
OptimizersStatus = Union[
    OptimizersStatusOneOf,
    OptimizersStatusOneOf1,
]
PayloadSchemaParams = Union[
    PayloadSchemaParamsOneOf,
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
PointStructPayload = Union[
    Payload,
    bool,
    date,
    datetime,
    dict,
    float,
    int,
    list,
    str,
    none_type,
]
PointsSelector = Union[
    PointIdsList,
    FilterSelector,
]
RaftInfoRole = Union[
    StateRole,
    bool,
    date,
    datetime,
    dict,
    float,
    int,
    list,
    str,
    none_type,
]
RecordPayload = Union[
    Payload,
    bool,
    date,
    datetime,
    dict,
    float,
    int,
    list,
    str,
    none_type,
]
ScrollRequestFilter = Union[
    Filter,
    bool,
    date,
    datetime,
    dict,
    float,
    int,
    list,
    str,
    none_type,
]
SearchRequestFilter = Union[
    Filter,
    bool,
    date,
    datetime,
    dict,
    float,
    int,
    list,
    str,
    none_type,
]
SearchRequestParams = Union[
    SearchParams,
    bool,
    date,
    datetime,
    dict,
    float,
    int,
    list,
    str,
    none_type,
]
ShardTelemetry = Union[
    ShardTelemetryOneOf,
    ShardTelemetryOneOf1,
    ShardTelemetryOneOf2,
    ShardTelemetryOneOf3,
]
StorageType = Union[
    StorageTypeOneOf,
    StorageTypeOneOf1,
]
UpdateCollectionOptimizersConfig = Union[
    OptimizersConfigDiff,
    bool,
    date,
    datetime,
    dict,
    float,
    int,
    list,
    str,
    none_type,
]
ValueVariants = Union[
    StrictBool,
    StrictInt,
    StrictStr,
]
VectorStruct = Union[
    List[StrictFloat],
    {str: ([float],)},
]
VectorsConfig = Union[
    VectorParams,
    {str: (VectorParams,)},
]
WithVector = Union[
    List[StrictStr],
    StrictBool,
]
CollectionParamsVectors = Union[
    VectorsConfig,
    bool,
    date,
    datetime,
    dict,
    float,
    int,
    list,
    str,
    none_type,
]
CreateCollectionVectors = Union[
    VectorsConfig,
    bool,
    date,
    datetime,
    dict,
    float,
    int,
    list,
    str,
    none_type,
]
FieldConditionMatch = Union[
    Match,
    bool,
    date,
    datetime,
    dict,
    float,
    int,
    list,
    str,
    none_type,
]
PayloadFieldSchema = Union[
    PayloadSchemaType,
    PayloadSchemaParams,
]
PayloadIndexInfoParams = Union[
    PayloadSchemaParams,
    bool,
    date,
    datetime,
    dict,
    float,
    int,
    list,
    str,
    none_type,
]
RecordVector = Union[
    VectorStruct,
    bool,
    date,
    datetime,
    dict,
    float,
    int,
    list,
    str,
    none_type,
]
ScrollRequestOffset = Union[
    ExtendedPointId,
    bool,
    date,
    datetime,
    dict,
    float,
    int,
    list,
    str,
    none_type,
]
ScrollResultNextPageOffset = Union[
    ExtendedPointId,
    bool,
    date,
    datetime,
    dict,
    float,
    int,
    list,
    str,
    none_type,
]
WithPayloadInterface = Union[
    PayloadSelector,
    List[StrictStr],
    StrictBool,
]
CreateFieldIndexFieldSchema = Union[
    PayloadFieldSchema,
    bool,
    date,
    datetime,
    dict,
    float,
    int,
    list,
    str,
    none_type,
]
PointRequestWithPayload = Union[
    WithPayloadInterface,
    bool,
    date,
    datetime,
    dict,
    float,
    int,
    list,
    str,
    none_type,
]
SearchRequestWithPayload = Union[
    WithPayloadInterface,
    bool,
    date,
    datetime,
    dict,
    float,
    int,
    list,
    str,
    none_type,
]
