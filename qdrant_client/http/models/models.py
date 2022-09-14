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
ClusterStatusOneOf = dict
ClusterStatusOneOf1 = dict
CollectionClusterInfo = dict
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
CountRequest = dict
CountResult = dict
CreateAlias = dict
CreateAliasOperation = dict
CreateCollection = dict
CreateFieldIndex = dict
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
HasIdCondition = dict
HnswConfig = dict
HnswConfigDiff = dict
IndexesOneOf = dict
IndexesOneOf1 = dict
InlineResponse200 = dict
InlineResponse2001 = dict
InlineResponse20010 = dict
InlineResponse20011 = dict
InlineResponse20012 = dict
InlineResponse20013 = dict
InlineResponse20014 = dict
InlineResponse2002 = dict
InlineResponse2003 = dict
InlineResponse2004 = dict
InlineResponse2005 = dict
InlineResponse2006 = dict
InlineResponse2007 = dict
InlineResponse2008 = dict
InlineResponse2009 = dict
IsEmptyCondition = dict
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
RenameAlias = dict
RenameAliasOperation = dict
RunningEnvironmentTelemetry = dict
ScoredPoint = dict
ScrollRequest = dict
ScrollResult = dict
SearchParams = dict
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
ExtendedPointId = Union[
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
PointsSelector = Union[
    PointIdsList,
    FilterSelector,
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
PayloadFieldSchema = Union[
    PayloadSchemaType,
    PayloadSchemaParams,
]
WithPayloadInterface = Union[
    PayloadSelector,
    List[StrictStr],
    StrictBool,
]
