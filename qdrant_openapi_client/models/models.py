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


class CollectionDescription(BaseModel):
    name: str = Field(..., description="")


class CollectionInfo(BaseModel):
    """
    Current statistics and configuration of the collection.
    """

    config: "SegmentConfig" = Field(..., description="Current statistics and configuration of the collection.")
    disk_data_size: int = Field(..., description="Disk space, used by collection")
    ram_data_size: int = Field(..., description="RAM used by collection")
    segments_count: int = Field(..., description="Number of segments in collection")
    vectors_count: int = Field(..., description="Number of vectors in collection")


class CollectionsResponse(BaseModel):
    collections: List["CollectionDescription"] = Field(..., description="")


class ConditionAnyOf(BaseModel):
    """
    Nested filter
    """

    filter: "Filter" = Field(..., description="Nested filter")


class ConditionAnyOf1(BaseModel):
    """
    Check if point has field with a given value
    """

    match: "Match" = Field(..., description="Check if point has field with a given value")


class ConditionAnyOf2(BaseModel):
    """
    Check if points value lies in a given range
    """

    range: "Range" = Field(..., description="Check if points value lies in a given range")


class ConditionAnyOf3(BaseModel):
    """
    Check if points geo location lies in a given area
    """

    geo_bounding_box: "GeoBoundingBox" = Field(..., description="Check if points geo location lies in a given area")


class ConditionAnyOf4(BaseModel):
    """
    Check if geo point is within a given radius
    """

    geo_radius: "GeoRadius" = Field(..., description="Check if geo point is within a given radius")


class ConditionAnyOf5(BaseModel):
    """
    Check if points id is in a given set
    """

    has_id: List[int] = Field(..., description="Check if points id is in a given set")


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


class Filter(BaseModel):
    must: Optional[List["Condition"]] = Field(None, description="All conditions must match")
    must_not: Optional[List["Condition"]] = Field(None, description="All conditions must NOT match")
    should: Optional[List["Condition"]] = Field(None, description="At least one of thous conditions should match")


class GeoBoundingBox(BaseModel):
    bottom_right: "GeoPoint" = Field(..., description="")
    key: str = Field(..., description="Name of the field to match with")
    top_left: "GeoPoint" = Field(..., description="")


class GeoPoint(BaseModel):
    lat: float = Field(..., description="")
    lon: float = Field(..., description="")


class GeoRadius(BaseModel):
    center: "GeoPoint" = Field(..., description="")
    key: str = Field(..., description="Name of the field to match with")
    radius: float = Field(..., description="Radius of the area in meters")


class IndexesAnyOf(BaseModel):
    """
    Do not use any index, scan whole vector collection during search. Guarantee 100% precision, but may be time consuming on large collections.
    """

    options: Any = Field(
        ...,
        description="Do not use any index, scan whole vector collection during search. Guarantee 100% precision, but may be time consuming on large collections.",
    )
    type: Literal["plain",] = Field(
        ...,
        description="Do not use any index, scan whole vector collection during search. Guarantee 100% precision, but may be time consuming on large collections.",
    )


class IndexesAnyOf1(BaseModel):
    """
    Use filterable HNSW index for approximate search. Is very fast even on a very huge collections, but require additional space to store index and additional time to build it.
    """

    options: "IndexesAnyOf1Options" = Field(
        ...,
        description="Use filterable HNSW index for approximate search. Is very fast even on a very huge collections, but require additional space to store index and additional time to build it.",
    )
    type: Literal["hnsw",] = Field(
        ...,
        description="Use filterable HNSW index for approximate search. Is very fast even on a very huge collections, but require additional space to store index and additional time to build it.",
    )


class IndexesAnyOf1Options(BaseModel):
    ef_construct: int = Field(
        ...,
        description="Number of neighbours to consider during the index building. Larger the value - more accurate the search, more time required to build index.",
    )
    m: int = Field(
        ...,
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
    result: Optional[List["ScoredPoint"]] = Field(None, description="")


class Match(BaseModel):
    integer: Optional[int] = Field(None, description="Integer value to match")
    key: str = Field(..., description="Name of the field to match with")
    keyword: Optional[str] = Field(None, description="Keyword value to match")


class PayloadInterfaceAnyOf(BaseModel):
    type: Literal[
        "keyword",
    ] = Field(..., description="")
    value: "PayloadVariantForString" = Field(..., description="")


class PayloadInterfaceAnyOf1(BaseModel):
    type: Literal[
        "integer",
    ] = Field(..., description="")
    value: "PayloadVariantForInt64" = Field(..., description="")


class PayloadInterfaceAnyOf2(BaseModel):
    type: Literal[
        "float",
    ] = Field(..., description="")
    value: "PayloadVariantForDouble" = Field(..., description="")


class PayloadInterfaceAnyOf3(BaseModel):
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


class PointInsertOpsAnyOf(BaseModel):
    """
    Inset points from a batch.
    """

    batch: "PointInsertOpsAnyOfBatch" = Field(..., description="Inset points from a batch.")


class PointInsertOpsAnyOf1(BaseModel):
    """
    Insert points from a list
    """

    points: List["PointStruct"] = Field(..., description="Insert points from a list")


class PointInsertOpsAnyOfBatch(BaseModel):
    ids: List[int] = Field(..., description="")
    payloads: Optional[List[Dict[str, "PayloadInterface"]]] = Field(None, description="")
    vectors: List[List[float]] = Field(..., description="")


class PointOpsAnyOf(BaseModel):
    """
    Insert or update points
    """

    upsert_points: "PointInsertOps" = Field(..., description="Insert or update points")


class PointOpsAnyOf1(BaseModel):
    """
    Delete point if exists
    """

    delete_points: "PointOpsAnyOf1DeletePoints" = Field(..., description="Delete point if exists")


class PointOpsAnyOf1DeletePoints(BaseModel):
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
    key: str = Field(..., description="Name of the field to match with")
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


class SearchParamsAnyOf(BaseModel):
    """
    Params relevant to HNSW index
    """

    hnsw: "SearchParamsAnyOfHnsw" = Field(..., description="Params relevant to HNSW index")


class SearchParamsAnyOfHnsw(BaseModel):
    ef: int = Field(
        ...,
        description="Size of the beam in a beam-search. Larger the value - more accurate the result, more time required for search.",
    )


class SearchRequest(BaseModel):
    """
    Search request
    """

    filter: Optional["Filter"] = Field(None, description="Look only for points which satisfies this conditions")
    params: Optional["SearchParams"] = Field(None, description="Additional search params")
    top: int = Field(..., description="Max number of result to return")
    vector: List[float] = Field(..., description="Look for vectors closest to this")


class SegmentConfig(BaseModel):
    distance: "Distance" = Field(..., description="")
    index: "Indexes" = Field(..., description="")
    storage_type: "StorageType" = Field(..., description="")
    vector_size: int = Field(..., description="Size of a vectors used")


class StorageOpsAnyOf(BaseModel):
    """
    Create new collection and (optionally) specify index params
    """

    create_collection: "StorageOpsAnyOfCreateCollection" = Field(
        ..., description="Create new collection and (optionally) specify index params"
    )


class StorageOpsAnyOf1(BaseModel):
    """
    Delete collection with given name
    """

    delete_collection: str = Field(..., description="Delete collection with given name")


class StorageOpsAnyOf2(BaseModel):
    """
    Perform changes of collection aliases Alias changes are atomic, meaning that no collection modifications can happen between alias operations
    """

    change_aliases: "StorageOpsAnyOf2ChangeAliases" = Field(
        ...,
        description="Perform changes of collection aliases Alias changes are atomic, meaning that no collection modifications can happen between alias operations",
    )


class StorageOpsAnyOf2ChangeAliases(BaseModel):
    actions: List["AliasOperations"] = Field(..., description="")


class StorageOpsAnyOfCreateCollection(BaseModel):
    distance: "Distance" = Field(..., description="")
    index: Optional["Indexes"] = Field(None, description="")
    name: str = Field(..., description="")
    vector_size: int = Field(..., description="")


class StorageTypeAnyOf(BaseModel):
    """
    Store vectors in memory and use persistence storage only if vectors are changed
    """

    type: Literal[
        "in_memory",
    ] = Field(..., description="Store vectors in memory and use persistence storage only if vectors are changed")


class StorageTypeAnyOf1(BaseModel):
    """
    Use memmap to store vectors, a little slower than `InMemory`, but requires little RAM
    """

    type: Literal[
        "mmap",
    ] = Field(..., description="Use memmap to store vectors, a little slower than `InMemory`, but requires little RAM")


class UpdateResult(BaseModel):
    operation_id: int = Field(..., description="Sequential number of the operation")
    status: "UpdateStatus" = Field(..., description="")


class UpdateStatus(str, Enum):
    ACKNOWLEDGED = "acknowledged"
    COMPLETED = "completed"


AliasOperations = Union[
    AliasOperationsAnyOf,
    AliasOperationsAnyOf1,
    AliasOperationsAnyOf2,
]
Condition = Union[
    ConditionAnyOf,
    ConditionAnyOf1,
    ConditionAnyOf2,
    ConditionAnyOf3,
    ConditionAnyOf4,
    ConditionAnyOf5,
]
Indexes = Union[
    IndexesAnyOf,
    IndexesAnyOf1,
]
PayloadInterface = Union[
    PayloadInterfaceAnyOf,
    PayloadInterfaceAnyOf1,
    PayloadInterfaceAnyOf2,
    PayloadInterfaceAnyOf3,
]
PayloadOps = Union[
    PayloadOpsAnyOf,
    PayloadOpsAnyOf1,
    PayloadOpsAnyOf2,
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
PointInsertOps = Union[
    PointInsertOpsAnyOf,
    PointInsertOpsAnyOf1,
]
PointOps = Union[
    PointOpsAnyOf,
    PointOpsAnyOf1,
]
SearchParams = Union[
    SearchParamsAnyOf,
]
StorageOps = Union[
    StorageOpsAnyOf,
    StorageOpsAnyOf1,
    StorageOpsAnyOf2,
]
StorageType = Union[
    StorageTypeAnyOf,
    StorageTypeAnyOf1,
]
CollectionUpdateOperations = Union[
    PayloadOps,
    PointOps,
]
