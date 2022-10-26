from typing import Union, List

from qdrant_client import grpc as grpc
from qdrant_client.http import models as rest

Filter = Union[rest.Filter, grpc.Filter]
SearchParams = Union[rest.SearchParams, grpc.SearchParams]
PayloadSelector = Union[rest.PayloadSelector, grpc.WithPayloadSelector]
Distance = Union[rest.Distance, int]  # type(grpc.Distance) == int
HnswConfigDiff = Union[rest.HnswConfigDiff, grpc.HnswConfigDiff]
OptimizersConfigDiff = Union[rest.OptimizersConfigDiff, grpc.OptimizersConfigDiff]
CollectionParamsDiff = Union[rest.CollectionParamsDiff, grpc.CollectionParamsDiff]
WalConfigDiff = Union[rest.WalConfigDiff, grpc.WalConfigDiff]
PointId = Union[int, str, grpc.PointId]
PayloadSchemaType = Union[rest.PayloadSchemaType, rest.PayloadSchemaParams, int]  # type(grpc.PayloadSchemaType) == int
Points = Union[rest.Batch, List[Union[rest.PointStruct, grpc.PointStruct]]]
PointsSelector = Union[rest.PointsSelector, grpc.PointsSelector]
AliasOperations = Union[
    rest.CreateAliasOperation,
    rest.RenameAliasOperation,
    rest.DeleteAliasOperation,
    grpc.AliasOperations
]
Payload = rest.Payload

ScoredPoint = rest.ScoredPoint
UpdateResult = rest.UpdateResult
Record = rest.Record
CollectionsResponse = rest.CollectionsResponse
CollectionInfo = rest.CollectionInfo
CountResult = rest.CountResult
SnapshotDescription = rest.SnapshotDescription
NamedVector = rest.NamedVector
VectorParams = rest.VectorParams
LocksOption = rest.LocksOption

SearchRequest = Union[rest.SearchRequest, grpc.SearchPoints]
RecommendRequest = Union[rest.RecommendRequest, grpc.RecommendPoints]
