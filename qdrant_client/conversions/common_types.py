from typing import Union, Type, List, Optional

import betterproto
from pydantic import BaseModel

from qdrant_client.http import models as rest
from qdrant_client import grpc

Filter = Union[rest.Filter, grpc.Filter]
SearchParams = Union[rest.SearchParams, grpc.SearchParams]
PayloadSelector = Union[rest.PayloadSelector, grpc.WithPayloadSelector]
Distance = Union[rest.Distance, grpc.Distance]
HnswConfigDiff = Union[rest.HnswConfigDiff, grpc.HnswConfigDiff]
OptimizersConfigDiff = Union[rest.OptimizersConfigDiff, grpc.OptimizersConfigDiff]
WalConfigDiff = Union[rest.WalConfigDiff, grpc.WalConfigDiff]
PointId = Union[rest.ExtendedPointId, grpc.PointId]
PayloadSchemaType = Union[rest.PayloadSchemaType, grpc.PayloadSchemaType]
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
