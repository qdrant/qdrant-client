import sys

import numpy as np

if sys.version_info >= (3, 10):
    from typing import Any, TypeAlias
else:
    from typing_extensions import TypeAlias

from typing import List, Union

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
QuantizationConfig = Union[rest.QuantizationConfig, grpc.QuantizationConfig]
PointId = Union[int, str, grpc.PointId]
PayloadSchemaType = Union[
    rest.PayloadSchemaType, rest.PayloadSchemaParams, int, grpc.PayloadIndexParams
]  # type(grpc.PayloadSchemaType) == int
Points = Union[rest.Batch, List[Union[rest.PointStruct, grpc.PointStruct]]]
PointsSelector = Union[
    List[PointId], rest.Filter, grpc.Filter, rest.PointsSelector, grpc.PointsSelector
]
LookupLocation = Union[rest.LookupLocation, grpc.LookupLocation]

AliasOperations = Union[
    rest.CreateAliasOperation,
    rest.RenameAliasOperation,
    rest.DeleteAliasOperation,
    grpc.AliasOperations,
]
Payload: TypeAlias = rest.Payload

ScoredPoint: TypeAlias = rest.ScoredPoint
UpdateResult: TypeAlias = rest.UpdateResult
Record: TypeAlias = rest.Record
CollectionsResponse: TypeAlias = rest.CollectionsResponse
CollectionInfo: TypeAlias = rest.CollectionInfo
CountResult: TypeAlias = rest.CountResult
SnapshotDescription: TypeAlias = rest.SnapshotDescription
NamedVector: TypeAlias = rest.NamedVector
PointVectors: TypeAlias = rest.PointVectors
VectorStruct: TypeAlias = rest.VectorStruct
VectorParams: TypeAlias = rest.VectorParams
LocksOption: TypeAlias = rest.LocksOption
SnapshotPriority: TypeAlias = rest.SnapshotPriority
CollectionsAliasesResponse: TypeAlias = rest.CollectionsAliasesResponse
InitFrom: TypeAlias = rest.InitFrom

SearchRequest = Union[rest.SearchRequest, grpc.SearchPoints]
RecommendRequest = Union[rest.RecommendRequest, grpc.RecommendPoints]

ReadConsistency: TypeAlias = rest.ReadConsistency
WriteOrdering: TypeAlias = rest.WriteOrdering
WithLookupInterface: TypeAlias = rest.WithLookupInterface

GroupsResult: TypeAlias = rest.GroupsResult

# we can't use `nptyping` package due to numpy/python-version incompatibilities
# thus we need to define precise type annotations while we support python3.7
_np_numeric = Union[
    np.bool_,  # pylance can't handle np.bool8 alias
    np.int8,
    np.int16,
    np.int32,
    np.int64,
    np.uint8,
    np.uint16,
    np.uint32,
    np.uint64,
    np.intp,
    np.uintp,
    np.float16,
    np.float32,
    np.float64,
    np.longdouble,  # np.float96 and np.float128 are platform dependant aliases for longdouble
]


if sys.version_info >= (3, 8):
    # typing is included into numpy since 1.20
    # NDArray is included since 1.21
    # pyproject.toml is configured to install numpy>=1.21 in case of python>=3.8
    # thus we don't need an additional check for numpy version
    import numpy.typing as npt

    NumpyArray: TypeAlias = npt.NDArray[_np_numeric]
else:
    NumpyArray: TypeAlias = np.ndarray
