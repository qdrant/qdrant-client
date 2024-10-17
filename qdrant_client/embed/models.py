from typing import Union, List, Dict

from pydantic import StrictFloat, StrictStr

from qdrant_client.grpc import SparseVector
from qdrant_client.http.models import ExtendedPointId
from qdrant_client.models import Document  # type: ignore[attr-defined]


NumericVector = Union[
    List[StrictFloat],
    SparseVector,
    List[List[StrictFloat]],
]
NumericVectorInput = Union[
    List[StrictFloat],
    SparseVector,
    List[List[StrictFloat]],
    ExtendedPointId,
]
NumericVectorStruct = Union[
    List[StrictFloat],
    List[List[StrictFloat]],
    Dict[StrictStr, NumericVector],
]

__all__ = ["Document", "NumericVector", "NumericVectorInput", "NumericVectorStruct"]
