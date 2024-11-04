from typing import Union, List, Dict

from pydantic import StrictFloat, StrictStr

from qdrant_client.http.models import ExtendedPointId, SparseVector


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

__all__ = ["NumericVector", "NumericVectorInput", "NumericVectorStruct"]
