from typing import Union, List

from pydantic import StrictFloat

from qdrant_client.grpc import SparseVector
from qdrant_client.http.models import ExtendedPointId
from qdrant_client.models import Document  # type: ignore[attr-defined]


NumericVectorInput = Union[
    List[StrictFloat],
    SparseVector,
    List[List[StrictFloat]],
    ExtendedPointId,
]
__all__ = ["Document", "NumericVectorInput"]
