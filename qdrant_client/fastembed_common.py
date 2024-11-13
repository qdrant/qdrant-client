from typing import Any, Optional, Union

from pydantic import BaseModel, Field

from qdrant_client.conversions.common_types import SparseVector


class QueryResponse(BaseModel, extra="forbid"):  # type: ignore
    id: Union[str, int]
    embedding: Optional[list[float]]
    sparse_embedding: Optional[SparseVector] = Field(default=None)
    metadata: dict[str, Any]
    document: str
    score: float
