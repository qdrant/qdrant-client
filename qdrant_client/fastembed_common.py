from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel


class QueryResponse(BaseModel, extra="forbid"):  # type: ignore
    id: Union[str, int]
    embedding: Optional[List[float]]
    metadata: Dict[str, Any]
    document: str
    score: float
