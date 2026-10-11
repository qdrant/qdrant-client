"""Numpy vectors in PointStruct, PointVectors and Batch are converted to lists before validation.

pydantic validates a numpy array given for a `List[StrictFloat]` field element by element, ~25
times slower than a list of the same floats. The generator (pydantic_openapi_v3) annotates every
VectorStruct and BatchVectorStruct field with `NumpyVectorsToLists`.
"""

from typing import Any

import numpy as np

from qdrant_client._pydantic_compat import PYDANTIC_V2


def _to_lists(vector: Any) -> Any:
    """A numpy vector, or a list of them (multivector rows, a batch), as lists; else as is"""
    if isinstance(vector, np.ndarray):
        return vector.tolist()
    if isinstance(vector, list) and vector and isinstance(vector[0], np.ndarray):
        return [item.tolist() if isinstance(item, np.ndarray) else item for item in vector]
    return vector


def _vectors_to_lists(vectors: Any) -> Any:
    """Numpy vectors in a vector struct, a vector or a dict of named vectors, as lists"""
    if isinstance(vectors, dict):
        return {name: _to_lists(vector) for name, vector in vectors.items()}
    return _to_lists(vectors)


if PYDANTIC_V2:
    from pydantic import BeforeValidator

    NumpyVectorsToLists: Any = BeforeValidator(_vectors_to_lists)
else:
    NumpyVectorsToLists = None  # pydantic 1 ignores it, it rejects numpy arrays anyway
