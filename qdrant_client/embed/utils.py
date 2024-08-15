from typing import Union, List, Sequence

from qdrant_client.conversions import common_types as types
from qdrant_client.http import models
from qdrant_client import grpc


def inspect_query_and_prefetch_types(
    query: Union[
        types.PointId,
        List[float],
        List[List[float]],
        types.SparseVector,
        types.Query,
        types.NumpyArray,
        types.Document,
        None,
    ],
    prefetch: Union[types.Prefetch, List[types.Prefetch], None],
) -> bool:
    """Check whether there are types which require inference

    Args:
        query: types.QueryInterface - vector or query instance to verify
        prefetch: prefetch structure which might contain nested queries

    Returns:
        bool: whether inference is required or not
    """
    query_requires_inference = inspect_query_types(query)
    if query_requires_inference:
        return True

    return inspect_prefetch_types(prefetch)


def inspect_prefetch_types(prefetch: Union[types.Prefetch, List[types.Prefetch], None]) -> bool:
    prefetch_requires_inference = False

    if isinstance(prefetch, types.Prefetch):
        prefetch_requires_inference = inspect_query_and_prefetch_types(
            prefetch.query, prefetch.prefetch
        )
    elif isinstance(prefetch, List):
        prefetch_requires_inference = any(
            [
                inspect_query_and_prefetch_types(single_prefetch.query, single_prefetch.prefetch)
                for single_prefetch in prefetch
            ]
        )
    return prefetch_requires_inference


def inspect_query_types(
    query: Union[
        types.PointId,
        List[float],
        List[List[float]],
        types.SparseVector,
        types.Query,
        types.NumpyArray,
        types.Document,
        None,
    ],
) -> bool:
    """Check whether query requires inference"""
    return isinstance(query, types.Document)


def inspect_batch(points: models.Batch) -> bool:
    vectors = points.vectors
    if isinstance(vectors, dict):
        for key, values in vectors.items():
            if isinstance(next(iter(values)), models.Document):
                return True
        return False
    else:
        for vector in points.vectors:
            return isinstance(vector, types.Document)
    return False


def inspect_points(points: types.Points) -> bool:
    """Check whether point requires inference"""

    if isinstance(points, models.Batch):
        return inspect_batch(points)

    return inspect_point_structs(points)


def inspect_point_structs(points: Sequence[Union[models.PointStruct, grpc.PointStruct]]) -> bool:
    for point in points:
        if isinstance(point, grpc.PointStruct):
            return False

        if isinstance(point.vector, types.Document):
            return True
    return False


def inspect_query_requests(requests: Sequence[types.QueryRequest]) -> bool:
    """Check whether query request contains queries requiring inference"""
    for request in requests:
        if inspect_query_and_prefetch_types(request.query, request.prefetch):
            return True
    return False


def inspect_point_vectors(points: Sequence[types.PointVectors]) -> bool:
    """Check whether point vectors require inference"""
    for point_vector in points:
        if isinstance(point_vector.vector, types.Document):
            return True
    return False


def inspect_update_operations(update_operations: Sequence[types.UpdateOperation]) -> bool:
    """Check whether vectors in update_operations require inference"""
    requires_inference = False

    for update_operation in update_operations:
        if isinstance(update_operation, models.UpsertOperation):
            operation = update_operation.upsert
            if isinstance(operation, models.PointsBatch):
                requires_inference = inspect_batch(operation.batch)
            else:
                requires_inference = inspect_point_structs(operation.points)

        elif isinstance(update_operation, models.UpdateVectorsOperation):
            operation = update_operation.update_vectors
            requires_inference = inspect_point_vectors(operation.vectors)
        if requires_inference:
            return True

    return False
