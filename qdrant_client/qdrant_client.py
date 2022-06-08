import asyncio
import json
from typing import Optional, Iterable, List, Union, Tuple

import numpy as np
from loguru import logger
from tqdm import tqdm

from qdrant_client import grpc
from qdrant_client.conversions import common_types as types
from qdrant_client.conversions.conversion import RestToGrpc, GrpcToRest
from qdrant_client.http import SyncApis
from qdrant_client.http import models as rest
from qdrant_client.parallel_processor import ParallelWorkerPool
from qdrant_client.uploader.grpc_uploader import GrpcBatchUploader
from qdrant_client.uploader.rest_uploader import RestBatchUploader


class QdrantClient:
    """Entry point to communicate with Qdrant service via REST or gPRC API.

    It combines interface classes and endpoint implementation.
    Additionally, it provides custom implementations for frequently used methods like initial collection upload.

    All methods in QdrantClient accept both gRPC and REST structures as an input.
    Conversion will be performed automatically.

    .. note::
        This module methods are wrappers around generated client code for gRPC and REST methods.
        If you need lower-level access to generated clients, use following properties:

        - :meth:`QdrantClient.grpc_points`
        - :meth:`QdrantClient.grpc_collections`
        - :meth:`QdrantClient.rest`

    .. note::
        If you need to use async versions of API, please consider using raw implementations of clients directly:

        - For REST: :class:`~qdrant_client.http.api_client.AsyncApis`
        - For gRPC: :class:`~qdrant_client.grpc.PointsStub` and :class:`~qdrant_client.grpc.CollectionsStub`

    Args:
        host: Host name of Qdrant service. Default: `localhost`
        port: Port of the REST API interface. Default: 6333
        grpc_port: Port of the gRPC interface. Default: 6334
        prefer_grpc: If `true` - use gPRC interface whenever possible in custom methods.
        **kwargs: Additional arguments passed directly into REST client initialization
    """

    def __init__(self,
                 host="localhost",
                 port=6333,
                 grpc_port=6334,
                 prefer_grpc=False,
                 **kwargs):
        self._prefer_grpc = prefer_grpc
        self._grpc_port = grpc_port
        self._host = host
        self._port = port
        self.openapi_client = SyncApis(host=f"http://{host}:{port}", **kwargs)

        self._grpc_channel = None
        self._grpc_points_client = None
        self._grpc_collections_client = None
        if prefer_grpc:
            self._init_grpc_points_client()
            self._init_grpc_collections_client()

    def _init_grpc_points_client(self):
        if self._grpc_channel is None:
            from grpclib.client import Channel
            self._grpc_channel = Channel(host=self._host, port=self._grpc_port)
        self._grpc_points_client = grpc.PointsStub(self._grpc_channel)

    def _init_grpc_collections_client(self):
        if self._grpc_channel is None:
            from grpclib.client import Channel
            self._grpc_channel = Channel(host=self._host, port=self._grpc_port)
        self._grpc_collections_client = grpc.CollectionsStub(self._grpc_channel)

    @property
    def grpc_collections(self):
        """gRPC client for collections methods

        Returns:
            An instance of raw gRPC client, generated from Protobuf
        """
        if self._grpc_collections_client is None:
            self._init_grpc_collections_client()
        return self._grpc_collections_client

    @property
    def grpc_points(self):
        """gRPC client for points methods

        Returns:
            An instance of raw gRPC client, generated from Protobuf
        """
        if self._grpc_points_client is None:
            self._init_grpc_points_client()
        return self._grpc_points_client

    @property
    def rest(self):
        """REST Client

        Returns:
            An instance of raw REST API client, generated from OpenAPI schema
        """
        return self.openapi_client

    @property
    def http(self):
        """REST Client

        Returns:
            An instance of raw REST API client, generated from OpenAPI schema
        """
        return self.openapi_client

    def search(self,
               collection_name: str,
               query_vector: Union[np.ndarray, List[float]],
               query_filter: Optional[types.Filter] = None,
               search_params: Optional[types.SearchParams] = None,
               top: int = 10,
               with_payload: Union[bool, List[str], types.PayloadSelector] = True,
               with_vector: bool = False,
               score_threshold: Optional[float] = None,
               append_payload=True) -> List[types.ScoredPoint]:
        """Search for closest vectors in collection taking into account filtering conditions

        Args:
            collection_name: Collection to search in
            query_vector: Search for vectors closest to this
            query_filter:
                - Exclude vectors which doesn't fit given conditions.
                - If `None` - search among all vectors
            search_params: Additional search params
            top: How many results return
            with_payload:
                - Specify which stored payload should be attached to the result.
                - If `True` - attach all payload
                - If `False` - do not attach any payload
                - If List of string - include only specified fields
                - If `PayloadSelector` - use explicit rules
            with_vector:
                - If `True` - Attach stored vector to the search result.
                - If `False` - Do not attach vector.
                - Default: `False`
            score_threshold:
                Define a minimal score threshold for the result.
                If defined, less similar results will not be returned.
                Score of the returned result might be higher or smaller than the threshold depending
                on the Distance function used.
                E.g. for cosine similarity only higher scores will be returned.
            append_payload: Same as `with_payload`. Deprecated.

        Examples:

        `Search with filter`::

            qdrant.search(
                collection_name="test_collection",
                query_vector=[1.0, 0.1, 0.2, 0.7],
                query_filter=Filter(
                    must=[
                        FieldCondition(
                            key='color',
                            range=Match(
                                value="red"
                            )
                        )
                    ]
                )
            )

        Returns:
            List of found close points with similarity scores.
        """
        if not append_payload:
            logger.warning("Usage of `append_payload` is deprecated. Please consider using `with_payload` instead")
            with_payload = append_payload

        if isinstance(query_vector, np.ndarray):
            query_vector = query_vector.tolist()

        if self._prefer_grpc:
            if isinstance(query_filter, rest.Filter):
                query_filter = RestToGrpc.convert_filter(model=query_filter)

            if isinstance(search_params, rest.SearchParams):
                search_params = RestToGrpc.convert_search_params(search_params)

            if isinstance(with_payload, (
                    bool,
                    list,
                    rest.PayloadSelectorInclude,
                    rest.PayloadSelectorExclude
            )):
                with_payload = RestToGrpc.convert_with_payload_interface(with_payload)

            loop = asyncio.get_event_loop()

            res: grpc.SearchResponse = loop.run_until_complete(self._grpc_points_client.search(
                collection_name=collection_name,
                vector=query_vector,
                filter=query_filter,
                top=top,
                with_vector=with_vector,
                with_payload=with_payload,
                params=search_params,
                score_threshold=score_threshold
            ))

            return [GrpcToRest.convert_scored_point(hit) for hit in res.result]

        else:
            if isinstance(query_filter, grpc.Filter):
                query_filter = GrpcToRest.convert_filter(model=query_filter)

            if isinstance(search_params, grpc.SearchParams):
                search_params = GrpcToRest.convert_search_params(search_params)

            if isinstance(with_payload, grpc.WithPayloadSelector):
                with_payload = GrpcToRest.convert_with_payload_selector(with_payload)

            search_result = self.http.points_api.search_points(
                collection_name=collection_name,
                search_request=rest.SearchRequest(
                    vector=query_vector,
                    filter=query_filter,
                    top=top,
                    params=search_params,
                    with_vector=with_vector,
                    with_payload=with_payload,
                    score_threshold=score_threshold
                )
            )

            return search_result.result

    def recommend(
            self,
            collection_name: str,
            positive: List[types.PointId],
            negative: List[types.PointId] = None,
            query_filter: Optional[types.Filter] = None,
            search_params: Optional[types.SearchParams] = None,
            top: int = 10,
            with_payload: Union[bool, List[str], types.PayloadSelector] = True,
            with_vector: bool = False,
            score_threshold: Optional[float] = None,
    ) -> List[types.ScoredPoint]:
        """Recommend points: search for similar points based on already stored in Qdrant examples.

        Provide IDs of the stored points, and Qdrant will perform search based on already existing vectors.
        This functionality is especially useful for recommendation over existing collection of points.

        Args:
            collection_name: Collection to search in
            positive:
                List of stored point IDs, which should be used as reference for similarity search.
                If there is only one ID provided - this request is equivalent to the regular search with vector of that point.
                If there are more than one IDs, Qdrant will attempt to search for similar to all of them.
                Recommendation for multiple vectors is experimental. Its behaviour may change in the future.
            negative:
                List of stored point IDs, which should be dissimilar to the search result.
                Negative examples is an experimental functionality. Its behaviour may change in the future.
            query_filter:
                - Exclude vectors which doesn't fit given conditions.
                - If `None` - search among all vectors
            search_params: Additional search params
            top: How many results return
            with_payload:
                - Specify which stored payload should be attached to the result.
                - If `True` - attach all payload
                - If `False` - do not attach any payload
                - If List of string - include only specified fields
                - If `PayloadSelector` - use explicit rules
            with_vector:
                - If `True` - Attach stored vector to the search result.
                - If `False` - Do not attach vector.
                - Default: `False`
            score_threshold:
                Define a minimal score threshold for the result.
                If defined, less similar results will not be returned.
                Score of the returned result might be higher or smaller than the threshold depending
                on the Distance function used.
                E.g. for cosine similarity only higher scores will be returned.

        Returns:
            List of recommended points with similarity scores.
        """

        if negative is None:
            negative = []

        if self._prefer_grpc:
            positive = [
                RestToGrpc.convert_extended_point_id(point_id) if isinstance(point_id, (str, int)) else point_id
                for point_id in positive
            ]

            negative = [
                RestToGrpc.convert_extended_point_id(point_id) if isinstance(point_id, (str, int)) else point_id
                for point_id in negative
            ]

            if isinstance(query_filter, rest.Filter):
                query_filter = RestToGrpc.convert_filter(model=query_filter)

            if isinstance(search_params, rest.SearchParams):
                search_params = RestToGrpc.convert_search_params(search_params)

            if isinstance(with_payload, (
                    bool,
                    list,
                    rest.PayloadSelectorInclude,
                    rest.PayloadSelectorExclude
            )):
                with_payload = RestToGrpc.convert_with_payload_interface(with_payload)

            loop = asyncio.get_event_loop()

            res: grpc.SearchResponse = loop.run_until_complete(self._grpc_points_client.recommend(
                collection_name=collection_name,
                positive=positive,
                negative=negative,
                filter=query_filter,
                top=top,
                with_vector=with_vector,
                with_payload=with_payload,
                params=search_params,
                score_threshold=score_threshold
            ))

            return [GrpcToRest.convert_scored_point(hit) for hit in res.result]
        else:
            positive = [
                GrpcToRest.convert_point_id(point_id) if isinstance(point_id, grpc.PointId) else point_id
                for point_id in positive
            ]

            negative = [
                GrpcToRest.convert_point_id(point_id) if isinstance(point_id, grpc.PointId) else point_id
                for point_id in negative
            ]

            if isinstance(query_filter, grpc.Filter):
                query_filter = GrpcToRest.convert_filter(model=query_filter)

            if isinstance(search_params, grpc.SearchParams):
                search_params = GrpcToRest.convert_search_params(search_params)

            if isinstance(with_payload, grpc.WithPayloadSelector):
                with_payload = GrpcToRest.convert_with_payload_selector(with_payload)

            return self.openapi_client.points_api.recommend_points(
                collection_name=collection_name,
                recommend_request=rest.RecommendRequest(
                    filter=query_filter,
                    negative=negative,
                    params=search_params,
                    positive=positive,
                    top=top,
                    with_payload=with_payload,
                    with_vector=with_vector,
                    score_threshold=score_threshold
                )
            ).result

    def scroll(
            self,
            collection_name: str,
            scroll_filter: Optional[types.Filter] = None,
            limit: int = 10,
            offset: Optional[types.PointId] = None,
            with_payload: Union[bool, List[str], types.PayloadSelector] = True,
            with_vector: bool = False,
    ) -> Tuple[List[types.Record], Optional[types.PointId]]:
        """Scroll over all (matching) points in the collection.

        This method provides a way to iterate over all stored points with some optional filtering condition.
        Scroll does not apply any similarity estimations, it will return points sorted by id in ascending order.

        Args:
            collection_name: Name of the collection
            scroll_filter: If provided - only returns points matching filtering conditions
            limit: How many points to return
            offset: If provided - skip points with ids less than given `offset`
            with_payload:
                - Specify which stored payload should be attached to the result.
                - If `True` - attach all payload
                - If `False` - do not attach any payload
                - If List of string - include only specified fields
                - If `PayloadSelector` - use explicit rules
            with_vector:
                - If `True` - Attach stored vector to the search result.
                - If `False` - Do not attach vector.
                - Default: `False`

        Returns:
            A pair of (List of points) and (optional offset for the next scroll request).
            If next page offset is `None` - there is no more points in the collection to scroll.
        """
        if self._prefer_grpc:
            if isinstance(offset, (int, str)):
                offset = RestToGrpc.convert_extended_point_id(offset)

            if isinstance(scroll_filter, rest.Filter):
                scroll_filter = RestToGrpc.convert_filter(model=scroll_filter)

            if isinstance(with_payload, (
                    bool,
                    list,
                    rest.PayloadSelectorInclude,
                    rest.PayloadSelectorExclude
            )):
                with_payload = RestToGrpc.convert_with_payload_interface(with_payload)

            loop = asyncio.get_event_loop()

            res: grpc.ScrollResponse = loop.run_until_complete(self._grpc_points_client.scroll(
                collection_name=collection_name,
                filter=scroll_filter,
                offset=offset,
                with_vector=with_vector,
                with_payload=with_payload,
                limit=limit
            ))

            return [
                       GrpcToRest.convert_retrieved_point(point)
                       for point in res.result
                   ], res.next_page_offset
        else:
            if isinstance(offset, grpc.PointId):
                offset = GrpcToRest.convert_point_id(offset)

            if isinstance(scroll_filter, grpc.Filter):
                scroll_filter = GrpcToRest.convert_filter(model=scroll_filter)

            if isinstance(with_payload, grpc.WithPayloadSelector):
                with_payload = GrpcToRest.convert_with_payload_selector(with_payload)

            scroll_result: rest.ScrollResult = self.openapi_client.points_api.scroll_points(
                collection_name=collection_name,
                scroll_request=rest.ScrollRequest(
                    filter=scroll_filter,
                    limit=limit,
                    offset=offset,
                    with_payload=with_payload,
                    with_vector=with_vector
                )
            ).result

            return scroll_result.points, scroll_result.next_page_offset

    def upsert(
            self,
            collection_name: str,
            points: types.Points,
            wait: bool = True,
    ) -> types.UpdateResult:
        """Update or insert a new point into the collection.

        If point with given ID already exists - it will be overwritten.

        Args:
            collection_name: To which collection to insert
            wait: Await for the results to be processed.

                - If `true`, result will be returned only when all changes are applied
                - If `false`, result will be returned immediately after the confirmation of receiving.
            points: Batch or list of points to insert

        Returns:
            Operation result
        """
        if self._prefer_grpc:
            if isinstance(points, rest.Batch):
                points = [
                    grpc.PointStruct(
                        id=RestToGrpc.convert_extended_point_id(points.ids[idx]),
                        vector=points.vectors[idx],
                        payload=RestToGrpc.convert_payload(
                            points.payloads[idx]) if points.payloads is not None else None,
                    ) for idx in range(len(points.ids))
                ]
            if isinstance(points, list):
                points = [
                    RestToGrpc.convert_point_struct(point)
                    if isinstance(point, rest.PointStruct) else point
                    for point in points
                ]

            return GrpcToRest.convert_update_result(
                asyncio.get_event_loop().run_until_complete(self._grpc_points_client.upsert(
                    collection_name=collection_name,
                    wait=wait,
                    points=points
                )).result)
        else:
            if isinstance(points, list):
                points = [
                    GrpcToRest.convert_point_struct(point)
                    if isinstance(point, grpc.PointStruct) else point
                    for point in points
                ]

                points = rest.PointsList(points=points)

            if isinstance(points, rest.Batch):
                points = rest.PointsBatch(batch=points)

            return self.openapi_client.points_api.upsert_points(
                collection_name=collection_name,
                wait=wait,
                point_insert_operations=points
            ).result

    def retrieve(
            self,
            collection_name: str,
            ids: List[types.PointId],
            with_payload: Union[bool, List[str], types.PayloadSelector] = True,
            with_vector: bool = False,
    ) -> List[types.Record]:
        """Retrieve stored points by IDs

        Args:
            collection_name: Name of the collection to lookup in
            ids: list of IDs to lookup
            with_payload:
                - Specify which stored payload should be attached to the result.
                - If `True` - attach all payload
                - If `False` - do not attach any payload
                - If List of string - include only specified fields
                - If `PayloadSelector` - use explicit rules
            with_vector:
                - If `True` - Attach stored vector to the search result.
                - If `False` - Do not attach vector.
                - Default: `False`

        Returns:
            List of points
        """
        if self._prefer_grpc:
            if isinstance(with_payload, (
                    bool,
                    list,
                    rest.PayloadSelectorInclude,
                    rest.PayloadSelectorExclude
            )):
                with_payload = RestToGrpc.convert_with_payload_interface(with_payload)

            ids = [
                RestToGrpc.convert_extended_point_id(idx) if isinstance(idx, (int, str)) else idx
                for idx in ids
            ]

            result = asyncio.get_event_loop().run_until_complete(
                self._grpc_points_client.get(
                    collection_name=collection_name,
                    ids=ids,
                    with_payload=with_payload,
                    with_vector=with_vector
                )).result

            return [
                GrpcToRest.convert_retrieved_point(record)
                for record in result
            ]

        else:
            if isinstance(with_payload, grpc.WithPayloadSelector):
                with_payload = GrpcToRest.convert_with_payload_selector(with_payload)

            ids = [
                GrpcToRest.convert_point_id(idx) if isinstance(idx, grpc.PointId) else idx
                for idx in ids
            ]

            return self.openapi_client.points_api.get_points(
                collection_name=collection_name,
                point_request=rest.PointRequest(
                    ids=ids,
                    with_payload=with_payload,
                    with_vector=with_vector
                )
            ).result

    def delete(
            self,
            collection_name: str,
            points_selector: types.PointsSelector,
            wait: bool = True,
    ) -> types.UpdateResult:
        """Deletes selected points from collection

        Args:
            collection_name: Name of the collection
            wait: Await for the results to be processed.

                - If `true`, result will be returned only when all changes are applied
                - If `false`, result will be returned immediately after the confirmation of receiving.
            points_selector: Selects points based on list of IDs or filter

        Returns:
            Operation result
        """
        if self._prefer_grpc:
            if isinstance(points_selector, (rest.PointIdsList, rest.FilterSelector)):
                points_selector = RestToGrpc.convert_points_selector(points_selector)

            return GrpcToRest.convert_update_result(
                asyncio.get_event_loop().run_until_complete(self._grpc_points_client.delete(
                    collection_name=collection_name,
                    wait=wait,
                    points=points_selector
                )).result)
        else:
            if isinstance(points_selector, grpc.PointsSelector):
                points_selector = GrpcToRest.convert_points_selector(points_selector)

            self.openapi_client.points_api.delete_points(
                collection_name=collection_name,
                wait=wait,
                points_selector=points_selector
            )

    def set_payload(
            self,
            collection_name: str,
            payload: types.Payload,
            points: List[types.PointId],
            wait: bool = True,
    ) -> types.UpdateResult:
        """Modifies payload of the specified points

        Examples:

        `Set payload`::

            # Assign payload value with key `"key"` to points 1, 2, 3.
            # If payload value with specified key already exists - it will be overwritten
            qdrant_client.set_payload(
                collection_name="test_collection",
                wait=True,
                payload={
                    "key": "value"
                },
                points=[1,2,3]
            )

        Args:
            collection_name: Name of the collection
            wait: Await for the results to be processed.

                - If `true`, result will be returned only when all changes are applied
                - If `false`, result will be returned immediately after the confirmation of receiving.
            payload: Key-value pairs of payload to assign
            points: List of affected points. Example: `points=[1, 2, 3, "cd3b53f0-11a7-449f-bc50-d06310e7ed90"]`

        Returns:
            Operation result
        """
        if self._prefer_grpc:
            points = [
                RestToGrpc.convert_extended_point_id(idx) if isinstance(idx, (int, str)) else idx
                for idx in points
            ]
            return GrpcToRest.convert_update_result(
                asyncio.get_event_loop().run_until_complete(self._grpc_points_client.set_payload(
                    collection_name=collection_name,
                    wait=wait,
                    payload=RestToGrpc.convert_payload(payload),
                    points=points
                )).result)
        else:
            points = [
                GrpcToRest.convert_point_id(idx) if isinstance(idx, grpc.PointId) else idx
                for idx in points
            ]

            return self.openapi_client.points_api.set_payload(
                collection_name=collection_name,
                wait=wait,
                set_payload=rest.SetPayload(
                    payload=payload,
                    points=points
                )
            )

    def delete_payload(
            self,
            collection_name: str,
            keys: List[str],
            points: List[types.PointId],
            wait: bool = True,
    ):
        """Remove values from point's payload

        Args:
            collection_name: Name of the collection
            wait: Await for the results to be processed.

                - If `true`, result will be returned only when all changes are applied
                - If `false`, result will be returned immediately after the confirmation of receiving.
            keys: List of payload keys to remove
            points: List of affected points. Example: `points=[1, 2, 3, "cd3b53f0-11a7-449f-bc50-d06310e7ed90"]`

        Returns:
            Operation result
        """
        if self._prefer_grpc:
            points = [
                RestToGrpc.convert_extended_point_id(idx) if isinstance(idx, (int, str)) else idx
                for idx in points
            ]
            return GrpcToRest.convert_update_result(
                asyncio.get_event_loop().run_until_complete(self._grpc_points_client.delete_payload(
                    collection_name=collection_name,
                    wait=wait,
                    keys=keys,
                    points=points
                )).result)
        else:
            points = [
                GrpcToRest.convert_point_id(idx) if isinstance(idx, grpc.PointId) else idx
                for idx in points
            ]
            return self.openapi_client.points_api.delete_payload(
                collection_name=collection_name,
                wait=wait,
                delete_payload=rest.DeletePayload(
                    keys=keys,
                    points=points
                )
            )

    def clear_payload(
            self,
            collection_name: str,
            points_selector: types.PointsSelector,
            wait: bool = True,
    ):
        """Delete all payload for selected points

        Args:
            collection_name: Name of the collection
            wait: Await for the results to be processed.

                - If `true`, result will be returned only when all changes are applied
                - If `false`, result will be returned immediately after the confirmation of receiving.
            points_selector: Selects points based on list of IDs or filter

        Returns:
            Operation result
        """
        if self._prefer_grpc:
            if isinstance(points_selector, (rest.PointIdsList, rest.FilterSelector)):
                points_selector = RestToGrpc.convert_points_selector(points_selector)

            return GrpcToRest.convert_update_result(
                asyncio.get_event_loop().run_until_complete(self._grpc_points_client.clear_payload(
                    collection_name=collection_name,
                    wait=wait,
                    points=points_selector
                )).result)
        else:
            if isinstance(points_selector, grpc.PointsSelector):
                points_selector = GrpcToRest.convert_points_selector(points_selector)

            return self.openapi_client.points_api.clear_payload(
                collection_name=collection_name,
                wait=wait,
                points_selector=points_selector
            ).result

    def update_collection_aliases(
            self,
            change_aliases_operations: List[types.AliasOperations],
            timeout: Optional[int] = None
    ):
        """Operation for performing changes of collection aliases.

        Alias changes are atomic, meaning that no collection modifications can happen between alias operations.

        Args:
            change_aliases_operations: List of operations to perform
            timeout:
                Wait for operation commit timeout in seconds.
                If timeout is reached - request will return with service error.

        Returns:
            Operation result
        """
        change_aliases_operation = [
            GrpcToRest.convert_alias_operations(operation)
            if isinstance(operation, grpc.AliasOperations) else operation
            for operation in change_aliases_operations
        ]

        return self.http.collections_api.update_aliases(
            timeout=timeout,
            change_aliases_operation=rest.ChangeAliasesOperation(
                actions=change_aliases_operation
            )
        )

    def get_collections(self) -> types.CollectionsResponse:
        """Get list name of all existing collections

        Returns:
            List of the collections
        """
        return self.http.collections_api.get_collections().result

    def get_collection(self, collection_name: str) -> types.CollectionInfo:
        """Get detailed information about specified existing collection

        Args:
            collection_name: Name of the collection

        Returns:
            Detailed information about the collection
        """
        return self.http.collections_api.get_collection(collection_name=collection_name).result

    def update_collection(
            self,
            collection_name: str,
            optimizer_config: Optional[types.OptimizersConfigDiff],
            timeout: Optional[int] = None
    ):
        """Update parameters of the collection

        Args:
            collection_name: Name of the collection
            optimizer_config: Override for optimizer configuration
            timeout:
                Wait for operation commit timeout in seconds.
                If timeout is reached - request will return with service error.

        Returns:
            Operation result
        """
        if isinstance(optimizer_config, grpc.OptimizersConfigDiff):
            optimizer_config = GrpcToRest.convert_optimizers_config_diff(optimizer_config)
        return self.http.collections_api.update_collection(
            collection_name,
            update_collection=rest.UpdateCollection(
                optimizers_config=optimizer_config
            ),
            timeout=timeout
        )

    def delete_collection(
            self,
            collection_name: str,
            timeout: Optional[int] = None
    ):
        """Removes collection and all it's data

        Args:
            collection_name: Name of the collection to delete
            timeout:
                Wait for operation commit timeout in seconds.
                If timeout is reached - request will return with service error.

        Returns:
            Operation result
        """

        return self.http.collections_api.delete_collection(
            collection_name,
            timeout=timeout
        )

    def recreate_collection(self,
                            collection_name: str,
                            vector_size: int,
                            distance: types.Distance,
                            shard_number: Optional[int] = None,
                            on_disk_payload: Optional[bool] = None,
                            hnsw_config: Optional[types.HnswConfigDiff] = None,
                            optimizers_config: Optional[types.OptimizersConfigDiff] = None,
                            wal_config: Optional[types.WalConfigDiff] = None,
                            timeout: Optional[int] = None
                            ):
        """Delete and create empty collection with given parameters

        Args:
            collection_name: Name of the collection to recreate
            vector_size: Vector size of the collection
            distance: Which metric to use
            shard_number: Number of shards in collection. Default is 1, minimum is 1.
            on_disk_payload:
                If true - point`s payload will not be stored in memory.
                It will be read from the disk every time it is requested.
                This setting saves RAM by (slightly) increasing the response time.
                Note: those payload values that are involved in filtering and are indexed - remain in RAM.
            hnsw_config: Params for HNSW index
            optimizers_config: Params for optimizer
            wal_config: Params for Write-Ahead-Log
            timeout:
                Wait for operation commit timeout in seconds.
                If timeout is reached - request will return with service error.

        Returns:
            Operation result
        """

        if isinstance(distance, grpc.Distance):
            distance = GrpcToRest.convert_distance(distance)

        if isinstance(hnsw_config, grpc.HnswConfigDiff):
            hnsw_config = GrpcToRest.convert_hnsw_config_diff(hnsw_config)

        if isinstance(optimizers_config, grpc.OptimizersConfigDiff):
            optimizers_config = GrpcToRest.convert_optimizers_config_diff(optimizers_config)

        if isinstance(wal_config, grpc.WalConfigDiff):
            wal_config = GrpcToRest.convert_wal_config_diff(wal_config)

        self.delete_collection(collection_name)

        create_collection_request = rest.CreateCollection(
            distance=distance,
            vector_size=vector_size,
            shard_number=shard_number,
            on_disk_payload=on_disk_payload,
            hnsw_config=hnsw_config,
            optimizers_config=optimizers_config,
            wal_config=wal_config
        )

        print(json.dumps(create_collection_request.dict(), indent=2))

        self.http.collections_api.create_collection(
            collection_name=collection_name,
            create_collection=create_collection_request,
            timeout=timeout
        )

    def upload_collection(self,
                          collection_name,
                          vectors: np.ndarray,
                          payload: Optional[Iterable[dict]] = None,
                          ids: Optional[Iterable[types.PointId]] = None,
                          batch_size: int = 64,
                          parallel: int = 1):
        """Upload vectors and payload to the collection

        Args:
            collection_name:  Name of the collection to upload to
            vectors: np.ndarray of vectors to upload. Might be mmaped
            payload: Iterable of vectors payload, Optional, Default: None
            ids: Iterable of custom vectors ids, Optional, Default: None
            batch_size: How many vectors upload per-request, Default: 64
            parallel: Number of parallel processes of upload
        """
        if self._prefer_grpc:
            updater_class = GrpcBatchUploader
            port = self._grpc_port
        else:
            updater_class = RestBatchUploader
            port = self._port

        batches_iterator = updater_class.iterate_batches(vectors=vectors,
                                                         payload=payload,
                                                         ids=ids,
                                                         batch_size=batch_size)
        if parallel == 1:
            updater = updater_class.start(collection_name=collection_name, host=self._host, port=port)
            for _ in tqdm(updater.process(batches_iterator)):
                pass
        else:
            pool = ParallelWorkerPool(parallel, updater_class)
            for _ in tqdm(pool.unordered_map(batches_iterator,
                                             collection_name=collection_name,
                                             host=self._host,
                                             port=port)):
                pass

    def create_payload_index(self,
                             collection_name: str,
                             field_name: str,
                             field_type: types.PayloadSchemaType,
                             wait: bool = True,
                             ):
        """Creates index for a given payload field.
        Indexed fields allow to perform filtered search operations faster.

        Args:
            collection_name: Name of the collection
            field_name: Name of the payload field
            field_type: Type of data to index
            wait: Await for the results to be processed.

                - If `true`, result will be returned only when all changes are applied
                - If `false`, result will be returned immediately after the confirmation of receiving.

        Returns:
            Operation Result
        """
        if isinstance(field_type, grpc.PayloadSchemaType):
            field_type = GrpcToRest.convert_payload_schema_type(field_type)

        return self.openapi_client.collections_api.create_field_index(
            collection_name=collection_name,
            create_field_index=rest.CreateFieldIndex(field_name=field_name, field_type=field_type),
            wait=wait
        )

    def delete_payload_index(self, collection_name: str, field_name: str, wait: bool = True):
        """Removes index for a given payload field.

        Args:
            collection_name: Name of the collection
            field_name: Name of the payload field
            wait: Await for the results to be processed.

                - If `true`, result will be returned only when all changes are applied
                - If `false`, result will be returned immediately after the confirmation of receiving.

        Returns:
            Operation Result
        """
        return self.openapi_client.collections_api.delete_field_index(
            collection_name=collection_name,
            field_name=field_name,
            wait=wait
        )
