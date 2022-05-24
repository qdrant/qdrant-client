import asyncio
from typing import Optional, Iterable, List, Union

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
        if prefer_grpc:
            from grpclib.client import Channel
            self._grpc_channel = Channel(host=self._host, port=self._grpc_port)
            self._grpc_points_client = grpc.PointsStub(self._grpc_channel)

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
               with_vector=False,
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
                params=search_params
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
                    with_payload=with_payload
                )
            )

            return search_result.result

    def delete_collection(self, collection_name: str):
        """Removes collection and all it's data

        Args:
            collection_name: Name of the collection to delete

        Returns:
            Operation result
        """

        return self.http.collections_api.delete_collection(collection_name)

    def recreate_collection(self,
                            collection_name: str,
                            vector_size: int,
                            distance: types.Distance,
                            hnsw_config: Optional[types.HnswConfigDiff] = None,
                            optimizers_config: Optional[types.OptimizersConfigDiff] = None,
                            wal_config: Optional[types.WalConfigDiff] = None,
                            ):
        """Delete and create empty collection with given parameters

        Args:
            collection_name: Name of the collection to recreate
            vector_size: Vector size of the collection
            distance: Which metric to use
            hnsw_config: Params for HNSW index
            optimizers_config: Params for optimizer
            wal_config: Params for Write-Ahead-Log

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

        self.http.collections_api.create_collection(
            collection_name=collection_name,
            create_collection=rest.CreateCollection(
                distance=distance,
                vector_size=vector_size,
                hnsw_config=hnsw_config,
                optimizers_config=optimizers_config,
                wal_config=wal_config
            )
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
                             wait=True,
                             ):
        """Creates index for a given payload field.
        Indexed fields allow to perform filtered search operations faster.

        Args:
            collection_name: Name of the collection
            field_name: Name of the payload field
            field_type: Type of data to index
            wait: If `True` - block on the request until the operation is completely applied

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

    def delete_payload_index(self, collection_name: str, field_name: str):
        """Removes index for a given payload field.

        Args:
            collection_name: Name of the collection
            field_name: Name of the payload field

        Returns:
            Operation Result
        """
        return self.openapi_client.collections_api.delete_field_index(
            collection_name=collection_name,
            field_name=field_name,
            wait=True
        )
