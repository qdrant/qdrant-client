import collections
from typing import Optional, Iterable, Dict, List, Union

import numpy as np
from tqdm import tqdm

from qdrant_client.http import SyncApis
from qdrant_client.http.models import Filter, SearchParams, SearchRequest, Distance, \
    HnswConfigDiff, OptimizersConfigDiff, WalConfigDiff, CreateCollection, CreateFieldIndex, PointRequest, \
    ExtendedPointId, Payload, PayloadSchemaType
from qdrant_client.parallel_processor import ParallelWorkerPool
from qdrant_client.uploader.grpc_uploader import GrpcBatchUploader
from qdrant_client.uploader.rest_uploader import RestBatchUploader


class QdrantClient:
    # If True - single-value payload arrays will be replaced with just values.
    # Warn: Deprecated
    unwrap_payload = False

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

    @property
    def http(self):
        return self.openapi_client

    def get_payload(self, collection_name: str, ids: List[ExtendedPointId]) -> Dict[ExtendedPointId, dict]:
        """
        Retrieve points payload by ids
        :param collection_name:
        :param ids: List of ids to retrieve
        :return:
        """
        result = {}
        response = self.http.points_api.get_points(
            collection_name=collection_name,
            point_request=PointRequest(ids=ids, with_payload=True)
        )

        for record in response.result:
            result[record.id] = record.payload

        return result

    def search(self,
               collection_name: str,
               query_vector: Union[np.ndarray, List[float]],
               query_filter: Optional[Filter] = None,
               search_params: Optional[SearchParams] = None,
               top: int = 10,
               append_payload=True):
        """
        Search for closest vectors in collection taking into account filtering conditions
        :param collection_name: Collection to search in
        :param query_vector: Search for vectors closest to this
        :param query_filter: Exclude vectors which doesn't fit this conditions
        :param search_params: Additional search params
        :param top: How many results return
        :param append_payload: Also return found vectors payload
        :return: Vector ids with score ( + payload if append_payload == True )
        """
        if isinstance(query_vector, np.ndarray):
            query_vector = query_vector.tolist()

        search_result = self.http.points_api.search_points(
            collection_name=collection_name,
            search_request=SearchRequest(
                vector=query_vector,
                filter=query_filter,
                top=top,
                params=search_params,
                with_payload=append_payload
            )
        )

        return search_result.result

    def delete_collection(self, collection_name: str):
        """
        Removes collection and all it's data

        :param collection_name: Name of the collection to delete
        :return:
        """
        return self.http.collections_api.delete_collection(collection_name)

    def recreate_collection(
            self,
            collection_name: str,
            vector_size: int,
            distance: Distance = None,
            hnsw_config: Optional[HnswConfigDiff] = None,
            optimizers_config: Optional[OptimizersConfigDiff] = None,
            wal_config: Optional[WalConfigDiff] = None,
    ):
        """
        Delete and create empty collection

        :param collection_name: Name of the collection to recreate
        :param vector_size: Vector size of the collection
        :param distance: Which metric to use, default: Dot
        :param hnsw_config: Params for HNSW index
        :param optimizers_config: Params for optimizer
        :param wal_config: Params for Write-Ahead-Log
        :return:
        """
        if distance is None:
            distance = Distance.DOT

        self.delete_collection(collection_name)

        self.http.collections_api.create_collection(
            collection_name=collection_name,
            create_collection=CreateCollection(
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
                          payload: Optional[Iterable[dict]],
                          ids: Optional[Iterable[ExtendedPointId]],
                          batch_size: int = 64,
                          parallel: int = 1):
        """
        Upload vectors and payload to the collection

        :param collection_name: Name of the collection to upload to
        :param vectors: np.ndarray of vectors to upload. Might be mmaped
        :param payload: Iterable of vectors payload, Optional
        :param ids: Iterable of custom vectors ids, Optional
        :param batch_size: How many vectors upload per-request
        :param parallel: Number of parallel processes of upload
        :return:
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

    def create_payload_index(self, collection_name: str, field_name: str, field_type: PayloadSchemaType):
        """
        Creates index for a given payload field. Indexed fields allow to perform filtered search operations faster.

        :param collection_name: Name of the collection
        :param field_name: Name of the payload field
        :param field_type: Type of data to index
        :return:
        """
        return self.openapi_client.collections_api.create_field_index(
            collection_name=collection_name,
            create_field_index=CreateFieldIndex(field_name=field_name, field_type=field_type),
            wait=True
        )

    def delete_payload_index(self, collection_name: str, field_name: str):
        """
        Removes index for a given payload field.

        :param collection_name: Name of the collection
        :param field_name: Name of the payload field
        :return:
        """

        return self.openapi_client.collections_api.delete_field_index(
            collection_name=collection_name,
            field_name=field_name,
            wait=True
        )
