import math
from itertools import count, islice
from typing import Optional, Iterable, Dict, List, Union

import numpy as np
from tqdm import tqdm

from qdrant_openapi_client import SyncApis, ApiClient
from qdrant_openapi_client.models.models import PointOperationsAnyOf, PointInsertOperationsAnyOfBatch, PayloadInterface, \
    PayloadInterfaceAnyOf, PayloadInterfaceAnyOf1, PayloadInterfaceAnyOf2, PayloadInterfaceAnyOf3, GeoPoint, \
    Distance, Indexes, PointInsertOperationsAnyOf, PointRequest, SearchRequest, Filter, SearchParams, \
    StorageOperationsAnyOf, \
    StorageOperationsAnyOfCreateCollection, StorageOperationsAnyOf1, FieldIndexOperationsAnyOf, \
    FieldIndexOperationsAnyOf1


def iter_batch(iterable, size) -> Iterable:
    """
    >>> list(iter_batch([1,2,3,4,5], 3))
    [[1, 2, 3], [4, 5]]
    """
    source_iter = iter(iterable)
    while source_iter:
        b = list(islice(source_iter, size))
        if len(b) == 0:
            break
        yield b


class QdrantClient:

    def __init__(self, host="localhost", port=6333):
        self.openapi_client = SyncApis(host=f"http://{host}:{port}")

    @classmethod
    def _json_to_payload(cls, json_data, prefix="") -> Dict[str, PayloadInterface]:
        """
        Function converts json data into flatten typed representation, which Qdrant is able to store

        >>> QdrantClient._json_to_payload({"idx": 123})['idx'].dict()
        {'type': 'integer', 'value': 123}
        >>> QdrantClient._json_to_payload({"idx": 123, "data": {"hi": "there"}})['data__hi'].dict()
        {'type': 'keyword', 'value': 'there'}

        :param json_data: Any json data
        :return: Flatten Qdrant payload. Raises exception if data is not compatible
        """

        res = {}
        for key, val in json_data.items():
            if isinstance(val, str):
                res[prefix + key] = PayloadInterfaceAnyOf(value=val, type="keyword")
            if isinstance(val, int):
                res[prefix + key] = PayloadInterfaceAnyOf1(value=val, type="integer")
            if isinstance(val, float):
                res[prefix + key] = PayloadInterfaceAnyOf2(value=val, type="float")

            if isinstance(val, dict):
                if 'lon' in val and 'lat' in val:
                    res[prefix + key] = PayloadInterfaceAnyOf3(
                        value=GeoPoint(lat=val['lat'], lon=val['lon']),
                        type="geo"
                    )
                else:
                    res = {
                        **res,
                        **cls._json_to_payload(val, prefix=f"{key}__")
                    }

        return res

    @classmethod
    def _payload_to_json(cls, payload: Dict[str, PayloadInterface]) -> dict:
        """
        Function converts Qdrant payload into convenient json
        :param payload: Payload from Qdrant
        :return: Json data
        """
        res = {}
        for key, value_interface in payload.items():
            value = value_interface.dict()['value']
            if len(value) == 1:
                value = value[0]
            res[key] = value
        return res

    @property
    def http(self):
        return self.openapi_client

    @classmethod
    def _iterate_batches(cls,
                         vectors: np.ndarray,
                         payload: Optional[Iterable[dict]],
                         ids: Optional[Iterable[int]],
                         batch_size: int) -> Iterable:
        num_vectors, _dim = vectors.shape
        if ids is None:
            ids = range(num_vectors)

        ids_batches = iter_batch(ids, batch_size)
        if payload is None:
            payload_batches = (None for _ in count())
        else:
            payload = map(cls._json_to_payload, payload)
            payload_batches = iter_batch(payload, batch_size)

        num_batches = int(math.ceil(num_vectors / batch_size))
        vector_batches = (vectors[i * batch_size:(i + 1) * batch_size].tolist() for i in range(num_batches))

        yield from zip(ids_batches, vector_batches, payload_batches)

    def get_payload(self, collection_name: str, ids: List[int]) -> Dict[int, dict]:
        """
        Retrieve points payload by ids
        :param collection_name:
        :param ids: List of ids to retrieve
        :return:
        """
        result = {}
        response = self.http.points_api.get_points(name=collection_name, point_request=PointRequest(ids=ids))

        for record in response.result:
            payload_json = self._payload_to_json(record.payload)
            result[record.id] = payload_json

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
            name=collection_name,
            search_request=SearchRequest(
                vector=query_vector,
                filter=query_filter,
                top=top,
                params=search_params
            )
        )

        if append_payload:
            found_ids = [hit.id for hit in search_result.result]
            payloads = self.get_payload(collection_name=collection_name, ids=found_ids)
            return [
                (hit, payloads.get(hit.id)) for hit in search_result.result
            ]
        else:
            return search_result.result

    def recreate_collection(
            self,
            collection_name: str,
            vector_size: int,
            distance: Distance = None,
            index_params: Optional[Indexes] = None):
        """
        Delete and create empty collection

        :param collection_name: Name of the collection to recreate
        :param vector_size: Vector size of the collection
        :param distance: Which metric to use, default: Dot
        :param index_params: Params for collection index
        :return:
        """
        if distance is None:
            distance = Distance.DOT

        self.http.collections_api.update_collections(
            storage_operations=StorageOperationsAnyOf1(
                delete_collection=collection_name
            )
        )

        self.http.collections_api.update_collections(
            storage_operations=StorageOperationsAnyOf(
                create_collection=StorageOperationsAnyOfCreateCollection(
                    name=collection_name,
                    distance=distance,
                    vector_size=vector_size,
                    index=index_params
                )
            )
        )

    def upload_collection(self,
                          collection_name,
                          vectors: np.ndarray,
                          payload: Optional[Iterable[dict]],
                          ids: Optional[Iterable[int]],
                          batch_size: int = 64):
        """
        Upload vectors and payload to the collection

        :param collection_name: Name of the collection to upload to
        :param vectors: np.ndarray of vectors to upload. Might be mmaped
        :param payload: Iterable of vectors payload, Optional
        :param ids: Iterable of custom vectors ids, Optional
        :param batch_size: How much vectors upload per-request
        :return:
        """
        for ids_batch, vectors_batch, payload_batch in tqdm(self._iterate_batches(vectors, payload, ids, batch_size)):
            self.openapi_client.points_api.update_points(
                name=collection_name,
                collection_update_operations=PointOperationsAnyOf(
                    upsert_points=PointInsertOperationsAnyOf(
                        batch=PointInsertOperationsAnyOfBatch(
                            ids=ids_batch,
                            payloads=payload_batch,
                            vectors=vectors_batch
                        )
                    )
                )
            )

    def create_payload_index(self, collection_name: str, field_name: str):
        """
        Creates index for a given payload field. Indexed fields allow to perform filtered search operations faster.

        :param collection_name: Name of the collection
        :param field_name: Name of the payload field
        :return:
        """
        return self.openapi_client.points_api.update_points(
            name=collection_name,
            wait='true',
            collection_update_operations=FieldIndexOperationsAnyOf(create_index=field_name),
        )

    def delete_payload_index(self, collection_name: str, field_name: str):
        """
        Removes index for a given payload field.

        :param collection_name: Name of the collection
        :param field_name: Name of the payload field
        :return:
        """

        return self.openapi_client.points_api.update_points(
            name=collection_name,
            wait='true',
            storage_operations=FieldIndexOperationsAnyOf1(delete_index=field_name)
        )
