from typing import Iterable, Any

from qdrant_client.http import SyncApis
from qdrant_client.http.models import PointsBatch, Batch
from qdrant_client.uploader.uploader import BaseUploader


def upload_batch(openapi_client: SyncApis, collection_name: str, batch) -> bool:
    ids_batch, vectors_batch, payload_batch = batch

    # Make sure we do not send too many ids in case there is an iterable over vectors,
    # and we do not know how many ids are required in advance
    if len(ids_batch) > len(vectors_batch):
        ids_batch = ids_batch[:len(vectors_batch)]

    if payload_batch is not None:
        payload_batch = list(payload_batch)

    openapi_client.points_api.upsert_points(
        collection_name=collection_name,
        point_insert_operations=PointsBatch(
            batch=Batch(
                ids=ids_batch,
                payloads=payload_batch,
                vectors=vectors_batch
            )
        )
    )
    return True


class RestBatchUploader(BaseUploader):

    def __init__(self, uri, collection_name):
        self.collection_name = collection_name
        self.openapi_client = SyncApis(host=uri)

    @classmethod
    def start(cls, collection_name=None, uri="http://localhost:6333", **kwargs) -> 'RestBatchUploader':
        if not collection_name:
            raise RuntimeError("Collection name could not be empty")
        return cls(uri=uri, collection_name=collection_name)

    def process(self, items: Iterable[Any]) -> Iterable[Any]:
        for batch in items:
            yield upload_batch(self.openapi_client, self.collection_name, batch)
