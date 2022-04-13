from typing import Iterable, Any

from qdrant_client.http import SyncApis
from qdrant_client.http.models import PointsBatch, Batch
from qdrant_client.uploader.uploader import BaseUploader


def upload_batch(openapi_client: SyncApis, collection_name: str, batch) -> bool:
    ids_batch, vectors_batch, payload_batch = batch

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

    def __init__(self, host, port, collection_name):
        self.collection_name = collection_name
        self.openapi_client = SyncApis(host=f"http://{host}:{port}")

    @classmethod
    def start(cls, collection_name=None, host="localhost", port=6333, **kwargs) -> 'RestBatchUploader':
        if not collection_name:
            raise RuntimeError("Collection name could not be empty")
        return cls(host=host, port=port, collection_name=collection_name)

    def process(self, items: Iterable[Any]) -> Iterable[Any]:
        for batch in items:
            yield upload_batch(self.openapi_client, self.collection_name, batch)
