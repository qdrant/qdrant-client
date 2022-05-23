import asyncio
from itertools import count
from typing import Iterable, Any

from grpclib.client import Channel

from qdrant_client.conversions.conversion import payload_to_grpc, RestToGrpc
from qdrant_client.grpc import PointsStub, PointStruct, PointId
from qdrant_client.uploader.uploader import BaseUploader


def iter_over_async(ait, loop):
    ait = ait.__aiter__()

    async def get_next():
        try:
            obj = await ait.__anext__()
            return False, obj
        except StopAsyncIteration:
            return True, None

    while True:
        done, obj = loop.run_until_complete(get_next())
        if done:
            break
        yield obj


async def upload_batch_grpc(points_client: PointsStub, collection_name: str, batch):
    ids_batch, vectors_batch, payload_batch = batch
    if payload_batch is None:
        payload_batch = (None for _ in count())

    points = [
        PointStruct(
            id=RestToGrpc.convert_extended_point_id(idx) if not isinstance(idx, PointId) else idx,
            vector=vector,
            payload=payload_to_grpc(payload or {}),
        ) for idx, vector, payload in zip(ids_batch, vectors_batch, payload_batch)
    ]
    return await points_client.upsert(collection_name=collection_name, points=points)


class GrpcBatchUploader(BaseUploader):

    def __init__(self, host, port, collection_name):
        self.collection_name = collection_name
        self._host = host
        self._port = port

    @classmethod
    def start(cls, collection_name=None, host="localhost", port=6334, **kwargs) -> 'GrpcBatchUploader':
        if not collection_name:
            raise RuntimeError("Collection name could not be empty")

        return cls(host=host, port=port, collection_name=collection_name)

    async def process_upload(self, items):
        async with Channel(host=self._host, port=self._port) as channel:
            points_client = PointsStub(channel)
            for batch in items:
                yield await upload_batch_grpc(points_client, self.collection_name, batch)

    def process(self, items: Iterable[Any]) -> Iterable[Any]:
        loop = asyncio.get_event_loop()
        async_gen = self.process_upload(items)
        sync_gen = iter_over_async(async_gen, loop)
        return sync_gen
