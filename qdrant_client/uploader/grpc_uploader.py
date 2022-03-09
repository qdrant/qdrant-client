import asyncio
from itertools import count
from typing import List, Iterable, Any

from grpclib.client import Channel

from qdrant_client import grpc
from qdrant_client.grpc import PointsStub, PointStruct, PointId, KeywordPayload, IntegerPayload, FloatPayload, \
    GeoPayload
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


def process_payload(payload: dict):
    res = {}
    for key, val in payload.items():
        if isinstance(val, str):
            res[key] = grpc.Payload(keyword=KeywordPayload(values=[val]))
            continue

        if isinstance(val, int):
            res[key] = grpc.Payload(integer=IntegerPayload(values=[val]))
            continue

        if isinstance(val, float):
            res[key] = grpc.Payload(float=FloatPayload(values=[val]))
            continue

        if isinstance(val, dict):
            if 'lon' in val and 'lat' in val:
                res[key] = grpc.Payload(geo=GeoPayload(values=[grpc.GeoPoint(lat=val['lat'], lon=val['lon'])]))
                continue
            else:
                raise RuntimeWarning(f"Unsupported payload: ({key} -> {val})")

        if isinstance(val, list):
            if all(isinstance(v, str) for v in val):
                res[key] = grpc.Payload(keyword=KeywordPayload(values=val))
                continue

            if all(isinstance(v, int) for v in val):
                res[key] = grpc.Payload(integer=IntegerPayload(values=val))
                continue

            if all(isinstance(v, float) for v in val):
                res[key] = grpc.Payload(float=FloatPayload(values=val))
                continue

            if all(isinstance(v, dict) and 'lon' in v and 'lat' in v for v in val):
                res[key] = grpc.Payload(geo=GeoPayload(
                    values=[grpc.GeoPoint(lat=v['lat'], lon=v['lon']) for v in val],
                ))
                continue
        raise RuntimeError(f"Payload {key} have unsupported type {type(val)}")
    return res


async def upload_batch_grpc(points_client: PointsStub, collection_name: str, batch):
    ids_batch, vectors_batch, payload_batch = batch
    if payload_batch is None:
        payload_batch = (None for _ in count())

    points = [
        PointStruct(
            id=PointId(uuid=idx) if isinstance(idx, str) else PointId(num=idx),
            vector=vector,
            payload=process_payload(payload) if payload else {},
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
