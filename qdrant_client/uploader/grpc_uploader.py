import asyncio
from itertools import count
from typing import Iterable, Any, Dict

import betterproto
from grpclib.client import Channel

from betterproto.lib.google.protobuf import Value, ListValue, Struct, NullValue
from qdrant_client.grpc import PointsStub, PointStruct, PointId
from qdrant_client.uploader.uploader import BaseUploader


def json_to_value(payload: Any) -> Value:
    if payload is None:
        return Value(null_value=NullValue.NULL_VALUE)
    if isinstance(payload, bool):
        return Value(bool_value=payload)
    if isinstance(payload, int):
        return Value(number_value=payload)
    if isinstance(payload, float):
        return Value(number_value=payload)
    if isinstance(payload, str):
        return Value(string_value=payload)
    if isinstance(payload, list):
        return Value(list_value=ListValue(values=[json_to_value(v) for v in payload]))
    if isinstance(payload, dict):
        return Value(struct_value=Struct(fields=dict((k, json_to_value(v)) for k, v in payload.items())))
    return Value(null_value=NullValue.NULL_VALUE)


def value_to_json(value: Value) -> Any:
    if isinstance(value, Value):
        value = value.to_dict(casing=betterproto.Casing.CAMEL)

    if "numberValue" in value:
        return value["numberValue"]
    if "stringValue" in value:
        return value["stringValue"]
    if "boolValue" in value:
        return value["boolValue"]
    if "structValue" in value:
        return dict((key, value_to_json(val)) for key, val in value["structValue"]['fields'].items())
    if "listValue" in value:
        return list(value_to_json(val) for val in value["listValue"]['values'])
    if "nullValue" in value:
        return None

    raise RuntimeError(f"Can't convert value: {value}")


def payload_to_grpc(payload: Dict[str, Any]) -> Dict[str, Value]:
    return dict(
        (key, json_to_value(val))
        for key, val in payload.items()
    )


def grpc_to_payload(grpc: Dict[str, Value]) -> Dict[str, Any]:
    return dict(
        (key, value_to_json(val))
        for key, val in grpc.items()
    )


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
            id=PointId(uuid=idx) if isinstance(idx, str) else PointId(num=idx),
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
