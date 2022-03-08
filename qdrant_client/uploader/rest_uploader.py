from typing import Dict, Iterable, Any

from qdrant_client.http import SyncApis
from qdrant_client.http.models import PayloadInterface, PayloadInterfaceStrictOneOf, PayloadInterfaceStrictOneOf1, \
    PayloadInterfaceStrictOneOf2, PayloadInterfaceStrictOneOf3, GeoPoint, PointsBatch, Batch
from qdrant_client.uploader.uploader import BaseUploader


def json_to_payload(json_data, prefix="") -> Dict[str, PayloadInterface]:
    """
    Function converts json data into flatten typed representation, which Qdrant is able to store

    >>> json_to_payload({"idx": 123})['idx'].dict()
    {'type': 'integer', 'value': 123}
    >>> json_to_payload({"idx": 123, "data": {"hi": "there"}})['data__hi'].dict()
    {'type': 'keyword', 'value': 'there'}

    :param json_data: Any json data
    :param prefix: key prefix
    :return: Flatten Qdrant payload. Raises exception if data is not compatible
    """

    res = {}
    for key, val in json_data.items():
        if isinstance(val, str):
            res[prefix + key] = PayloadInterfaceStrictOneOf(value=val, type="keyword")
            continue

        if isinstance(val, int):
            res[prefix + key] = PayloadInterfaceStrictOneOf1(value=val, type="integer")
            continue

        if isinstance(val, float):
            res[prefix + key] = PayloadInterfaceStrictOneOf2(value=val, type="float")
            continue

        if isinstance(val, dict):
            if 'lon' in val and 'lat' in val:
                res[prefix + key] = PayloadInterfaceStrictOneOf3(
                    value=GeoPoint(lat=val['lat'], lon=val['lon']),
                    type="geo"
                )
            else:
                res = {
                    **res,
                    **json_to_payload(val, prefix=f"{key}__")
                }
            continue

        if isinstance(val, list):
            if all(isinstance(v, str) for v in val):
                res[prefix + key] = PayloadInterfaceStrictOneOf(value=val, type="keyword")
                continue

            if all(isinstance(v, int) for v in val):
                res[prefix + key] = PayloadInterfaceStrictOneOf1(value=val, type="integer")
                continue

            if all(isinstance(v, float) for v in val):
                res[prefix + key] = PayloadInterfaceStrictOneOf2(value=val, type="float")
                continue

            if all(isinstance(v, dict) and 'lon' in v and 'lat' in v for v in val):
                res[prefix + key] = PayloadInterfaceStrictOneOf3(
                    value=[GeoPoint(lat=v['lat'], lon=v['lon']) for v in val],
                    type="geo"
                )
                continue

        raise RuntimeError(f"Payload {key} have unsupported type {type(val)}")

    return res


def upload_batch(openapi_client: SyncApis, collection_name: str, batch) -> bool:
    ids_batch, vectors_batch, payload_batch = batch

    if payload_batch is not None:
        payload_batch = list(map(json_to_payload, payload_batch))

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

    @classmethod
    def process_payload(cls, payload: dict):
        return json_to_payload(payload, prefix='')

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
