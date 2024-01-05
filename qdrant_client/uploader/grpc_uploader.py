import logging
from typing import Any, Generator, Iterable, Optional, Tuple, Union

from qdrant_client import grpc as grpc
from qdrant_client.connection import get_channel
from qdrant_client.conversions.conversion import RestToGrpc, payload_to_grpc
from qdrant_client.grpc import PointId, PointsStub, PointStruct
from qdrant_client.http.models import Batch
from qdrant_client.uploader.uploader import BaseUploader


def upload_batch_grpc(
    points_client: PointsStub,
    collection_name: str,
    batch: Union[Batch, Tuple],
    max_retries: int,
    wait: bool = False,
) -> bool:
    def send_batch(points_list, key):
        for attempt in range(max_retries):
            try:
                points_client.Upsert(
                    grpc.UpsertPoints(
                        collection_name=collection_name,
                        points=points_list,
                        wait=wait,
                        shard_key_selector=RestToGrpc.convert_shard_key_selector(key)
                        if key is not None
                        else None,
                    )
                )
            except Exception as e:
                logging.warning(f"Batch upload failed {attempt + 1} times. Retrying...")

                if attempt == max_retries - 1:
                    raise e

    ids_batch, vectors_batch, payload_batch, shard_key_batch = batch

    warning_emitted = False
    prev_shard_key = None
    points = []
    for i, (idx, vector, payload, shard_key) in enumerate(
        zip(ids_batch, vectors_batch, payload_batch, shard_key_batch)
    ):
        if i == 0:
            prev_shard_key = shard_key

        if prev_shard_key != shard_key:
            if not warning_emitted:
                logging.warning(
                    "Batch contains points with different shard keys. It can affect the performance."
                )
                warning_emitted = True

            send_batch(points, prev_shard_key)
            points = []
            prev_shard_key = shard_key

        points.append(
            PointStruct(
                id=RestToGrpc.convert_extended_point_id(idx)
                if not isinstance(idx, PointId)
                else idx,
                vectors=RestToGrpc.convert_vector_struct(vector),
                payload=payload_to_grpc(payload or {}),
            )
        )

    if points:
        send_batch(points, prev_shard_key)
    return True


class GrpcBatchUploader(BaseUploader):
    def __init__(
        self,
        host: str,
        port: int,
        collection_name: str,
        max_retries: int,
        wait: bool = False,
        **kwargs: Any,
    ):
        self.collection_name = collection_name
        self._host = host
        self._port = port
        self.max_retries = max_retries
        self._kwargs = kwargs
        self._wait = wait

    @classmethod
    def start(
        cls,
        collection_name: Optional[str] = None,
        host: str = "localhost",
        port: int = 6334,
        max_retries: int = 3,
        **kwargs: Any,
    ) -> "GrpcBatchUploader":
        if not collection_name:
            raise RuntimeError("Collection name could not be empty")

        return cls(
            host=host,
            port=port,
            collection_name=collection_name,
            max_retries=max_retries,
            **kwargs,
        )

    def process_upload(self, items: Iterable[Any]) -> Generator[bool, None, None]:
        channel = get_channel(host=self._host, port=self._port, **self._kwargs)
        points_client = PointsStub(channel)
        for batch in items:
            yield upload_batch_grpc(
                points_client,
                self.collection_name,
                batch,
                max_retries=self.max_retries,
                wait=self._wait,
            )

    def process(self, items: Iterable[Any]) -> Generator[bool, None, None]:
        yield from self.process_upload(items)
