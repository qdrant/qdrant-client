from itertools import count
from time import sleep
from typing import Any, Generator, Iterable, Optional, Union
from uuid import uuid4

from qdrant_client import grpc as grpc
from qdrant_client.common.client_exceptions import ResourceExhaustedResponse
from qdrant_client.connection import get_channel
from qdrant_client.conversions.conversion import RestToGrpc, payload_to_grpc
from qdrant_client.grpc import PointId, PointsStub, PointStruct
from qdrant_client.http.models import Batch, ShardKeySelector
from qdrant_client.uploader.uploader import BaseUploader
from qdrant_client.common.client_warnings import show_warning


def upload_batch_grpc(
    points_client: PointsStub,
    collection_name: str,
    batch: Union[Batch, tuple],
    max_retries: int,
    shard_key_selector: Optional[ShardKeySelector],
    wait: bool = False,
    timeout: Optional[int] = None,
) -> bool:
    ids_batch, vectors_batch, payload_batch = batch

    ids_batch = (PointId(uuid=str(uuid4())) for _ in count()) if ids_batch is None else ids_batch
    payload_batch = (None for _ in count()) if payload_batch is None else payload_batch

    points = [
        PointStruct(
            id=RestToGrpc.convert_extended_point_id(idx) if not isinstance(idx, PointId) else idx,
            vectors=RestToGrpc.convert_vector_struct(vector),
            payload=payload_to_grpc(payload or {}),
        )
        for idx, vector, payload in zip(ids_batch, vectors_batch, payload_batch)
    ]

    attempt = 0
    while attempt < max_retries:
        try:
            points_client.Upsert(
                grpc.UpsertPoints(
                    collection_name=collection_name,
                    points=points,
                    wait=wait,
                    shard_key_selector=RestToGrpc.convert_shard_key_selector(shard_key_selector)
                    if shard_key_selector is not None
                    else None,
                ),
                timeout=timeout,
            )
            break
        except ResourceExhaustedResponse as ex:
            show_warning(
                message=f"Batch upload failed due to rate limit. Waiting for {ex.retry_after_s} seconds before retrying...",
                category=UserWarning,
                stacklevel=8,
            )
            sleep(ex.retry_after_s)

        except Exception as e:
            show_warning(
                message=f"Batch upload failed {attempt + 1} times. Retrying...",
                category=UserWarning,
                stacklevel=8,
            )

            if attempt == max_retries - 1:
                raise e

            attempt += 1
    return True


class GrpcBatchUploader(BaseUploader):
    def __init__(
        self,
        host: str,
        port: int,
        collection_name: str,
        max_retries: int,
        wait: bool = False,
        shard_key_selector: Optional[ShardKeySelector] = None,
        **kwargs: Any,
    ):
        self.collection_name = collection_name
        self._host = host
        self._port = port
        self.max_retries = max_retries
        self._kwargs = kwargs
        self._wait = wait
        self._shard_key_selector = shard_key_selector
        self._timeout = kwargs.pop("timeout", None)

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
                shard_key_selector=self._shard_key_selector,
                max_retries=self.max_retries,
                wait=self._wait,
                timeout=self._timeout,
            )

    def process(self, items: Iterable[Any]) -> Iterable[bool]:
        yield from self.process_upload(items)
