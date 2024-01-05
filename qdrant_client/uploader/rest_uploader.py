import logging
from typing import Any, Generator, Iterable, List, Optional, Tuple, Union

import numpy as np

from qdrant_client.http import SyncApis
from qdrant_client.http.models import Batch, PointsList, PointStruct, ShardKeySelector
from qdrant_client.uploader.uploader import BaseUploader


def upload_batch(
    openapi_client: SyncApis,
    collection_name: str,
    batch: Union[Tuple, Batch],
    max_retries: int,
    wait: bool = False,
) -> bool:
    def send_batch(points_list: List[PointStruct], key: Optional[ShardKeySelector]) -> None:
        for attempt in range(max_retries):
            try:
                openapi_client.points_api.upsert_points(
                    collection_name=collection_name,
                    point_insert_operations=PointsList(points=points_list, shard_key=key),
                    wait=wait,
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
                id=idx,
                vector=(vector.tolist() if isinstance(vector, np.ndarray) else vector) or {},
                payload=payload,
            )
        )

    if points:
        send_batch(points, prev_shard_key)

    return True


class RestBatchUploader(BaseUploader):
    def __init__(
        self,
        uri: str,
        collection_name: str,
        max_retries: int,
        wait: bool = False,
        **kwargs: Any,
    ):
        self.collection_name = collection_name
        self.openapi_client: SyncApis = SyncApis(host=uri, **kwargs)
        self.max_retries = max_retries
        self._wait = wait

    @classmethod
    def start(
        cls,
        collection_name: Optional[str] = None,
        uri: str = "http://localhost:6333",
        max_retries: int = 3,
        **kwargs: Any,
    ) -> "RestBatchUploader":
        if not collection_name:
            raise RuntimeError("Collection name could not be empty")
        return cls(uri=uri, collection_name=collection_name, max_retries=max_retries, **kwargs)

    def process(self, items: Iterable[Any]) -> Generator[bool, None, None]:
        for batch in items:
            yield upload_batch(
                self.openapi_client,
                self.collection_name,
                batch,
                max_retries=self.max_retries,
                wait=self._wait,
            )
