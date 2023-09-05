import logging
from itertools import count
from typing import Any, Generator, Iterable, Optional, Tuple, Union

import numpy as np

from qdrant_client.http import SyncApis
from qdrant_client.http.models import Batch, PointsList, PointStruct
from qdrant_client.uploader.uploader import BaseUploader


def upload_batch(
    openapi_client: SyncApis,
    collection_name: str,
    batch: Union[Tuple, Batch],
    max_retries: int,
    wait: bool = False,
) -> bool:
    ids_batch, vectors_batch, payload_batch = batch

    if payload_batch is not None:
        payload_batch = list(payload_batch)
    else:
        payload_batch = (None for _ in count())

    points = [
        PointStruct(
            id=idx,
            vector=(vector.tolist() if isinstance(vector, np.ndarray) else vector) or {},
            payload=payload,
        )
        for idx, vector, payload in zip(ids_batch, vectors_batch, payload_batch)
    ]

    for attempt in range(max_retries):
        try:
            openapi_client.points_api.upsert_points(
                collection_name=collection_name,
                point_insert_operations=PointsList(points=points),
                wait=wait,
            )
        except Exception as e:
            logging.warning(f"Batch upload failed {attempt + 1} times. Retrying...")

            if attempt == max_retries - 1:
                raise e
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
                self.max_retries,
                self._wait,
            )
