import json
from typing import Any

from qdrant_client.http import models
from qdrant_client.http.api.collections_api import SyncCollectionsApi


class RecordingApiClient:
    def __init__(self) -> None:
        self.request_kwargs: dict[str, Any] | None = None

    def request(self, **kwargs: Any) -> models.InlineResponse2001:
        self.request_kwargs = kwargs
        return models.InlineResponse2001(result=True, status="ok", time=0.0)


def test_update_collection_serializes_sparse_vectors_config() -> None:
    api_client = RecordingApiClient()
    collections_api = SyncCollectionsApi(api_client)

    collections_api.update_collection(
        collection_name="text_documents",
        update_collection=models.UpdateCollection(
            sparse_vectors={
                "text-bm25": models.SparseVectorParams(
                    index=models.SparseIndexParams(on_disk=False),
                    modifier=models.Modifier.IDF,
                )
            }
        ),
    )

    assert api_client.request_kwargs is not None
    body = json.loads(api_client.request_kwargs["content"])

    assert "sparse_vectors_config" in body
    assert "sparse_vectors" not in body
    assert body["sparse_vectors_config"]["text-bm25"] == {
        "index": {"on_disk": False},
        "modifier": "idf",
    }
