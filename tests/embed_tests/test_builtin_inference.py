from typing import Any, Optional

import pytest

from qdrant_client import QdrantClient, models
from qdrant_client.fastembed_common import FastEmbedMisc

COLLECTION_NAME = "inference_collection"
MODEL_NAME = "Qdrant/Bm25"
DEFAULT_VECTOR_NAME = "bm25"


def prepare_collection(
    client: QdrantClient,
    collection_name: str,
    vectors_config: Optional[dict[str, Any]] = None,
    sparse_vectors_config: Optional[dict[str, Any]] = None,
) -> None:
    if client.collection_exists(collection_name):
        client.delete_collection(collection_name)
    config = (
        {DEFAULT_VECTOR_NAME: models.SparseVectorParams(modifier=models.Modifier.IDF)}
        if sparse_vectors_config is None
        else sparse_vectors_config
    )
    client.create_collection(
        collection_name, vectors_config=vectors_config or {}, sparse_vectors_config=config
    )


def test_bm25_inference():
    remote_client = QdrantClient()
    prepare_collection(remote_client, COLLECTION_NAME)
    local_client = QdrantClient(":memory:")
    prepare_collection(local_client, COLLECTION_NAME)

    remote_client.upsert(
        COLLECTION_NAME,
        points=[
            models.PointStruct(
                id=1,
                vector={DEFAULT_VECTOR_NAME: models.Document(text="good text", model=MODEL_NAME)},
            )
        ],
    )
    assert remote_client.count(collection_name=COLLECTION_NAME, exact=True).count == 1

    # not calling is_installed() on purpose, since it changes `IS_INSTALLED` and might conceal a bug
    if FastEmbedMisc.IS_INSTALLED:
        local_client.upsert(
            COLLECTION_NAME,
            points=[
                models.PointStruct(
                    id=1,
                    vector={
                        DEFAULT_VECTOR_NAME: models.Document(text="good text", model=MODEL_NAME)
                    },
                )
            ],
        )
        assert local_client.count(collection_name=COLLECTION_NAME, exact=True).count == 1
    else:
        # inference is done via builtin Qdrant bm25 in remote client, and is not available in local mode
        with pytest.raises(ImportError):
            local_client.upsert(
                COLLECTION_NAME,
                points=[
                    models.PointStruct(
                        id=1,
                        vector={
                            DEFAULT_VECTOR_NAME: models.Document(text="bad text", model=MODEL_NAME)
                        },
                    )
                ],
            )


def test_bm25_inference_server_version(monkeypatch):
    server_version = "1.11.0"

    def patched_get_server_version(*args, **kwargs):
        return server_version

    monkeypatch.setattr(
        "qdrant_client.qdrant_remote.get_server_version", patched_get_server_version
    )
    remote_client = QdrantClient()
    prepare_collection(remote_client, COLLECTION_NAME)

    if FastEmbedMisc.IS_INSTALLED:
        # inference is done via fastembed in both remote and local client
        remote_client.upsert(
            COLLECTION_NAME,
            points=[
                models.PointStruct(
                    id=1,
                    vector={
                        DEFAULT_VECTOR_NAME: models.Document(text="good text", model=MODEL_NAME)
                    },
                )
            ],
        )

        assert remote_client.count(collection_name=COLLECTION_NAME, exact=True).count == 1
    else:
        with pytest.raises(ImportError):
            remote_client.upsert(
                COLLECTION_NAME,
                points=[
                    models.PointStruct(
                        id=1,
                        vector={
                            DEFAULT_VECTOR_NAME: models.Document(text="bad text", model=MODEL_NAME)
                        },
                    )
                ],
            )

    server_version = None
    monkeypatch.setattr("qdrant_client.qdrant_remote.get_server_version", lambda: server_version)
    remote_client = QdrantClient()
    prepare_collection(remote_client, COLLECTION_NAME)

    remote_client.upsert(
        COLLECTION_NAME,
        points=[
            models.PointStruct(
                id=1,
                vector={DEFAULT_VECTOR_NAME: models.Document(text="good text", model=MODEL_NAME)},
            )
        ],
    )
    assert remote_client.count(collection_name=COLLECTION_NAME, exact=True).count == 1

    server_version = "1.15.3"
    monkeypatch.setattr("qdrant_client.qdrant_remote.get_server_version", lambda: server_version)
    remote_client = QdrantClient()
    prepare_collection(remote_client, COLLECTION_NAME)
    remote_client.upsert(
        COLLECTION_NAME,
        points=[
            models.PointStruct(
                id=1,
                vector={DEFAULT_VECTOR_NAME: models.Document(text="good text", model=MODEL_NAME)},
            )
        ],
    )
    assert remote_client.count(collection_name=COLLECTION_NAME, exact=True).count == 1


def test_not_supported_models():
    if FastEmbedMisc.is_installed():
        pytest.skip(reason="testing builtin inference")

    remote_client = QdrantClient()
    prepare_collection(
        remote_client,
        COLLECTION_NAME,
        vectors_config={
            "all-minilm": models.VectorParams(size=384, distance=models.Distance.COSINE)
        },
    )

    with pytest.raises(ValueError):
        remote_client.upsert(
            COLLECTION_NAME,
            points=[
                models.PointStruct(
                    id=1,
                    vector={
                        "all-minilm": models.Document(
                            text="bad text", model="sentence-transformers/all-MiniLM-L6-v2"
                        ),
                        "bm25": models.Document(text="good text", model=MODEL_NAME),
                    },
                )
            ],
        )
