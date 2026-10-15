from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from qdrant_client import AsyncQdrantClient, QdrantClient, models
from qdrant_client.embed.embedder import Embedder
from qdrant_client.fastembed_common import FastEmbedMisc


pytestmark = pytest.mark.fastembed

DENSE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TEXTS = ["alpha", "beta", "gamma", "delta", "epsilon"]
OPTIONS = {"threads": 1, "local_files_only": True}
COLLECTION = "query_batch_size"


@pytest.fixture(autouse=True)
def require_fastembed():
    pytest.importorskip("fastembed")


class RecordingModel:
    def __init__(self, embeddings):
        self.embeddings = embeddings
        self.calls = []

    def embed(self, *, documents, **kwargs):
        self.calls.append(("embed", list(documents), kwargs))
        return iter(self.embeddings)

    def query_embed(self, *, query, **kwargs):
        self.calls.append(("query_embed", list(query), kwargs))
        return iter(self.embeddings)


@pytest.mark.parametrize(
    "family,model_name,initializer",
    [
        ("dense", DENSE_MODEL, "get_or_init_model"),
        ("sparse", "prithivida/Splade_PP_en_v1", "get_or_init_sparse_model"),
        (
            "late_interaction",
            "answerdotai/answerai-colbert-small-v1",
            "get_or_init_late_interaction_model",
        ),
    ],
)
@pytest.mark.parametrize("is_query", [False, True], ids=["document", "query"])
@pytest.mark.parametrize("batch_size", [1, 2])
def test_text_batch_size_forwarding(family, model_name, initializer, is_query, batch_size):
    if family == "sparse":
        embeddings = [
            SimpleNamespace(
                indices=np.asarray([i, i + 10], dtype=np.int64),
                values=np.asarray([i + 0.25, i + 0.5], dtype=np.float32),
            )
            for i in range(len(TEXTS))
        ]
        expected = [
            models.SparseVector(
                indices=embedding.indices.tolist(), values=embedding.values.tolist()
            )
            for embedding in embeddings
        ]
    else:
        embeddings = [
            np.asarray(
                [i + 1.0, i + 0.5]
                if family == "dense"
                else [[i + 1.0, i + 0.5], [i + 0.25, i + 0.75]],
                dtype=np.float32,
            )
            for i in range(len(TEXTS))
        ]
        expected = [embedding.tolist() for embedding in embeddings]

    backend = RecordingModel(embeddings)
    embedder = Embedder()
    texts = TEXTS.copy()
    options = deepcopy(OPTIONS)
    with patch.object(embedder, initializer, return_value=backend) as initialize:
        actual = embedder.embed(
            texts=texts,
            model_name=model_name,
            options=options,
            is_query=is_query,
            batch_size=batch_size,
        )

    assert actual == expected
    assert texts == TEXTS
    assert options == OPTIONS
    initialize.assert_called_once_with(model_name=model_name, **OPTIONS)
    assert backend.calls == [
        ("query_embed" if is_query else "embed", TEXTS, {"batch_size": batch_size})
    ]


def recommendation_data():
    dimension, distance = FastEmbedMisc.list_text_models()[DENSE_MODEL]
    vectors = []
    for i in range(len(TEXTS)):
        vector = np.zeros(dimension, dtype=np.float32)
        vector[:2] = [i + 1.0, 1.0]
        vectors.append(vector)
    points = [models.PointStruct(id=i, vector=vector.tolist()) for i, vector in enumerate(vectors)]
    query = models.RecommendQuery(
        recommend=models.RecommendInput(
            positive=[
                models.Document(text=text, model=DENSE_MODEL, options=OPTIONS) for text in TEXTS
            ]
        )
    )
    explicit_query = models.RecommendQuery(
        recommend=models.RecommendInput(positive=[vector.tolist() for vector in vectors])
    )
    return (
        models.VectorParams(size=dimension, distance=distance),
        points,
        query,
        explicit_query,
        vectors,
    )


@pytest.mark.parametrize("batch_size", [2, None], ids=["configured", "default"])
def test_nested_query_batch_size_sync(batch_size):
    client = QdrantClient(
        ":memory:",
        **({"local_inference_batch_size": batch_size} if batch_size is not None else {}),
    )
    vector_params, points, query, explicit_query, vectors = recommendation_data()
    original_query = deepcopy(query)
    backend = RecordingModel(vectors)
    try:
        client.create_collection(COLLECTION, vectors_config=vector_params)
        client.upsert(COLLECTION, points)
        expected = client.query_points(COLLECTION, query=explicit_query, limit=len(points))
        with patch.object(
            client._model_embedder.embedder, "get_or_init_model", return_value=backend
        ) as initialize:
            actual = client.query_points(COLLECTION, query=query, limit=len(points))

        assert actual == expected
        assert query == original_query
        initialize.assert_called_once_with(model_name=DENSE_MODEL, **OPTIONS)
        assert backend.calls == [
            ("query_embed", TEXTS, {"batch_size": batch_size or client.DEFAULT_BATCH_SIZE})
        ]
    finally:
        client.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("batch_size", [2, None], ids=["configured", "default"])
async def test_nested_query_batch_size_async(batch_size):
    client = AsyncQdrantClient(
        ":memory:",
        **({"local_inference_batch_size": batch_size} if batch_size is not None else {}),
    )
    vector_params, points, query, explicit_query, vectors = recommendation_data()
    original_query = deepcopy(query)
    backend = RecordingModel(vectors)
    try:
        await client.create_collection(COLLECTION, vectors_config=vector_params)
        await client.upsert(COLLECTION, points)
        expected = await client.query_points(COLLECTION, query=explicit_query, limit=len(points))
        with patch.object(
            client._model_embedder.embedder, "get_or_init_model", return_value=backend
        ) as initialize:
            actual = await client.query_points(COLLECTION, query=query, limit=len(points))

        assert actual == expected
        assert query == original_query
        initialize.assert_called_once_with(model_name=DENSE_MODEL, **OPTIONS)
        assert backend.calls == [
            ("query_embed", TEXTS, {"batch_size": batch_size or client.DEFAULT_BATCH_SIZE})
        ]
    finally:
        await client.close()


def test_core_bm25_query_passthrough():
    embedder = Embedder(use_core_bm25=True)
    with patch.object(embedder, "get_or_init_sparse_model") as initialize:
        actual = embedder.embed(
            model_name="Qdrant/bm25", texts=TEXTS, options=OPTIONS, is_query=True, batch_size=2
        )

    assert actual == [
        models.Document(text=text, model="Qdrant/bm25", options=OPTIONS) for text in TEXTS
    ]
    initialize.assert_not_called()
