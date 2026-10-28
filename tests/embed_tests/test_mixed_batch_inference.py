"""Named batches can mix precomputed vectors with objects requiring inference."""

from copy import deepcopy
from typing import Any
from unittest.mock import Mock

import pytest

from qdrant_client import AsyncQdrantClient, QdrantClient, models
from qdrant_client.embed.embedder import Embedder
from qdrant_client.embed.model_embedder import ModelEmbedder
from qdrant_client.fastembed_common import FastEmbedMisc

MODEL = "test-model"
EMBEDDINGS = {"first": [1.0, 2.0], "second": [3.0, 4.0]}
PRECOMPUTED = [9.0, 8.0]
ORDERS = [
    pytest.param([None, "first", "second"], id="precomputed-first"),
    pytest.param(["first", None, "second"], id="precomputed-between"),
    pytest.param(["first", "second", None], id="precomputed-last"),
    pytest.param(["first", "second"], id="all-documents"),
    pytest.param([None, None], id="all-precomputed"),
]


@pytest.fixture
def provider(monkeypatch: pytest.MonkeyPatch) -> Mock:
    """Replace the embedding provider; inspection, traversal and storage remain real."""
    provider = Mock(spec=Embedder)
    provider.is_supported_text_model.return_value = True
    for method in (
        "is_supported_sparse_model",
        "is_supported_late_interaction_text_model",
        "is_supported_image_model",
        "is_supported_late_interaction_multimodal_model",
    ):
        getattr(provider, method).return_value = False

    def embed(**kwargs: Any) -> list[list[float]]:
        """Give each document a distinct deterministic embedding without model downloads."""
        return [EMBEDDINGS[text].copy() for text in kwargs["texts"]]

    provider.embed.side_effect = embed
    monkeypatch.setattr("qdrant_client.embed.model_embedder.Embedder", Mock(return_value=provider))
    monkeypatch.setattr(FastEmbedMisc, "import_fastembed", Mock())
    return provider


def make_batch(order: list[str | None]) -> models.Batch:
    """Build a valid named batch containing documents and precomputed vectors."""
    return models.Batch(
        ids=list(range(len(order))),
        vectors={
            "text": [
                models.Document(text=text, model=MODEL) if text else PRECOMPUTED.copy()
                for text in order
            ]
        },
    )


def assert_embeddings(
    records: list[models.Record], order: list[str | None], provider: Mock
) -> None:
    """Each point retains its own vector and only documents are sent to the provider."""
    assert [record.id for record in records] == list(range(len(order)))
    assert [record.vector for record in records] == [
        {"text": EMBEDDINGS[text] if text else PRECOMPUTED} for text in order
    ]
    texts = [text for text in order if text]
    if texts:
        provider.embed.assert_called_once()
        assert provider.embed.call_args.kwargs["texts"] == texts
    else:
        provider.embed.assert_not_called()


@pytest.mark.parametrize("order", ORDERS)
def test_upsert_mixed_named_batch(order: list[str | None], provider: Mock) -> None:
    """Sync upsert embeds every document regardless of its position in the batch."""
    client = QdrantClient(":memory:")
    try:
        client.create_collection(
            "example",
            vectors_config={"text": models.VectorParams(size=2, distance=models.Distance.DOT)},
        )
        batch = make_batch(order)
        original = deepcopy(batch)
        client.upsert("example", batch)
        assert_embeddings(
            client.retrieve("example", batch.ids, with_vectors=True), order, provider
        )
        assert batch == original
    finally:
        client.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("order", ORDERS)
async def test_async_upsert_mixed_named_batch(order: list[str | None], provider: Mock) -> None:
    """The async client uses the same mixed-batch inference behavior."""
    client = AsyncQdrantClient(":memory:")
    try:
        await client.create_collection(
            "example",
            vectors_config={"text": models.VectorParams(size=2, distance=models.Distance.DOT)},
        )
        batch = make_batch(order)
        original = deepcopy(batch)
        await client.upsert("example", batch)
        assert_embeddings(
            await client.retrieve("example", batch.ids, with_vectors=True), order, provider
        )
        assert batch == original
    finally:
        await client.close()


@pytest.mark.parametrize(
    "vector",
    [[9.0, 8.0], [[9.0, 8.0], [7.0, 6.0]], models.SparseVector(indices=[7], values=[6.0])],
    ids=["dense", "multivector", "sparse"],
)
def test_mixed_batch_preserves_precomputed_vector_kinds(vector: Any, provider: Mock) -> None:
    """All precomputed vector kinds are preserved while later documents are embedded."""
    batch = models.Batch(
        ids=[0, 1], vectors={"text": [vector, models.Document(text="first", model=MODEL)]}
    )
    original = deepcopy(batch)
    (result,) = ModelEmbedder().embed_models_batch([batch])
    assert isinstance(result, models.Batch)
    assert result.vectors == {"text": [vector, EMBEDDINGS["first"]]}
    assert batch == original


def test_point_with_precomputed_and_inferred_named_vectors(provider: Mock) -> None:
    """A plain numeric vector is preserved when another named vector requires inference."""
    point = models.PointStruct(
        id=0,
        vector={
            "precomputed": PRECOMPUTED.copy(),
            "text": models.Document(text="first", model=MODEL),
        },
    )
    original = deepcopy(point)
    (result,) = ModelEmbedder().embed_models_batch([point])
    assert isinstance(result, models.PointStruct)
    assert result.vector == {"precomputed": PRECOMPUTED, "text": EMBEDDINGS["first"]}
    provider.embed.assert_called_once()
    assert provider.embed.call_args.kwargs["texts"] == ["first"]
    assert point == original
