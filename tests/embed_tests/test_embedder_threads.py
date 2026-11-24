import numpy as np
import pytest

import qdrant_client.embed.embedder as embedder_module
from qdrant_client import models
from qdrant_client.embed.embedder import Embedder
from qdrant_client.embed.model_embedder import ModelEmbedderWorker
from qdrant_client.fastembed_common import FastEmbedMisc


class RecordingEmbedding:
    """Record the FastEmbed constructor boundary without downloading a model."""

    def __init__(self, model_name: str, threads: int | None = None, **kwargs):
        self.threads = threads

    def embed(self, documents, **kwargs):
        return [np.array([1.0, 0.0]) for _ in documents]


@pytest.fixture(
    params=[
        ("get_or_init_model", "TextEmbedding", "is_supported_text_model"),
        ("get_or_init_sparse_model", "SparseTextEmbedding", "is_supported_sparse_model"),
        (
            "get_or_init_late_interaction_model",
            "LateInteractionTextEmbedding",
            "is_supported_late_interaction_text_model",
        ),
        (
            "get_or_init_late_interaction_multimodal_model",
            "LateInteractionMultimodalEmbedding",
            "is_supported_late_interaction_multimodal_model",
        ),
        ("get_or_init_image_model", "ImageEmbedding", "is_supported_image_model"),
    ],
    ids=["text", "sparse", "late-interaction", "multimodal", "image"],
)
def model_factory(request, monkeypatch):
    method, model_class, supported = request.param
    monkeypatch.setattr(embedder_module, model_class, RecordingEmbedding)
    monkeypatch.setattr(FastEmbedMisc, supported, lambda name: True)
    return method


@pytest.mark.parametrize(
    "default_threads, requested_threads, expected_threads",
    [(None, None, None), (None, 0, 0), (1, None, 1), (1, 0, 0), (1, 2, 2), (0, None, 0)],
)
def test_model_threads_override(
    model_factory, default_threads, requested_threads, expected_threads
):
    embedder = Embedder(threads=default_threads)
    model = getattr(embedder, model_factory)("test-model", threads=requested_threads)

    assert model.threads == expected_threads


def test_automatic_threads_use_a_separate_cached_model(model_factory):
    embedder = Embedder(threads=1)
    get_model = getattr(embedder, model_factory)
    default_model = get_model("test-model")
    automatic_model = get_model("test-model", threads=0)

    assert automatic_model is not default_model
    assert get_model("test-model", threads=0) is automatic_model
    assert get_model("test-model", threads=1) is default_model
    assert get_model("test-model") is default_model


@pytest.mark.parametrize("threads", [None, 0, 2])
def test_inference_worker_preserves_document_thread_option(monkeypatch, threads):
    monkeypatch.setattr(FastEmbedMisc, "is_installed", lambda: True)
    for kind in [
        "text",
        "sparse",
        "late_interaction_text",
        "late_interaction_multimodal",
        "image",
    ]:
        monkeypatch.setattr(FastEmbedMisc, f"is_supported_{kind}_model", lambda name: False)
    monkeypatch.setattr(FastEmbedMisc, "is_supported_text_model", lambda name: True)
    monkeypatch.setattr(embedder_module, "TextEmbedding", RecordingEmbedding)

    worker = ModelEmbedderWorker.start(batch_size=8)
    point = models.PointStruct(
        id=1,
        vector=models.Document(
            text="retrieval example", model="test-model", options={"threads": threads}
        ),
    )

    [(batch_id, embedded)] = list(worker.process([(0, [point])]))

    assert batch_id == 0
    assert embedded[0].vector == [1.0, 0.0]
    assert point.vector.options == {"threads": threads}
    [instance] = worker.model_embedder.embedder.embedding_models["test-model"]
    assert instance.model.threads == (1 if threads is None else threads)
