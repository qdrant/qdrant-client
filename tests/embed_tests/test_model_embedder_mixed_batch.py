from contextlib import closing
from copy import deepcopy
from unittest.mock import MagicMock

import pytest

from qdrant_client import QdrantClient, models
from qdrant_client.embed.model_embedder import ModelEmbedder

TEXT_MODEL = "BAAI/bge-small-en"
SPARSE_MODEL = "Qdrant/bm25"


def _make_mock_embedder(parser=None) -> ModelEmbedder:
    embedder = ModelEmbedder(parser=parser, is_local_mode=False)
    embedder.embedder.is_supported_text_model = MagicMock(
        side_effect=lambda name: name == TEXT_MODEL
    )
    embedder.embedder.is_supported_sparse_model = MagicMock(
        side_effect=lambda name: name == SPARSE_MODEL
    )

    def _fake_embed(model_name, texts=None, **_kwargs):
        assert texts is not None
        if model_name == SPARSE_MODEL:
            return [
                models.SparseVector(indices=[idx + 1], values=[float(len(text))])
                for idx, text in enumerate(texts)
            ]
        return [[float(len(text)), float(idx + 1)] for idx, text in enumerate(texts)]

    embedder.embedder.embed = MagicMock(side_effect=_fake_embed)
    return embedder


@pytest.mark.parametrize(
    ("vectors", "expected"),
    [
        (
            [[0.1, 0.2], models.Document(text="hello", model=TEXT_MODEL)],
            [[0.1, 0.2], [5.0, 1.0]],
        ),
        (
            [
                models.Document(text="hi", model=TEXT_MODEL),
                [0.1, 0.2],
                models.Document(text="world", model=TEXT_MODEL),
            ],
            [[2.0, 1.0], [0.1, 0.2], [5.0, 2.0]],
        ),
        (
            [models.Document(text="hello", model=TEXT_MODEL), [0.1, 0.2]],
            [[5.0, 1.0], [0.1, 0.2]],
        ),
    ],
)
def test_embed_models_mixed_named_dense_batch(vectors, expected):
    embedder = _make_mock_embedder()
    batch = models.Batch(ids=list(range(1, len(vectors) + 1)), vectors={"dense": vectors})
    original = deepcopy(batch)

    embedded = list(embedder.embed_models(batch))

    assert len(embedded) == 1
    assert embedded[0].vectors == {"dense": expected}
    assert batch == original


def test_embed_models_mixed_named_sparse_and_multivector_batch():
    embedder = _make_mock_embedder()
    existing_sparse = models.SparseVector(indices=[0], values=[1.5])
    existing_multi = [[0.1, 0.2], [0.3, 0.4]]
    batch = models.Batch(
        ids=[1, 2],
        vectors={
            "sparse": [existing_sparse, models.Document(text="abc", model=SPARSE_MODEL)],
            "multi": [existing_multi, models.Document(text="abcd", model=TEXT_MODEL)],
        },
    )

    embedded = list(embedder.embed_models(batch))

    assert len(embedded) == 1
    assert embedded[0].vectors == {
        "sparse": [existing_sparse, models.SparseVector(indices=[1], values=[3.0])],
        "multi": [existing_multi, [4.0, 1.0]],
    }


def test_embed_models_pure_numeric_vectors_not_embedded():
    embedder = _make_mock_embedder()
    batch = models.Batch(
        ids=[1, 2],
        vectors={
            "dense": [[0.1, 0.2], [0.3, 0.4]],
            "multi": [[[0.1, 0.2], [0.3, 0.4]], [[0.5, 0.6]]],
        },
    )

    embedded = list(embedder.embed_models(batch))

    assert embedded == [batch]
    embedder.embedder.embed.assert_not_called()


def test_local_client_upsert_and_batch_update_mixed_named_vector_batch():
    with closing(QdrantClient(":memory:")) as client:
        client.create_collection(
            "mixed",
            vectors_config={"dense": models.VectorParams(size=2, distance=models.Distance.DOT)},
        )
        client._model_embedder = _make_mock_embedder(parser=client._inference_inspector.parser)

        batch = models.Batch(
            ids=[1, 2, 3],
            vectors={
                "dense": [
                    [0.1, 0.2],
                    models.Document(text="alpha", model=TEXT_MODEL),
                    models.Document(text="beta", model=TEXT_MODEL),
                ]
            },
        )
        client.upsert("mixed", points=batch)

        records = {
            record.id: record.vector
            for record in client.retrieve("mixed", ids=[1, 2, 3], with_vectors=True)
        }
        assert records[1] == {"dense": pytest.approx([0.1, 0.2])}
        assert records[2] == {"dense": pytest.approx([5.0, 1.0])}
        assert records[3] == {"dense": pytest.approx([4.0, 2.0])}

        op_batch = models.Batch(
            ids=[4, 5],
            vectors={
                "dense": [
                    [0.7, 0.8],
                    models.Document(text="gamma", model=TEXT_MODEL),
                ]
            },
        )
        client.batch_update_points(
            "mixed",
            update_operations=[
                models.UpsertOperation(upsert=models.PointsBatch(batch=op_batch)),
            ],
        )
        op_records = {
            record.id: record.vector
            for record in client.retrieve("mixed", ids=[4, 5], with_vectors=True)
        }
        assert op_records[4] == {"dense": pytest.approx([0.7, 0.8])}
        assert op_records[5] == {"dense": pytest.approx([5.0, 1.0])}
