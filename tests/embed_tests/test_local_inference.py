from typing import Optional
from pathlib import Path
from unittest.mock import patch
from functools import lru_cache
import contextlib

import numpy as np
import pytest

import qdrant_client.embed.embedder
from qdrant_client import QdrantClient, models
from qdrant_client.client_base import QdrantBase
from qdrant_client.qdrant_fastembed import IDF_EMBEDDING_MODELS
from qdrant_client.fastembed_common import (
    TextEmbedding,
    SparseTextEmbedding,
    LateInteractionTextEmbedding,
    ImageEmbedding,
)


COLLECTION_NAME = "inference_collection"
DENSE_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DENSE_DIM = 384
SPARSE_MODEL_NAME = "Qdrant/bm42-all-minilm-l6-v2-attentions"
COLBERT_MODEL_NAME = "answerdotai/answerai-colbert-small-v1"
COLBERT_DIM = 96
DENSE_IMAGE_MODEL_NAME = "Qdrant/resnet50-onnx"
DENSE_IMAGE_DIM = 2048

TEST_IMAGE_PATH = Path(__file__).parent / "misc" / "image.jpeg"


@lru_cache
def _cached_text_embedding(model_name, *args, **kwargs):
    return TextEmbedding(model_name=model_name, *args, **kwargs)


@lru_cache
def _cached_sparse_text_embedding(model_name, *args, **kwargs):
    return SparseTextEmbedding(model_name=model_name, *args, **kwargs)


@lru_cache
def _cached_late_interaction_text_embedding(model_name, *args, **kwargs):
    return LateInteractionTextEmbedding(model_name=model_name, *args, **kwargs)


@lru_cache
def _cached_image_embedding(model_name, *args, **kwargs):
    return ImageEmbedding(model_name=model_name, *args, **kwargs)


@pytest.fixture(scope="session")
def cached_embeddings():
    with contextlib.ExitStack() as stack:
        stack.enter_context(
            patch(
                "qdrant_client.embed.embedder.TextEmbedding",
                side_effect=_cached_text_embedding,
            )
        )
        stack.enter_context(
            patch(
                "qdrant_client.embed.embedder.SparseTextEmbedding",
                side_effect=_cached_sparse_text_embedding,
            )
        )
        stack.enter_context(
            patch(
                "qdrant_client.embed.embedder.LateInteractionTextEmbedding",
                side_effect=_cached_late_interaction_text_embedding,
            )
        )
        stack.enter_context(
            patch(
                "qdrant_client.embed.embedder.ImageEmbedding",
                side_effect=_cached_image_embedding,
            )
        )
        yield


def arg_interceptor(func, kwarg_storage):
    kwarg_storage.clear()

    def wrapper(**kwargs):
        kwarg_storage.update(kwargs)
        return func(**kwargs)

    return wrapper


def populate_dense_collection(
    client: QdrantBase,
    points: list[models.PointStruct],
    vector_name: Optional[str] = None,
    collection_name: str = COLLECTION_NAME,
    recreate: bool = True,
) -> None:
    if recreate:
        if client.collection_exists(collection_name):
            client.delete_collection(collection_name)
        vector_params = models.VectorParams(size=DENSE_DIM, distance=models.Distance.COSINE)
        vectors_config = {vector_name: vector_params} if vector_name else vector_params
        client.create_collection(collection_name, vectors_config=vectors_config)
    client.upsert(collection_name, points)


def populate_sparse_collection(
    client: QdrantBase,
    points: list[models.PointStruct],
    vector_name: str,
    collection_name: str = COLLECTION_NAME,
    recreate: bool = True,
    model_name: str = SPARSE_MODEL_NAME,
) -> None:
    if recreate:
        if client.collection_exists(collection_name):
            client.delete_collection(collection_name)
        sparse_vector_params = models.SparseVectorParams(
            modifier=(
                models.Modifier.IDF if model_name in IDF_EMBEDDING_MODELS else models.Modifier.NONE
            )
        )
        sparse_vectors_config = {vector_name: sparse_vector_params}
        client.create_collection(
            collection_name, vectors_config={}, sparse_vectors_config=sparse_vectors_config
        )
    client.upsert(collection_name, points)


def test_upsert(cached_embeddings):
    local_client = QdrantClient(":memory:")
    if not local_client._FASTEMBED_INSTALLED:
        pytest.skip("FastEmbed is not installed, skipping")
    local_kwargs = {}
    local_client._client.upsert = arg_interceptor(local_client._client.upsert, local_kwargs)

    dense_doc_1 = models.Document(text="hello world", model=DENSE_MODEL_NAME)
    dense_doc_2 = models.Document(text="bye world", model=DENSE_MODEL_NAME)
    sparse_doc_1 = models.Document(text="hello world", model=SPARSE_MODEL_NAME)
    sparse_doc_2 = models.Document(text="bye world", model=SPARSE_MODEL_NAME)
    multi_doc_1 = models.Document(text="hello world", model=COLBERT_MODEL_NAME)
    multi_doc_2 = models.Document(text="bye world", model=COLBERT_MODEL_NAME)
    dense_image_1 = models.Image(image=TEST_IMAGE_PATH, model=DENSE_IMAGE_MODEL_NAME)
    dense_image_2 = models.Image(image=TEST_IMAGE_PATH, model=DENSE_IMAGE_MODEL_NAME)

    # region dense unnamed
    points = [
        models.PointStruct(id=1, vector=dense_doc_1),
        models.PointStruct(id=2, vector=dense_doc_2),
    ]
    populate_dense_collection(local_client, points)

    vec_points = local_kwargs["points"]
    assert all([isinstance(vec_point.vector, list) for vec_point in vec_points])

    batch = models.Batch(ids=[1, 2], vectors=[dense_doc_1, dense_doc_2])
    local_client.upsert(COLLECTION_NAME, batch)
    batch = local_kwargs["points"]
    assert all([isinstance(vector, list) for vector in batch.vectors])

    local_client.delete_collection(COLLECTION_NAME)
    # endregion

    # region named vectors
    vectors_config = {
        "text": models.VectorParams(size=DENSE_DIM, distance=models.Distance.COSINE),
        "multi-text": models.VectorParams(
            size=COLBERT_DIM,
            distance=models.Distance.COSINE,
            multivector_config=models.MultiVectorConfig(
                comparator=models.MultiVectorComparator.MAX_SIM
            ),
        ),
        "image": models.VectorParams(size=DENSE_IMAGE_DIM, distance=models.Distance.COSINE),
    }
    sparse_vectors_config = {
        "sparse-text": models.SparseVectorParams(modifier=models.Modifier.IDF)
    }
    local_client.create_collection(
        COLLECTION_NAME,
        vectors_config=vectors_config,
        sparse_vectors_config=sparse_vectors_config,
    )
    points = [
        models.PointStruct(
            id=1,
            vector={
                "text": dense_doc_1,
                "multi-text": multi_doc_1,
                "sparse-text": sparse_doc_1,
                "image": dense_image_1,
            },
        ),
        models.PointStruct(
            id=2,
            vector={
                "text": dense_doc_2,
                "multi-text": multi_doc_2,
                "sparse-text": sparse_doc_2,
                "image": dense_image_2,
            },
        ),
    ]
    local_client.upsert(COLLECTION_NAME, points)

    vec_points = local_kwargs["points"]
    for vec_point in vec_points:
        assert isinstance(vec_point.vector, dict)
        assert isinstance(vec_point.vector["text"], list)
        assert isinstance(vec_point.vector["multi-text"], list)
        assert isinstance(vec_point.vector["sparse-text"], models.SparseVector)
        assert isinstance(vec_point.vector["image"], list)

    batch = models.Batch(
        ids=[1, 2],
        vectors={
            "text": [dense_doc_1, dense_doc_2],
            "multi-text": [multi_doc_1, multi_doc_2],
            "sparse-text": [sparse_doc_1, sparse_doc_2],
            "image": [dense_image_1, dense_image_2],
        },
    )
    local_client.upsert(COLLECTION_NAME, batch)

    batch = local_kwargs["points"]
    vectors = batch.vectors
    assert isinstance(vectors, dict)
    assert all([isinstance(vector, list) for vector in vectors["text"]])
    assert all([isinstance(vector, list) for vector in vectors["multi-text"]])
    assert all([isinstance(vector, list) for vector in vectors["image"]])
    assert all([isinstance(vector, models.SparseVector) for vector in vectors["sparse-text"]])

    local_client.delete_collection(COLLECTION_NAME)
    # endregion


def test_upload(cached_embeddings):
    def recreate_collection(client, collection_name):
        if client.collection_exists(collection_name):
            client.delete_collection(collection_name)
        vector_params = {
            "text": models.VectorParams(size=DENSE_DIM, distance=models.Distance.COSINE),
            "image": models.VectorParams(size=DENSE_IMAGE_DIM, distance=models.Distance.COSINE),
        }
        client.create_collection(
            collection_name,
            vectors_config=vector_params,
            sparse_vectors_config={
                "sparse-text": models.SparseVectorParams(modifier=models.Modifier.IDF)
            },
        )

    local_client = QdrantClient(":memory:")
    if not local_client._FASTEMBED_INSTALLED:
        pytest.skip("FastEmbed is not installed, skipping")
    remote_client = QdrantClient()

    dense_doc_1 = models.Document(text="hello world", model=DENSE_MODEL_NAME)
    dense_doc_2 = models.Document(text="bye world", model=DENSE_MODEL_NAME)
    dense_doc_3 = models.Document(text="world world", model=DENSE_MODEL_NAME)

    sparse_doc_1 = models.Document(text="hello world", model=SPARSE_MODEL_NAME)
    sparse_doc_2 = models.Document(text="bye world", model=SPARSE_MODEL_NAME)
    sparse_doc_3 = models.Document(text="world world", model=SPARSE_MODEL_NAME)

    dense_image_1 = models.Image(image=TEST_IMAGE_PATH, model=DENSE_IMAGE_MODEL_NAME)
    dense_image_2 = models.Image(image=TEST_IMAGE_PATH, model=DENSE_IMAGE_MODEL_NAME)
    dense_image_3 = models.Image(image=TEST_IMAGE_PATH, model=DENSE_IMAGE_MODEL_NAME)

    points = [
        models.PointStruct(
            id=1, vector={"text": dense_doc_1, "image": dense_image_1, "sparse-text": sparse_doc_1}
        ),
        models.PointStruct(
            id=2, vector={"text": dense_doc_2, "image": dense_image_2, "sparse-text": sparse_doc_2}
        ),
        models.PointStruct(
            id=3, vector={"text": dense_doc_3, "image": dense_image_3, "sparse-text": sparse_doc_3}
        ),
    ]

    recreate_collection(remote_client, COLLECTION_NAME)

    remote_client.upload_points(COLLECTION_NAME, points, wait=True)
    assert remote_client.count(COLLECTION_NAME).count == len(points)
    assert isinstance(
        remote_client.retrieve(COLLECTION_NAME, ids=[1], with_vectors=True)[0].vector["text"], list
    )  # assert doc has been substituted with its embedding

    recreate_collection(remote_client, COLLECTION_NAME)

    vectors = [
        {"text": dense_doc_1, "image": dense_image_1, "sparse-text": sparse_doc_1},
        {"text": dense_doc_2, "image": dense_image_2, "sparse-text": sparse_doc_2},
        {"text": dense_doc_3, "image": dense_image_3, "sparse-text": sparse_doc_3},
    ]
    ids = list(range(len(vectors)))
    remote_client.upload_collection(COLLECTION_NAME, ids=ids, vectors=vectors, wait=True)

    assert remote_client.count(COLLECTION_NAME).count == len(vectors)
    assert isinstance(
        remote_client.retrieve(COLLECTION_NAME, ids=[1], with_vectors=True)[0].vector["text"], list
    )  # assert doc has been substituted with its embedding

    recreate_collection(remote_client, COLLECTION_NAME)

    remote_client.upload_points(COLLECTION_NAME, points, parallel=2, batch_size=2, wait=True)
    assert remote_client.count(COLLECTION_NAME).count == len(points)
    assert isinstance(
        remote_client.retrieve(COLLECTION_NAME, ids=[1], with_vectors=True)[0].vector["text"], list
    )  # assert doc has been substituted with its embedding

    recreate_collection(remote_client, COLLECTION_NAME)

    remote_client.upload_collection(
        COLLECTION_NAME, ids=ids, vectors=vectors, parallel=2, batch_size=2, wait=True
    )

    assert remote_client.count(COLLECTION_NAME).count == len(vectors)
    assert isinstance(
        remote_client.retrieve(COLLECTION_NAME, ids=[1], with_vectors=True)[0].vector["text"], list
    )  # assert doc has been substituted with its embedding

    assert isinstance(points[0].vector["text"], models.Document)

    recreate_collection(remote_client, COLLECTION_NAME)

    remote_client.upload_points(COLLECTION_NAME, iter(points), parallel=2, batch_size=2, wait=True)

    assert remote_client.count(COLLECTION_NAME).count == len(points)
    assert isinstance(
        remote_client.retrieve(COLLECTION_NAME, ids=[1], with_vectors=True)[0].vector["text"], list
    )  # assert doc has been substituted with its embedding

    assert isinstance(vectors[0]["text"], models.Document)

    recreate_collection(remote_client, COLLECTION_NAME)

    remote_client.upload_collection(
        COLLECTION_NAME, ids=ids, vectors=iter(vectors), parallel=2, batch_size=2, wait=True
    )

    assert remote_client.count(COLLECTION_NAME).count == len(vectors)
    assert isinstance(
        remote_client.retrieve(COLLECTION_NAME, ids=[1], with_vectors=True)[0].vector["text"], list
    )  # assert doc has been substituted with its embedding


def test_query_points(cached_embeddings):
    local_client = QdrantClient(":memory:")
    if not local_client._FASTEMBED_INSTALLED:
        pytest.skip("FastEmbed is not installed, skipping")
    local_kwargs = {}
    local_client._client.query_points = arg_interceptor(
        local_client._client.query_points, local_kwargs
    )
    sparse_doc_1 = models.Document(text="hello world", model=SPARSE_MODEL_NAME)
    sparse_doc_2 = models.Document(text="bye world", model=SPARSE_MODEL_NAME)
    sparse_doc_3 = models.Document(text="goodbye world", model=SPARSE_MODEL_NAME)
    sparse_doc_4 = models.Document(text="good afternoon world", model=SPARSE_MODEL_NAME)
    sparse_doc_5 = models.Document(text="good morning world", model=SPARSE_MODEL_NAME)
    points = [
        models.PointStruct(id=i, vector={"sparse-text": doc}, payload={"content": doc.text})
        for i, doc in enumerate(
            [sparse_doc_1, sparse_doc_2, sparse_doc_3, sparse_doc_4, sparse_doc_5],
        )
    ]

    populate_sparse_collection(local_client, points, vector_name="sparse-text")

    # region non-prefetch queries
    local_client.query_points(COLLECTION_NAME, sparse_doc_1, using="sparse-text")

    current_query = local_kwargs["query"]
    assert isinstance(current_query.nearest, models.SparseVector)
    retrieved_point_id_0 = local_client.retrieve(COLLECTION_NAME, ids=[0], with_vectors=True)[0]
    # assert that we generate different embeddings for doc and query

    nearest_query = models.NearestQuery(nearest=sparse_doc_1)
    local_client.query_points(COLLECTION_NAME, nearest_query, using="sparse-text")
    current_query = local_kwargs["query"]
    assert isinstance(current_query.nearest, models.SparseVector)

    recommend_query = models.RecommendQuery(
        recommend=models.RecommendInput(
            positive=[sparse_doc_1],
            negative=[sparse_doc_1],
        )
    )
    local_client.query_points(COLLECTION_NAME, recommend_query, using="sparse-text")
    current_query = local_kwargs["query"]
    assert all(
        isinstance(vector, models.SparseVector) for vector in current_query.recommend.positive
    )
    assert all(
        isinstance(vector, models.SparseVector) for vector in current_query.recommend.negative
    )

    discover_query = models.DiscoverQuery(
        discover=models.DiscoverInput(
            target=sparse_doc_1,
            context=models.ContextPair(
                positive=sparse_doc_2,
                negative=sparse_doc_3,
            ),
        )
    )
    local_client.query_points(COLLECTION_NAME, discover_query, using="sparse-text")

    current_query = local_kwargs["query"]
    assert isinstance(current_query.discover.target, models.SparseVector)
    context_pair = current_query.discover.context
    assert isinstance(context_pair.positive, models.SparseVector)
    assert isinstance(context_pair.negative, models.SparseVector)

    discover_query_list = models.DiscoverQuery(
        discover=models.DiscoverInput(
            target=sparse_doc_1,
            context=[
                models.ContextPair(
                    positive=sparse_doc_2,
                    negative=sparse_doc_3,
                )
            ],
        )
    )
    local_client.query_points(COLLECTION_NAME, discover_query_list, using="sparse-text")

    current_query = local_kwargs["query"]
    assert isinstance(current_query.discover.target, models.SparseVector)
    context_pairs = current_query.discover.context
    assert all(isinstance(pair.positive, models.SparseVector) for pair in context_pairs)
    assert all(isinstance(pair.negative, models.SparseVector) for pair in context_pairs)

    context_query = models.ContextQuery(
        context=models.ContextPair(
            positive=sparse_doc_1,
            negative=sparse_doc_2,
        )
    )
    local_client.query_points(COLLECTION_NAME, context_query, using="sparse-text")

    current_query = local_kwargs["query"]
    context = current_query.context
    assert isinstance(context.positive, models.SparseVector)
    assert isinstance(context.negative, models.SparseVector)

    context_query_list = models.ContextQuery(
        context=[
            models.ContextPair(
                positive=sparse_doc_1,
                negative=sparse_doc_2,
            ),
            models.ContextPair(
                positive=sparse_doc_3,
                negative=sparse_doc_4,
            ),
        ]
    )
    local_client.query_points(COLLECTION_NAME, context_query_list, using="sparse-text")

    current_query = local_kwargs["query"]
    contexts = current_query.context
    assert all(isinstance(context.positive, models.SparseVector) for context in contexts)
    assert all(isinstance(context.negative, models.SparseVector) for context in contexts)
    # endregion

    # region prefetch queries
    prefetch = models.Prefetch(
        query=nearest_query,
        prefetch=models.Prefetch(
            query=nearest_query,
            prefetch=models.Prefetch(
                query=nearest_query,
                prefetch=[
                    models.Prefetch(query=discover_query_list, limit=5, using="sparse-text"),
                    models.Prefetch(query=nearest_query, using="sparse-text", limit=5),
                ],
                using="sparse-text",
                limit=4,
            ),
            using="sparse-text",
            limit=3,
        ),
        using="sparse-text",
        limit=2,
    )
    local_client.query_points(
        COLLECTION_NAME, query=nearest_query, prefetch=prefetch, limit=1, using="sparse-text"
    )

    current_query = local_kwargs["query"]
    current_prefetch = local_kwargs["prefetch"]
    assert isinstance(current_query.nearest, models.SparseVector)
    assert isinstance(current_prefetch.query.nearest, models.SparseVector)
    assert isinstance(current_prefetch.prefetch.query.nearest, models.SparseVector)
    assert isinstance(current_prefetch.prefetch.prefetch.query.nearest, models.SparseVector)
    assert isinstance(
        current_prefetch.prefetch.prefetch.prefetch[0].query.discover.target, models.SparseVector
    )
    context_pairs = current_prefetch.prefetch.prefetch.prefetch[0].query.discover.context
    assert all(isinstance(pair.positive, models.SparseVector) for pair in context_pairs)
    assert all(isinstance(pair.negative, models.SparseVector) for pair in context_pairs)

    assert isinstance(
        current_prefetch.prefetch.prefetch.prefetch[1].query.nearest, models.SparseVector
    )

    # endregion

    local_client.delete_collection(COLLECTION_NAME)


def test_query_points_is_query(cached_embeddings):
    # dense_model_name = "jinaai/jina-embeddings-v3"
    # dense_dim = 1024

    local_client = QdrantClient(":memory:")
    if not local_client._FASTEMBED_INSTALLED:
        pytest.skip("FastEmbed is not installed, skipping")
    local_kwargs = {}
    local_client._client.query_points = arg_interceptor(
        local_client._client.query_points, local_kwargs
    )
    # dense_doc_1 = models.Document(text="hello world", model=dense_model_name)  # todo: uncomment once this model is supported
    sparse_doc_1 = models.Document(text="hello world", model=SPARSE_MODEL_NAME)
    colbert_doc_1 = models.Document(text="hello world", model=COLBERT_MODEL_NAME)

    vectors_config = {
        # "dense-text": models.VectorParams(size=dense_dim, distance=models.Distance.COSINE),
        "colbert-text": models.VectorParams(
            size=COLBERT_DIM,
            distance=models.Distance.COSINE,
            multivector_config=models.MultiVectorConfig(
                comparator=models.MultiVectorComparator.MAX_SIM
            ),
        ),
    }
    sparse_vectors_config = {
        "sparse-text": models.SparseVectorParams(modifier=models.Modifier.IDF)
    }

    local_client.create_collection(
        COLLECTION_NAME,
        vectors_config=vectors_config,
        sparse_vectors_config=sparse_vectors_config,
    )

    points = [
        models.PointStruct(
            id=0, vector={"colbert-text": colbert_doc_1, "sparse-text": sparse_doc_1}
        )
    ]
    local_client.upsert(COLLECTION_NAME, points)

    retrieved_point = local_client.retrieve(COLLECTION_NAME, ids=[0], with_vectors=True)[0]

    # local_client.query_points(COLLECTION_NAME, dense_doc_1, using="dense-text")

    # assert isinstance(local_kwargs["query"].nearest, list)
    # assert not np.allclose(retrieved_point.vector["dense-text"], local_kwargs["query"].nearest, atol=1e-3)

    local_client.query_points(COLLECTION_NAME, sparse_doc_1, using="sparse-text")

    assert isinstance(local_kwargs["query"].nearest, models.SparseVector)
    assert not np.allclose(
        retrieved_point.vector["sparse-text"].values,
        local_kwargs["query"].nearest.values,
        atol=1e-3,
    )

    local_client.query_points(COLLECTION_NAME, colbert_doc_1, using="colbert-text")

    assert isinstance(local_kwargs["query"].nearest, list)
    # colbert has a min number of 32 tokens for query
    assert len(retrieved_point.vector["colbert-text"]) != len(local_kwargs["query"].nearest)

    local_client.delete_collection(COLLECTION_NAME)


def test_query_points_groups(cached_embeddings):
    local_client = QdrantClient(":memory:")
    if not local_client._FASTEMBED_INSTALLED:
        pytest.skip("FastEmbed is not installed, skipping")

    local_kwargs = {}
    local_client._client.query_points_groups = arg_interceptor(
        local_client._client.query_points_groups, local_kwargs
    )
    sparse_doc_1 = models.Document(text="hello world", model=SPARSE_MODEL_NAME)
    sparse_doc_2 = models.Document(text="bye world", model=SPARSE_MODEL_NAME)
    sparse_doc_3 = models.Document(text="goodbye world", model=SPARSE_MODEL_NAME)
    sparse_doc_4 = models.Document(text="good afternoon world", model=SPARSE_MODEL_NAME)
    sparse_doc_5 = models.Document(text="good morning world", model=SPARSE_MODEL_NAME)
    points = [
        models.PointStruct(id=i, vector={"sparse-text": doc}, payload={"content": doc.text})
        for i, doc in enumerate(
            [sparse_doc_1, sparse_doc_2, sparse_doc_3, sparse_doc_4, sparse_doc_5],
        )
    ]

    populate_sparse_collection(local_client, points, vector_name="sparse-text")

    # region query_points_groups
    local_client.query_points_groups(
        COLLECTION_NAME, group_by="content", query=sparse_doc_1, using="sparse-text"
    )

    current_query = local_kwargs["query"]
    assert isinstance(current_query.nearest, models.SparseVector)
    retrieved_point_id_0 = local_client.retrieve(COLLECTION_NAME, ids=[0], with_vectors=True)[0]
    # assert that we generate different embeddings for doc and query
    # we are using sparse_doc_1 as a query
    assert not (
        np.allclose(
            retrieved_point_id_0.vector["sparse-text"].values,
            current_query.nearest.values,
            atol=1e-3,
        )
    )

    prefetch_1 = models.Prefetch(
        query=models.NearestQuery(nearest=sparse_doc_2), using="sparse-text", limit=3
    )
    prefetch_2 = models.Prefetch(
        query=models.NearestQuery(nearest=sparse_doc_3), using="sparse-text", limit=3
    )

    local_client.query_points_groups(
        COLLECTION_NAME,
        group_by="content",
        query=sparse_doc_1,
        prefetch=[prefetch_1, prefetch_2],
        using="sparse-text",
    )

    current_query = local_kwargs["query"]
    current_prefetch = local_kwargs["prefetch"]
    assert isinstance(current_query.nearest, models.SparseVector)
    assert isinstance(current_prefetch[0].query.nearest, models.SparseVector)
    assert isinstance(current_prefetch[1].query.nearest, models.SparseVector)
    assert not (
        np.allclose(
            retrieved_point_id_0.vector["sparse-text"].values,
            current_query.nearest.values,
            atol=1e-3,
        )
    )
    retrieved_point_id_1 = local_client.retrieve(COLLECTION_NAME, ids=[1], with_vectors=True)[0]
    assert not (
        np.allclose(
            retrieved_point_id_1.vector["sparse-text"].values,
            current_prefetch[0].query.nearest.values,
            atol=1e-3,
        )
    )

    assert isinstance(prefetch_1.query.nearest, models.Document)
    local_kwargs.clear()
    local_client.query_points_groups(
        COLLECTION_NAME,
        group_by="content",
        query=sparse_doc_1,
        prefetch=prefetch_1,
        using="sparse-text",
    )
    current_prefetch = local_kwargs["prefetch"]
    assert isinstance(current_prefetch.query.nearest, models.SparseVector)
    assert not (
        np.allclose(
            retrieved_point_id_1.vector["sparse-text"].values,
            current_prefetch.query.nearest.values,
            atol=1e-3,
        )
    )
    # endregion

    local_client.delete_collection(COLLECTION_NAME)


def test_query_batch_points(cached_embeddings):
    local_client = QdrantClient(":memory:")
    if not local_client._FASTEMBED_INSTALLED:
        pytest.skip("FastEmbed is not installed, skipping")
    local_kwargs = {}
    local_client._client.query_batch_points = arg_interceptor(
        local_client._client.query_batch_points, local_kwargs
    )

    sparse_doc_1 = models.Document(text="hello world", model=SPARSE_MODEL_NAME)
    sparse_doc_2 = models.Document(text="bye world", model=SPARSE_MODEL_NAME)
    sparse_doc_3 = models.Document(text="goodbye world", model=SPARSE_MODEL_NAME)
    sparse_doc_4 = models.Document(text="good afternoon world", model=SPARSE_MODEL_NAME)
    sparse_doc_5 = models.Document(text="good morning world", model=SPARSE_MODEL_NAME)

    points = [
        models.PointStruct(id=i, vector={"sparse-text": dense_doc})
        for i, dense_doc in enumerate(
            [sparse_doc_1, sparse_doc_2, sparse_doc_3, sparse_doc_4, sparse_doc_5]
        )
    ]

    populate_sparse_collection(local_client, points, vector_name="sparse-text")

    prefetch_1 = models.Prefetch(
        query=models.NearestQuery(nearest=sparse_doc_2), limit=3, using="sparse-text"
    )
    prefetch_2 = models.Prefetch(
        query=models.NearestQuery(nearest=sparse_doc_3), limit=3, using="sparse-text"
    )

    query_requests = [
        models.QueryRequest(query=models.NearestQuery(nearest=sparse_doc_1), using="sparse-text"),
        models.QueryRequest(
            query=models.NearestQuery(nearest=sparse_doc_2),
            prefetch=[prefetch_1, prefetch_2],
            using="sparse-text",
        ),
    ]

    local_client.query_batch_points(COLLECTION_NAME, query_requests)
    current_requests = local_kwargs["requests"]
    assert all(
        [isinstance(request.query.nearest, models.SparseVector) for request in current_requests]
    )
    assert all(
        [
            isinstance(prefetch.query.nearest, models.SparseVector)
            for prefetch in current_requests[1].prefetch
        ]
    )

    retrieved_point = local_client.retrieve(COLLECTION_NAME, ids=[0], with_vectors=True)[0]
    assert not np.allclose(
        retrieved_point.vector["sparse-text"].values,
        current_requests[0].query.nearest.values,
        atol=1e-3,
    )
    local_client.delete_collection(COLLECTION_NAME)


def test_batch_update_points(cached_embeddings):
    local_client = QdrantClient(":memory:")
    if not local_client._FASTEMBED_INSTALLED:
        pytest.skip("FastEmbed is not installed, skipping")
    local_kwargs = {}
    local_client._client.batch_update_points = arg_interceptor(
        local_client._client.batch_update_points, local_kwargs
    )

    dense_doc_1 = models.Document(text="hello world", model=DENSE_MODEL_NAME)
    dense_doc_2 = models.Document(text="bye world", model=DENSE_MODEL_NAME)

    # region unnamed
    points = [
        models.PointStruct(id=1, vector=dense_doc_1),
        models.PointStruct(id=2, vector=dense_doc_2),
    ]

    populate_dense_collection(local_client, points)

    batch = models.Batch(ids=[2, 3], vectors=[dense_doc_1, dense_doc_2])
    upsert_operation = models.UpsertOperation(upsert=models.PointsBatch(batch=batch))
    local_client.batch_update_points(COLLECTION_NAME, [upsert_operation])
    current_operation = local_kwargs["update_operations"][0]
    current_batch = current_operation.upsert.batch
    assert all([isinstance(vector, list) for vector in current_batch.vectors])

    new_points = [
        models.PointStruct(id=3, vector=dense_doc_1),
        models.PointStruct(id=4, vector=dense_doc_2),
    ]
    upsert_operation = models.UpsertOperation(upsert=models.PointsList(points=new_points))
    local_client.batch_update_points(COLLECTION_NAME, [upsert_operation])
    current_operation = local_kwargs["update_operations"][0]
    current_batch = current_operation.upsert.points
    assert all([isinstance(vector.vector, list) for vector in current_batch])

    update_vectors_operation = models.UpdateVectorsOperation(
        update_vectors=models.UpdateVectors(points=[models.PointVectors(id=1, vector=dense_doc_2)])
    )
    upsert_operation = models.UpsertOperation(
        upsert=models.PointsList(points=[models.PointStruct(id=5, vector=dense_doc_2)])
    )
    local_client.batch_update_points(COLLECTION_NAME, [update_vectors_operation, upsert_operation])
    current_update_operation = local_kwargs["update_operations"][0]
    current_upsert_operation = local_kwargs["update_operations"][1]

    assert all(
        [
            isinstance(vector.vector, list)
            for vector in current_update_operation.update_vectors.points
        ]
    )
    assert all(
        [isinstance(vector.vector, list) for vector in current_upsert_operation.upsert.points]
    )

    local_client.delete_collection(COLLECTION_NAME)
    # endregion

    # region named
    points = [
        models.PointStruct(id=1, vector={"text": dense_doc_1}),
        models.PointStruct(id=2, vector={"text": dense_doc_2}),
    ]

    populate_dense_collection(local_client, points, vector_name="text")

    batch = models.Batch(ids=[2, 3], vectors={"text": [dense_doc_1, dense_doc_2]})
    upsert_operation = models.UpsertOperation(upsert=models.PointsBatch(batch=batch))
    local_client.batch_update_points(COLLECTION_NAME, [upsert_operation])
    current_operation = local_kwargs["update_operations"][0]
    current_batch = current_operation.upsert.batch
    assert all([isinstance(vector, list) for vector in current_batch.vectors.values()])

    new_points = [
        models.PointStruct(id=3, vector={"text": dense_doc_1}),
        models.PointStruct(id=4, vector={"text": dense_doc_2}),
    ]
    upsert_operation = models.UpsertOperation(upsert=models.PointsList(points=new_points))
    local_client.batch_update_points(COLLECTION_NAME, [upsert_operation])
    current_operation = local_kwargs["update_operations"][0]
    current_batch = current_operation.upsert.points
    assert all([isinstance(vector.vector["text"], list) for vector in current_batch])

    update_vectors_operation = models.UpdateVectorsOperation(
        update_vectors=models.UpdateVectors(
            points=[models.PointVectors(id=1, vector={"text": dense_doc_2})]
        )
    )
    upsert_operation = models.UpsertOperation(
        upsert=models.PointsList(points=[models.PointStruct(id=5, vector={"text": dense_doc_2})])
    )
    local_client.batch_update_points(COLLECTION_NAME, [update_vectors_operation, upsert_operation])
    current_update_operation = local_kwargs["update_operations"][0]
    current_upsert_operation = local_kwargs["update_operations"][1]

    assert all(
        [
            isinstance(vector.vector["text"], list)
            for vector in current_update_operation.update_vectors.points
        ]
    )
    assert all(
        [
            isinstance(vector.vector["text"], list)
            for vector in current_upsert_operation.upsert.points
        ]
    )

    local_client.delete_collection(COLLECTION_NAME)
    # endregion


def test_update_vectors(cached_embeddings):
    local_client = QdrantClient(":memory:")
    if not local_client._FASTEMBED_INSTALLED:
        pytest.skip("FastEmbed is not installed, skipping")

    local_kwargs = {}
    local_client._client.update_vectors = arg_interceptor(
        local_client._client.update_vectors, local_kwargs
    )

    dense_doc_1 = models.Document(
        text="hello world",
        model=DENSE_MODEL_NAME,
    )
    dense_doc_2 = models.Document(text="bye world", model=DENSE_MODEL_NAME)
    dense_doc_3 = models.Document(text="goodbye world", model=DENSE_MODEL_NAME)
    # region unnamed
    points = [
        models.PointStruct(id=1, vector=dense_doc_1),
        models.PointStruct(id=2, vector=dense_doc_2),
    ]

    populate_dense_collection(local_client, points)

    point_vectors = [
        models.PointVectors(id=1, vector=dense_doc_2),
        models.PointVectors(id=2, vector=dense_doc_3),
    ]

    local_client.update_vectors(COLLECTION_NAME, point_vectors)
    current_vectors = local_kwargs["points"]
    assert all([isinstance(vector.vector, list) for vector in current_vectors])

    local_client.delete_collection(COLLECTION_NAME)
    # endregion

    # region named
    points = [
        models.PointStruct(id=1, vector={"text": dense_doc_1}),
        models.PointStruct(id=2, vector={"text": dense_doc_2}),
    ]

    populate_dense_collection(local_client, points, vector_name="text")

    point_vectors = [
        models.PointVectors(id=1, vector={"text": dense_doc_2}),
        models.PointVectors(id=2, vector={"text": dense_doc_3}),
    ]

    local_client.update_vectors(COLLECTION_NAME, point_vectors)
    current_vectors = local_kwargs["points"]
    assert all([isinstance(vector.vector["text"], list) for vector in current_vectors])

    local_client.delete_collection(COLLECTION_NAME)
    # endregion


def test_propagate_options(cached_embeddings):
    local_client = QdrantClient(":memory:")
    if not local_client._FASTEMBED_INSTALLED:
        pytest.skip("FastEmbed is not installed, skipping")
    dense_doc_1 = models.Document(
        text="hello world", model=DENSE_MODEL_NAME, options={"lazy_load": True}
    )
    sparse_doc_1 = models.Document(
        text="hello world", model=SPARSE_MODEL_NAME, options={"lazy_load": True}
    )
    multi_doc_1 = models.Document(
        text="hello world", model=COLBERT_MODEL_NAME, options={"lazy_load": True}
    )

    dense_image_1 = models.Image(
        image=TEST_IMAGE_PATH, model=DENSE_IMAGE_MODEL_NAME, options={"lazy_load": True}
    )

    points = [
        models.PointStruct(
            id=1,
            vector={
                "text": dense_doc_1,
                "multi-text": multi_doc_1,
                "sparse-text": sparse_doc_1,
                "image": dense_image_1,
            },
        )
    ]

    vectors_config = {
        "text": models.VectorParams(size=DENSE_DIM, distance=models.Distance.COSINE),
        "multi-text": models.VectorParams(
            size=COLBERT_DIM,
            distance=models.Distance.COSINE,
            multivector_config=models.MultiVectorConfig(
                comparator=models.MultiVectorComparator.MAX_SIM
            ),
        ),
        "image": models.VectorParams(size=DENSE_IMAGE_DIM, distance=models.Distance.COSINE),
    }
    sparse_vectors_config = {
        "sparse-text": models.SparseVectorParams(modifier=models.Modifier.IDF)
    }
    local_client.create_collection(
        COLLECTION_NAME,
        vectors_config=vectors_config,
        sparse_vectors_config=sparse_vectors_config,
    )

    local_client.upsert(COLLECTION_NAME, points)

    assert local_client._model_embedder.embedder.embedding_models[DENSE_MODEL_NAME][
        0
    ].model.model.lazy_load
    assert local_client._model_embedder.embedder.sparse_embedding_models[SPARSE_MODEL_NAME][
        0
    ].model.model.lazy_load
    assert local_client._model_embedder.embedder.late_interaction_embedding_models[
        COLBERT_MODEL_NAME
    ][0].model.model.lazy_load
    assert local_client._model_embedder.embedder.image_embedding_models[DENSE_IMAGE_MODEL_NAME][
        0
    ].model.model.lazy_load

    local_client._model_embedder.embedder.embedding_models.clear()
    local_client._model_embedder.embedder.sparse_embedding_models.clear()
    local_client._model_embedder.embedder.late_interaction_embedding_models.clear()
    local_client._model_embedder.embedder.image_embedding_models.clear()

    inference_object_dense_doc_1 = models.InferenceObject(
        object="hello world",
        model=DENSE_MODEL_NAME,
        options={"lazy_load": True},
    )

    inference_object_sparse_doc_1 = models.InferenceObject(
        object="hello world",
        model=SPARSE_MODEL_NAME,
        options={"lazy_load": True},
    )

    inference_object_multi_doc_1 = models.InferenceObject(
        object="hello world",
        model=COLBERT_MODEL_NAME,
        options={"lazy_load": True},
    )

    inference_object_dense_image_1 = models.InferenceObject(
        object=TEST_IMAGE_PATH,
        model=DENSE_IMAGE_MODEL_NAME,
        options={"lazy_load": True},
    )

    points = [
        models.PointStruct(
            id=2,
            vector={
                "text": inference_object_dense_doc_1,
                "multi-text": inference_object_multi_doc_1,
                "sparse-text": inference_object_sparse_doc_1,
                "image": inference_object_dense_image_1,
            },
        )
    ]

    local_client.upsert(COLLECTION_NAME, points)

    assert local_client._model_embedder.embedder.embedding_models[DENSE_MODEL_NAME][
        0
    ].model.model.lazy_load
    assert local_client._model_embedder.embedder.sparse_embedding_models[SPARSE_MODEL_NAME][
        0
    ].model.model.lazy_load
    assert local_client._model_embedder.embedder.late_interaction_embedding_models[
        COLBERT_MODEL_NAME
    ][0].model.model.lazy_load
    assert local_client._model_embedder.embedder.image_embedding_models[DENSE_IMAGE_MODEL_NAME][
        0
    ].model.model.lazy_load


def test_inference_object(cached_embeddings):
    local_client = QdrantClient(":memory:")
    if not local_client._FASTEMBED_INSTALLED:
        pytest.skip("FastEmbed is not installed, skipping")
    local_kwargs = {}
    local_client._client.upsert = arg_interceptor(local_client._client.upsert, local_kwargs)

    inference_object_dense_doc_1 = models.InferenceObject(
        object="hello world",
        model=DENSE_MODEL_NAME,
        options={"lazy_load": True},
    )

    inference_object_sparse_doc_1 = models.InferenceObject(
        object="hello world",
        model=SPARSE_MODEL_NAME,
        options={"lazy_load": True},
    )

    inference_object_multi_doc_1 = models.InferenceObject(
        object="hello world",
        model=COLBERT_MODEL_NAME,
        options={"lazy_load": True},
    )

    inference_object_dense_image_1 = models.InferenceObject(
        object=TEST_IMAGE_PATH,
        model=DENSE_IMAGE_MODEL_NAME,
        options={"lazy_load": True},
    )

    points = [
        models.PointStruct(
            id=1,
            vector={
                "text": inference_object_dense_doc_1,
                "multi-text": inference_object_multi_doc_1,
                "sparse-text": inference_object_sparse_doc_1,
                "image": inference_object_dense_image_1,
            },
        )
    ]
    vectors_config = {
        "text": models.VectorParams(size=DENSE_DIM, distance=models.Distance.COSINE),
        "multi-text": models.VectorParams(
            size=COLBERT_DIM,
            distance=models.Distance.COSINE,
            multivector_config=models.MultiVectorConfig(
                comparator=models.MultiVectorComparator.MAX_SIM
            ),
        ),
        "image": models.VectorParams(size=DENSE_IMAGE_DIM, distance=models.Distance.COSINE),
    }
    sparse_vectors_config = {
        "sparse-text": models.SparseVectorParams(modifier=models.Modifier.IDF)
    }

    local_client.create_collection(
        COLLECTION_NAME,
        vectors_config=vectors_config,
        sparse_vectors_config=sparse_vectors_config,
    )
    local_client.upsert(COLLECTION_NAME, points)
    vec_points = local_kwargs["points"]
    vector = vec_points[0].vector
    assert isinstance(vector["text"], list)
    assert isinstance(vector["multi-text"], list)
    assert isinstance(vector["sparse-text"], models.SparseVector)
    assert isinstance(vector["image"], list)
    assert local_client.scroll(COLLECTION_NAME, limit=1, with_vectors=True)[0]

    local_client.query_points(COLLECTION_NAME, inference_object_dense_doc_1, using="text")
    local_client.query_points(COLLECTION_NAME, inference_object_sparse_doc_1, using="sparse-text")
    local_client.query_points(COLLECTION_NAME, inference_object_multi_doc_1, using="multi-text")
    local_client.query_points(COLLECTION_NAME, inference_object_dense_image_1, using="image")

    local_client.delete_collection(COLLECTION_NAME)


@pytest.mark.parametrize("parallel", [1, 2])
def test_upload_mixed_batches_upload_points(parallel, cached_embeddings):
    local_client = QdrantClient(":memory:")
    if not local_client._FASTEMBED_INSTALLED:
        pytest.skip("FastEmbed is not installed, skipping")
    remote_client = QdrantClient()
    half_dense_dim = DENSE_DIM // 2
    batch_size = 2

    ref_vector = [0.0, 0.2] * half_dense_dim
    norm_ref_vector = (np.array(ref_vector) / np.linalg.norm(ref_vector)).tolist()

    # region separate plain batches
    points = [
        models.PointStruct(
            id=1, vector=models.Document(text="hello world", model=DENSE_MODEL_NAME)
        ),
        models.PointStruct(id=2, vector=models.Document(text="bye world", model=DENSE_MODEL_NAME)),
        models.PointStruct(id=3, vector=ref_vector),
        models.PointStruct(id=4, vector=[0.1, 0.2] * half_dense_dim),
    ]

    vectors_config = models.VectorParams(size=DENSE_DIM, distance=models.Distance.COSINE)

    if remote_client.collection_exists(COLLECTION_NAME):
        remote_client.delete_collection(COLLECTION_NAME)
    remote_client.create_collection(COLLECTION_NAME, vectors_config=vectors_config)

    remote_client.upload_points(
        COLLECTION_NAME, points, batch_size=batch_size, wait=True, parallel=parallel
    )

    assert remote_client.count(COLLECTION_NAME).count == len(points)
    assert np.allclose(
        remote_client.retrieve(COLLECTION_NAME, ids=[3], with_vectors=True)[0].vector,
        norm_ref_vector,
    )

    remote_client.delete_collection(COLLECTION_NAME)
    # endregion

    # region mixed plain batches
    points = [
        models.PointStruct(
            id=1, vector=models.Document(text="hello world", model=DENSE_MODEL_NAME)
        ),
        models.PointStruct(id=2, vector=ref_vector),
        models.PointStruct(id=3, vector=models.Document(text="bye world", model=DENSE_MODEL_NAME)),
        models.PointStruct(id=4, vector=[0.1, 0.2] * half_dense_dim),
    ]

    remote_client.create_collection(COLLECTION_NAME, vectors_config=vectors_config)

    remote_client.upload_points(
        COLLECTION_NAME, points, batch_size=batch_size, wait=True, parallel=parallel
    )

    assert remote_client.count(COLLECTION_NAME).count == len(points)
    assert np.allclose(
        remote_client.retrieve(COLLECTION_NAME, ids=[2], with_vectors=True)[0].vector,
        norm_ref_vector,
    )

    remote_client.delete_collection(COLLECTION_NAME)
    # endregion

    # region mixed named batches

    vectors_config = {
        "text": models.VectorParams(size=DENSE_DIM, distance=models.Distance.COSINE),
        "plain": models.VectorParams(size=DENSE_DIM, distance=models.Distance.COSINE),
    }
    points = [
        models.PointStruct(
            id=1,
            vector={
                "text": models.Document(text="hello world", model=DENSE_MODEL_NAME),
                "plain": [0.1, 0.2] * half_dense_dim,
            },
        ),
        models.PointStruct(
            id=2,
            vector={
                "plain": ref_vector,
                "text": models.Document(text="bye world", model=DENSE_MODEL_NAME),
            },
        ),
        models.PointStruct(
            id=3,
            vector={"plain": [0.3, 0.2] * half_dense_dim},
        ),
        models.PointStruct(
            id=4,
            vector={"text": models.Document(text="bye world", model=DENSE_MODEL_NAME)},
        ),
    ]

    remote_client.create_collection(COLLECTION_NAME, vectors_config=vectors_config)

    remote_client.upload_points(
        COLLECTION_NAME, points, batch_size=batch_size, wait=True, parallel=parallel
    )

    assert remote_client.count(COLLECTION_NAME).count == len(points)
    assert np.allclose(
        remote_client.retrieve(COLLECTION_NAME, ids=[2], with_vectors=True)[0].vector["plain"],
        norm_ref_vector,
    )
    remote_client.delete_collection(COLLECTION_NAME)
    # endregion


@pytest.mark.parametrize("parallel", [1, 2])
def test_upload_mixed_batches_upload_collection(parallel, cached_embeddings):
    local_client = QdrantClient(":memory:")
    if not local_client._FASTEMBED_INSTALLED:
        pytest.skip("FastEmbed is not installed, skipping")
    remote_client = QdrantClient()
    half_dense_dim = DENSE_DIM // 2
    batch_size = 2
    ref_vector = [0.0, 0.2] * half_dense_dim
    norm_ref_vector = (np.array(ref_vector) / np.linalg.norm(ref_vector)).tolist()

    # region separate plain batches
    ids = [0, 1, 2, 3]
    vectors = [
        models.Document(text="hello world", model=DENSE_MODEL_NAME),
        models.Document(text="bye world", model=DENSE_MODEL_NAME),
        ref_vector,
        [0.1, 0.2] * half_dense_dim,
    ]

    vectors_config = models.VectorParams(size=DENSE_DIM, distance=models.Distance.COSINE)
    if remote_client.collection_exists(COLLECTION_NAME):
        remote_client.delete_collection(COLLECTION_NAME)
    remote_client.create_collection(COLLECTION_NAME, vectors_config=vectors_config)
    remote_client.upload_collection(
        COLLECTION_NAME,
        ids=ids,
        vectors=vectors,
        batch_size=batch_size,
        wait=True,
        parallel=parallel,
    )

    assert remote_client.count(COLLECTION_NAME).count == len(vectors)
    assert np.allclose(
        remote_client.retrieve(COLLECTION_NAME, ids=[2], with_vectors=True)[0].vector,
        norm_ref_vector,
    )

    remote_client.delete_collection(COLLECTION_NAME)
    # endregion

    # region mixed plain batches
    vectors = [
        models.Document(text="hello world", model=DENSE_MODEL_NAME),
        ref_vector,
        models.Document(text="bye world", model=DENSE_MODEL_NAME),
        [0.1, 0.2] * half_dense_dim,
    ]

    remote_client.create_collection(COLLECTION_NAME, vectors_config=vectors_config)
    remote_client.upload_collection(
        COLLECTION_NAME,
        ids=ids,
        vectors=vectors,
        batch_size=batch_size,
        wait=True,
        parallel=parallel,
    )

    assert remote_client.count(COLLECTION_NAME).count == len(vectors)
    assert np.allclose(
        remote_client.retrieve(COLLECTION_NAME, ids=[1], with_vectors=True)[0].vector,
        norm_ref_vector,
    )
    remote_client.delete_collection(COLLECTION_NAME)
    # endregion

    # region mixed named batches

    vectors_config = {
        "text": models.VectorParams(size=DENSE_DIM, distance=models.Distance.COSINE),
        "plain": models.VectorParams(size=DENSE_DIM, distance=models.Distance.COSINE),
    }
    vectors = [
        {
            "text": models.Document(text="hello world", model=DENSE_MODEL_NAME),
            "plain": [0.0, 0.2] * half_dense_dim,
        },
        {
            "plain": ref_vector,
            "text": models.Document(text="bye world", model=DENSE_MODEL_NAME),
        },
        {"plain": [0.3, 0.2] * half_dense_dim},
        {"text": models.Document(text="bye world", model=DENSE_MODEL_NAME)},
    ]

    remote_client.create_collection(COLLECTION_NAME, vectors_config=vectors_config)
    remote_client.upload_collection(
        COLLECTION_NAME,
        ids=ids,
        vectors=vectors,
        batch_size=batch_size,
        wait=True,
        parallel=parallel,
    )

    assert remote_client.count(COLLECTION_NAME).count == len(vectors)
    assert np.allclose(
        remote_client.retrieve(COLLECTION_NAME, ids=[1], with_vectors=True)[0].vector["plain"],
        norm_ref_vector,
    )

    remote_client.delete_collection(COLLECTION_NAME)
    # endregion


def test_upsert_batch_with_different_options():
    bm25_name = "Qdrant/bm25"
    local_client = QdrantClient(":memory:")
    if not local_client._FASTEMBED_INSTALLED:
        pytest.skip("FastEmbed is not installed, skipping")

    sparse_doc_1 = models.Document(
        text="running run", model=bm25_name, options={"language": "english"}
    )
    sparse_doc_2 = models.Document(
        text="running run", model=bm25_name, options={"language": "german"}
    )
    sparse_doc_3 = models.Document(
        text="running run", model=bm25_name, options={"language": "english"}
    )
    sparse_doc_4 = models.Document(
        text="running run", model=bm25_name, options={"language": "german"}
    )

    sparse_vectors_config = {
        "sparse-text-en": models.SparseVectorParams(modifier=models.Modifier.IDF),
        "sparse-text-de": models.SparseVectorParams(modifier=models.Modifier.IDF),
    }

    local_client.create_collection(
        COLLECTION_NAME, vectors_config={}, sparse_vectors_config=sparse_vectors_config
    )
    points = [
        models.PointStruct(
            id=0, vector={"sparse-text-en": sparse_doc_1, "sparse-text-de": sparse_doc_2}
        ),
        models.PointStruct(id=1, vector={"sparse-text-en": sparse_doc_3}),
        models.PointStruct(id=2, vector={"sparse-text-de": sparse_doc_4}),
    ]

    local_client.upsert(COLLECTION_NAME, points)

    read_points, _ = local_client.scroll(COLLECTION_NAME, limit=4, with_vectors=True)
    assert len(read_points) == 3
    assert (
        read_points[0].vector["sparse-text-en"].indices
        != read_points[0].vector["sparse-text-de"].indices
    )
    assert (
        read_points[0].vector["sparse-text-en"].indices
        == read_points[1].vector["sparse-text-en"].indices
    )
    assert (
        read_points[0].vector["sparse-text-de"].indices
        == read_points[2].vector["sparse-text-de"].indices
    )


def test_batch_size_propagation():
    def mock(func, kw_param_storage):
        def decorated(*args, **kwargs):
            for k in kwargs:
                kw_param_storage[k] = kwargs[k]
            return func(*args, **kwargs)

        return decorated

    param_storage = {}

    bm25_name = "Qdrant/bm25"
    inference_batch_size = 2
    local_client = QdrantClient(":memory:", local_inference_batch_size=inference_batch_size)
    if not local_client._FASTEMBED_INSTALLED:
        pytest.skip("FastEmbed is not installed, skipping")
    local_client._model_embedder.embedder.embed = mock(
        local_client._model_embedder.embedder.embed, param_storage
    )

    if not local_client._FASTEMBED_INSTALLED:
        pytest.skip("FastEmbed is not installed, skipping")

    sparse_doc_1 = models.Document(
        text="a quick",
        model=bm25_name,
    )
    sparse_doc_2 = models.Document(
        text="brown fox",
        model=bm25_name,
    )
    sparse_doc_3 = models.Document(
        text="jumps over",
        model=bm25_name,
    )
    sparse_doc_4 = models.Document(
        text="a lazy dog",
        model=bm25_name,
    )
    local_client.create_collection(
        COLLECTION_NAME, sparse_vectors_config={"sparse": models.SparseVectorParams()}
    )
    points = [
        models.PointStruct(id=0, vector={"sparse": sparse_doc_1}),
        models.PointStruct(id=1, vector={"sparse": sparse_doc_2}),
        models.PointStruct(id=2, vector={"sparse": sparse_doc_3}),
        models.PointStruct(id=3, vector={"sparse": sparse_doc_4}),
    ]
    local_client.upsert(  # uses _embed_models
        COLLECTION_NAME, points
    )
    assert param_storage["batch_size"] == inference_batch_size
    param_storage.clear()

    local_client.upload_points(
        COLLECTION_NAME, points=points, batch_size=3
    )  # uses _embed_models_strict
    assert param_storage["batch_size"] == inference_batch_size
    param_storage.clear()


@pytest.fixture
def mock_late_interaction_multimodal_embedding():
    try:
        import fastembed
    except ImportError:
        pytest.skip("FastEmbed is not installed, skipping")

    class LateInteractionMultimodalEmbeddingMock:
        text_calls = 0
        image_calls = 0
        num_image_columns = 10
        num_text_columns = 20
        text_embedding = np.array([[0.1, 0.2, 0.3]] * num_text_columns)
        image_embedding = np.array([[0.1, 0.2, 0.3]] * num_image_columns)

        def __init__(self, *args, **kwargs):
            pass

        def embed_text(self, documents, *args, **kwargs):
            LateInteractionMultimodalEmbeddingMock.text_calls += 1
            for _ in documents:
                yield self.text_embedding

        def embed_image(self, images, *args, **kwargs):
            LateInteractionMultimodalEmbeddingMock.image_calls += 1
            for _ in images:
                yield self.image_embedding

    original_class = qdrant_client.embed.embedder.LateInteractionMultimodalEmbedding
    qdrant_client.embed.embedder.LateInteractionMultimodalEmbedding = (
        LateInteractionMultimodalEmbeddingMock
    )
    yield LateInteractionMultimodalEmbeddingMock
    qdrant_client.embed.embedder.LateInteractionMultimodalEmbedding = original_class


def test_embed_multimodal(mock_late_interaction_multimodal_embedding):
    mock_cls = mock_late_interaction_multimodal_embedding

    local_client = QdrantClient(":memory:")
    if not local_client._FASTEMBED_INSTALLED:
        pytest.skip("FastEmbed is not installed, skipping")

    point_1 = models.PointStruct(
        id=1,
        vector={
            "text": models.Document(text="a quick brown fox", model="Qdrant/colpali-v1.3-fp16"),
            "image": models.Image(image=TEST_IMAGE_PATH, model="Qdrant/colpali-v1.3-fp16"),
        },
    )
    point_2 = models.PointStruct(
        id=2,
        vector={
            "text": models.Document(
                text="jumped over a lazy dog", model="Qdrant/colpali-v1.3-fp16"
            ),
        },
    )

    dim = 3
    local_client.create_collection(
        COLLECTION_NAME,
        vectors_config={
            "text": models.VectorParams(
                size=dim,
                distance=models.Distance.MANHATTAN,
                multivector_config=models.MultiVectorConfig(
                    comparator=models.MultiVectorComparator.MAX_SIM
                ),
            ),
            "image": models.VectorParams(
                size=dim,
                distance=models.Distance.MANHATTAN,
                multivector_config=models.MultiVectorConfig(
                    comparator=models.MultiVectorComparator.MAX_SIM
                ),
            ),
        },
    )

    local_client.upsert(COLLECTION_NAME, [point_1, point_2])

    assert mock_cls.text_calls == 1  # assert that text records were assembled into batches
    assert mock_cls.image_calls == 1

    records, _ = local_client.scroll(COLLECTION_NAME, limit=2, with_vectors=True)
    assert len(records) == 2
    assert len(records[1].vector) == 1

    np_text_vectors = np.array([records[0].vector["text"], records[1].vector["text"]])
    np_image_vector = np.array(records[0].vector["image"])

    assert all(vector.shape == (mock_cls.num_text_columns, dim) for vector in np_text_vectors)
    assert np_image_vector.shape == (mock_cls.num_image_columns, dim)

    # check that embeddings order was right
    assert np.allclose(np_text_vectors[0], mock_cls.text_embedding)
    assert np.allclose(np_text_vectors[1], mock_cls.text_embedding)
    assert np.allclose(np_image_vector, mock_cls.image_embedding)
