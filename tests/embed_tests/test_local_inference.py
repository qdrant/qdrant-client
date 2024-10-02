import random

import numpy as np
import pytest
from pydantic import BaseModel

from qdrant_client import QdrantClient, models
from tests.congruence_tests.test_common import compare_collections, compare_client_results
from tests.utils import read_version


@pytest.mark.parametrize("prefer_grpc", [False, True])
def test_query_embeddings_conversion(prefer_grpc):
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    remote_client = QdrantClient(prefer_grpc=prefer_grpc)
    if not remote_client._FASTEMBED_INSTALLED:
        pytest.skip("FastEmbed is not installed, skipping")

    plain_document_query = models.Document(text="plain text", model=model_name)
    nearest_query = models.NearestQuery(
        nearest=models.Document(text="plain text", model=model_name)
    )
    recommend_query = models.RecommendQuery(
        recommend=models.RecommendInput(
            positive=[models.Document(text="positive recommend", model=model_name)],
            negative=[models.Document(text="negative recommend", model=model_name)],
        )
    )
    discover_query = models.DiscoverQuery(
        discover=models.DiscoverInput(
            target=models.Document(text="No model discovery target", model=model_name),
            context=[
                models.ContextPair(
                    positive=models.Document(text="No model discovery positive", model=model_name),
                    negative=models.Document(text="No model discovery negative", model=model_name),
                )
            ],
        )
    )
    context_query = models.ContextQuery(
        context=models.ContextPair(
            positive=models.Document(text="No model context positive", model=model_name),
            negative=models.Document(text="No model context negative", model=model_name),
        )
    )

    modified_query = remote_client._embed_models(plain_document_query)
    assert isinstance(modified_query, list)

    modified_query = remote_client._embed_models(nearest_query)
    assert isinstance(modified_query.nearest, list)

    modified_query = remote_client._embed_models(recommend_query)
    positives = modified_query.recommend.positive
    negatives = modified_query.recommend.negative
    positive = positives[0]
    negative = negatives[0]

    assert isinstance(positive, list) and isinstance(positive[0], float)
    assert isinstance(negative, list) and isinstance(negative[0], float)
    modified_query = remote_client._embed_models(discover_query)
    target = modified_query.discover.target
    context_pair = modified_query.discover.context[0]

    assert isinstance(target, list) and isinstance(target[0], float)

    positive = context_pair.positive
    negative = context_pair.negative

    assert isinstance(positive, list) and isinstance(positive[0], float)
    assert isinstance(negative, list) and isinstance(negative[0], float)

    modified_query = remote_client._embed_models(context_query)

    context = modified_query.context
    assert isinstance(context.positive, list) and isinstance(context.positive[0], float)
    assert isinstance(context.negative, list) and isinstance(context.negative[0], float)

    order_by_query = models.OrderByQuery(order_by="payload_field")
    modified_query = remote_client._embed_models(order_by_query)
    assert order_by_query == modified_query

    fusion_query = models.FusionQuery(fusion=models.Fusion.RRF)
    modified_query = remote_client._embed_models(fusion_query)
    assert fusion_query == modified_query


@pytest.mark.parametrize("prefer_grpc", [False, True])
def test_query_embeddings_prefetch(prefer_grpc):
    def query_is_float_list(pref: BaseModel) -> bool:
        nearest = pref.query
        if isinstance(nearest, models.NearestQuery):
            nearest = nearest.nearest
        return isinstance(nearest, list) and isinstance(nearest[0], float)

    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    remote_client = QdrantClient(prefer_grpc=prefer_grpc)
    if not remote_client._FASTEMBED_INSTALLED:
        pytest.skip("FastEmbed is not installed, skipping")

    empty_list_prefetch = models.Prefetch(query=[0.2, 0.1], prefetch=[])
    none_prefetch = models.Prefetch(query=[0.2, 0.1], prefetch=None)
    assert remote_client._embed_models(empty_list_prefetch).prefetch == []
    assert remote_client._embed_models(none_prefetch).prefetch is None

    nearest_query = models.NearestQuery(
        nearest=models.Document(text="nearest on prefetch", model=model_name)
    )
    prefetch = models.Prefetch(query=nearest_query)
    converted_prefetch = remote_client._embed_models(prefetch)
    assert query_is_float_list(converted_prefetch)

    nested_prefetch = models.Prefetch(
        query=nearest_query,
        prefetch=models.Prefetch(
            query=[0.2, 0.3],
            prefetch=[
                models.Prefetch(
                    query=models.Document(text="nested on prefetch", model=model_name),
                    prefetch=models.Prefetch(
                        query=models.Document(text="deep text", model=model_name)
                    ),
                ),
                models.Prefetch(
                    prefetch=[
                        models.Prefetch(
                            query=models.Document(text="another deep text", model=model_name)
                        ),
                        models.Prefetch(
                            query=models.Document(text="yet another deep text", model=model_name)
                        ),
                        models.Prefetch(query=[0.2, 0.4]),
                    ]
                ),
            ],
        ),
    )
    converted_nested_prefetch = remote_client._embed_models(nested_prefetch)
    assert query_is_float_list(converted_nested_prefetch)  # nearest_query check

    child_prefetch = converted_nested_prefetch.prefetch
    grandchild_prefetches = child_prefetch.prefetch
    assert query_is_float_list(grandchild_prefetches[0])  # "nested on prefetch" check
    assert query_is_float_list(grandchild_prefetches[0].prefetch)  # "deep text" check
    assert query_is_float_list(grandchild_prefetches[1].prefetch[0])  # "another deep text" check
    assert query_is_float_list(grandchild_prefetches[1].prefetch[1])  # yet another deep text check


def test_upsert_and_update():
    local_client = QdrantClient(":memory:")
    if not local_client._FASTEMBED_INSTALLED:
        pytest.skip("FastEmbed is not installed, skipping")
    remote_client = QdrantClient()
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    collection_name = "test-doc-embed"
    local_client.create_collection(
        collection_name,
        vectors_config=models.VectorParams(size=384, distance=models.Distance.COSINE),
    )
    if remote_client.collection_exists(collection_name):
        remote_client.delete_collection(collection_name)
    remote_client.create_collection(
        collection_name,
        vectors_config=models.VectorParams(size=384, distance=models.Distance.COSINE),
    )

    points = [
        models.PointStruct(
            id=1, vector=models.Document(text="It's a short query", model=model_name)
        ),
        models.PointStruct(
            id=2, vector=models.Document(text="It's another short query", model=model_name)
        ),
        models.PointStruct(
            id=3, vector=models.Document(text="It's an old good query", model=model_name)
        ),
    ]
    local_client.upsert(
        collection_name,
        points=points,
    )
    remote_client.upsert(collection_name, points=points)

    compare_collections(
        local_client, remote_client, num_vectors=3, collection_name=collection_name
    )

    prev_vector = local_client.retrieve(collection_name, ids=[1], with_vectors=True)
    update_vector_points = [
        models.PointVectors(
            id=1, vector=models.Document(text="It's a substitution", model=model_name)
        )
    ]
    local_client.update_vectors(collection_name, update_vector_points)
    remote_client.update_vectors(collection_name, update_vector_points)

    updated_point = local_client.retrieve(collection_name, ids=[1], with_vectors=True)

    assert not np.allclose(updated_point[0].vector, prev_vector[0].vector, atol=10e-4)

    compare_collections(
        local_client, remote_client, num_vectors=3, collection_name=collection_name
    )

    operations = [
        models.UpsertOperation(
            upsert=models.PointsList(
                points=[
                    models.PointStruct(
                        id=1,
                        vector=models.Document(text="A completely new document", model=model_name),
                    )
                ]
            )
        ),
        models.UpsertOperation(
            upsert=models.PointsBatch(
                batch=models.Batch(
                    ids=[3], vectors=[models.Document(text="A decent document", model=model_name)]
                )
            )
        ),
        models.UpdateVectorsOperation(
            update_vectors=models.UpdateVectors(
                points=[
                    models.PointVectors(
                        id=2, vector=models.Document(text="Yet another document", model=model_name)
                    )
                ]
            )
        ),
    ]
    old_points = local_client.retrieve(collection_name, ids=[1, 2, 3], with_vectors=True)

    local_client.batch_update_points(collection_name, operations)
    remote_client.batch_update_points(collection_name, operations)
    new_points = local_client.retrieve(collection_name, ids=[1, 2, 3], with_vectors=True)

    assert not np.allclose(new_points[0].vector, old_points[0].vector, atol=10e-4)
    assert not np.allclose(new_points[1].vector, old_points[1].vector, atol=10e-4)
    assert not np.allclose(new_points[2].vector, old_points[2].vector, atol=10e-4)
    compare_collections(
        local_client, remote_client, num_vectors=3, collection_name=collection_name
    )


def test_query_batch_points():
    major, minor, patch, dev = read_version()
    if major is not None and (major, minor, patch) < (1, 10, 0):
        pytest.skip("Works as of qdrant 1.11.0")

    local_client = QdrantClient(":memory:")
    if not local_client._FASTEMBED_INSTALLED:
        pytest.skip("FastEmbed is not installed, skipping")
    remote_client = QdrantClient()
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    dim = 384
    collection_name = "test-doc-embed"
    local_client.create_collection(
        collection_name,
        vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE),
    )
    if remote_client.collection_exists(collection_name):
        remote_client.delete_collection(collection_name)
    remote_client.create_collection(
        collection_name,
        vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE),
    )

    texts = [
        "It's a short document",
        "Another short document",
        "Document to check query requests",
        "A nonsense document",
    ]
    points = [
        models.PointStruct(id=i, vector=models.Document(text=text, model=model_name))
        for i, text in enumerate(texts)
    ]
    # upload data
    local_client.upsert(collection_name, points=points)
    remote_client.upsert(collection_name, points=points)
    query_requests = [
        models.QueryRequest(
            query=models.NearestQuery(
                nearest=models.Document(text="It's a short query", model=model_name)
            )
        ),
        models.QueryRequest(
            query=models.NearestQuery(nearest=[random.random() for _ in range(dim)])
        ),
    ]

    compare_client_results(
        remote_client,
        local_client,
        lambda c: c.query_batch_points(collection_name, requests=query_requests),
    )
