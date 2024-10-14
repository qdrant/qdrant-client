# import random
from typing import Optional

import numpy as np

# import numpy as np
import pytest

# from pydantic import BaseModel

from qdrant_client import QdrantClient, models
from qdrant_client.client_base import QdrantBase
from tests.congruence_tests.test_common import (
    compare_collections,
    compare_client_results,
    # compare_client_results
)
from qdrant_client.qdrant_fastembed import IDF_EMBEDDING_MODELS

# from tests.utils import read_version


COLLECTION_NAME = "inference_collection"
DENSE_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DENSE_DIM = 384
SPARSE_MODEL_NAME = "Qdrant/bm42-all-minilm-l6-v2-attentions"
COLBERT_MODEL_NAME = "colbert-ir/colbertv2.0"
COLBERT_DIM = 128


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


@pytest.mark.parametrize("prefer_grpc", [True, False])
def test_upsert(prefer_grpc):
    local_client = QdrantClient(":memory:")
    if not local_client._FASTEMBED_INSTALLED:
        pytest.skip("FastEmbed is not installed, skipping")
    remote_client = QdrantClient(prefer_grpc=prefer_grpc)
    local_kwargs = {}
    local_client._client.upsert = arg_interceptor(local_client._client.upsert, local_kwargs)

    dense_doc_1 = models.Document(text="hello world", model=DENSE_MODEL_NAME)
    dense_doc_2 = models.Document(text="bye world", model=DENSE_MODEL_NAME)
    sparse_doc_1 = models.Document(text="hello world", model=SPARSE_MODEL_NAME)
    sparse_doc_2 = models.Document(text="bye world", model=SPARSE_MODEL_NAME)
    multi_doc_1 = models.Document(text="hello world", model=COLBERT_MODEL_NAME)
    multi_doc_2 = models.Document(text="bye world", model=COLBERT_MODEL_NAME)

    # region dense unnamed
    points = [
        models.PointStruct(id=1, vector=dense_doc_1),
        models.PointStruct(id=2, vector=dense_doc_2),
    ]
    populate_dense_collection(local_client, points)
    populate_dense_collection(remote_client, points)

    vec_points = local_kwargs["points"]
    assert all([isinstance(vec_point.vector, list) for vec_point in vec_points])

    compare_collections(
        local_client,
        remote_client,
        num_vectors=10,
        collection_name=COLLECTION_NAME,
    )

    batch = models.Batch(ids=[1, 2], vectors=[dense_doc_1, dense_doc_2])
    local_client.upsert(COLLECTION_NAME, batch)
    remote_client.upsert(COLLECTION_NAME, batch)
    batch = local_kwargs["points"]
    assert all([isinstance(vector, list) for vector in batch.vectors])

    compare_collections(
        local_client,
        remote_client,
        num_vectors=10,
        collection_name=COLLECTION_NAME,
    )

    local_client.delete_collection(COLLECTION_NAME)
    remote_client.delete_collection(COLLECTION_NAME)
    # endregion

    # region named vectors
    local_client.create_collection(
        COLLECTION_NAME,
        vectors_config={
            "text": models.VectorParams(size=DENSE_DIM, distance=models.Distance.COSINE),
            # todo: uncomment once late interaction models are supported in qdrant-client fastembed mixin
            # "multi-text": models.VectorParams(
            #     size=COLBERT_DIM,
            #     distance=models.Distance.COSINE,
            #     multivector_config=models.MultiVectorConfig(
            #         comparator=models.MultiVectorComparator.MAX_SIM
            #     ),
            # ),
        },
        sparse_vectors_config={
            "sparse-text": models.SparseVectorParams(modifier=models.Modifier.IDF)
        },
    )
    remote_client.create_collection(
        COLLECTION_NAME,
        vectors_config={
            "text": models.VectorParams(size=DENSE_DIM, distance=models.Distance.COSINE),
            # todo: uncomment once late interaction models are supported in qdrant-client fastembed mixin
            # "multi-text": models.VectorParams(
            #     size=COLBERT_DIM,
            #     distance=models.Distance.COSINE,
            #     multivector_config=models.MultiVectorConfig(
            #         comparator=models.MultiVectorComparator.MAX_SIM
            #     ),
            # ),
        },
        sparse_vectors_config={
            "sparse-text": models.SparseVectorParams(modifier=models.Modifier.IDF)
        },
    )
    points = [
        models.PointStruct(
            id=1,
            vector={
                "text": dense_doc_1,
                # todo: uncomment once late interaction models are supported in qdrant-client fastembed mixin
                # "multi-text": multi_doc_1,
                "sparse-text": sparse_doc_1,
            },
        ),
        models.PointStruct(
            id=2,
            vector={
                "text": dense_doc_2,
                # todo: uncomment once late interaction models are supported in qdrant-client fastembed mixin
                # "multi-text": multi_doc_2,
                "sparse-text": sparse_doc_2,
            },
        ),
    ]
    local_client.upsert(COLLECTION_NAME, points)
    remote_client.upsert(COLLECTION_NAME, points)

    vec_points = local_kwargs["points"]
    for vec_point in vec_points:
        assert isinstance(vec_point.vector, dict)
        assert isinstance(vec_point.vector["text"], list)
        # todo: uncomment once late interaction models are supported in qdrant-client fastembed mixin
        # assert isinstance(vec_point.vector["multi-text"], list)
        assert isinstance(vec_point.vector["sparse-text"], models.SparseVector)

    compare_collections(
        local_client, remote_client, num_vectors=10, collection_name=COLLECTION_NAME
    )

    batch = models.Batch(
        ids=[1, 2],
        vectors={
            "text": [dense_doc_1, dense_doc_2],
            # todo: uncomment once late interaction models are supported in qdrant-client fastembed mixin
            # "multi-text": [multi_doc_1, multi_doc_2],
            "sparse-text": [sparse_doc_1, sparse_doc_2],
        },
    )
    local_client.upsert(COLLECTION_NAME, batch)
    remote_client.upsert(COLLECTION_NAME, batch)

    batch = local_kwargs["points"]
    vectors = batch.vectors
    assert isinstance(vectors, dict)
    assert all([isinstance(vector, list) for vector in vectors["text"]])
    # todo: uncomment once late interaction models are supported in qdrant-client fastembed mixin
    # assert all([isinstance(vector, list) for vector in vec_point["multi-text"]])
    assert all([isinstance(vector, models.SparseVector) for vector in vectors["sparse-text"]])

    local_client.delete_collection(COLLECTION_NAME)
    remote_client.delete_collection(COLLECTION_NAME)
    # endregion


@pytest.mark.parametrize("prefer_grpc", [True, False])
def test_batch_update_points(prefer_grpc):
    local_client = QdrantClient(":memory:")
    if not local_client._FASTEMBED_INSTALLED:
        pytest.skip("FastEmbed is not installed, skipping")
    remote_client = QdrantClient(prefer_grpc=prefer_grpc)
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
    populate_dense_collection(remote_client, points)

    batch = models.Batch(ids=[2, 3], vectors=[dense_doc_1, dense_doc_2])
    upsert_operation = models.UpsertOperation(upsert=models.PointsBatch(batch=batch))
    local_client.batch_update_points(COLLECTION_NAME, [upsert_operation])
    remote_client.batch_update_points(COLLECTION_NAME, [upsert_operation])
    current_operation = local_kwargs["update_operations"][0]
    current_batch = current_operation.upsert.batch
    assert all([isinstance(vector, list) for vector in current_batch.vectors])

    compare_collections(
        local_client,
        remote_client,
        num_vectors=10,
        collection_name=COLLECTION_NAME,
    )

    new_points = [
        models.PointStruct(id=3, vector=dense_doc_1),
        models.PointStruct(id=4, vector=dense_doc_2),
    ]
    upsert_operation = models.UpsertOperation(upsert=models.PointsList(points=new_points))
    local_client.batch_update_points(COLLECTION_NAME, [upsert_operation])
    remote_client.batch_update_points(COLLECTION_NAME, [upsert_operation])
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
    remote_client.batch_update_points(
        COLLECTION_NAME, [update_vectors_operation, upsert_operation]
    )
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
    remote_client.delete_collection(COLLECTION_NAME)
    # endregion

    # region named
    points = [
        models.PointStruct(id=1, vector={"text": dense_doc_1}),
        models.PointStruct(id=2, vector={"text": dense_doc_2}),
    ]

    populate_dense_collection(local_client, points, vector_name="text")
    populate_dense_collection(remote_client, points, vector_name="text")

    batch = models.Batch(ids=[2, 3], vectors={"text": [dense_doc_1, dense_doc_2]})
    upsert_operation = models.UpsertOperation(upsert=models.PointsBatch(batch=batch))
    local_client.batch_update_points(COLLECTION_NAME, [upsert_operation])
    remote_client.batch_update_points(COLLECTION_NAME, [upsert_operation])
    current_operation = local_kwargs["update_operations"][0]
    current_batch = current_operation.upsert.batch
    assert all([isinstance(vector, list) for vector in current_batch.vectors.values()])

    compare_collections(
        local_client,
        remote_client,
        num_vectors=10,
        collection_name=COLLECTION_NAME,
    )

    new_points = [
        models.PointStruct(id=3, vector={"text": dense_doc_1}),
        models.PointStruct(id=4, vector={"text": dense_doc_2}),
    ]
    upsert_operation = models.UpsertOperation(upsert=models.PointsList(points=new_points))
    local_client.batch_update_points(COLLECTION_NAME, [upsert_operation])
    remote_client.batch_update_points(COLLECTION_NAME, [upsert_operation])
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
    remote_client.batch_update_points(
        COLLECTION_NAME, [update_vectors_operation, upsert_operation]
    )
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
    remote_client.delete_collection(COLLECTION_NAME)
    # endregion


@pytest.mark.parametrize("prefer_grpc", [True, False])
def test_update_vectors(prefer_grpc):
    local_client = QdrantClient(":memory:")
    if not local_client._FASTEMBED_INSTALLED:
        pytest.skip("FastEmbed is not installed, skipping")
    remote_client = QdrantClient(prefer_grpc=prefer_grpc)
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
    populate_dense_collection(remote_client, points)

    point_vectors = [
        models.PointVectors(id=1, vector=dense_doc_2),
        models.PointVectors(id=2, vector=dense_doc_3),
    ]

    local_client.update_vectors(COLLECTION_NAME, point_vectors)
    remote_client.update_vectors(COLLECTION_NAME, point_vectors)
    current_vectors = local_kwargs["points"]
    assert all([isinstance(vector.vector, list) for vector in current_vectors])

    compare_collections(
        local_client,
        remote_client,
        num_vectors=10,
        collection_name=COLLECTION_NAME,
    )

    local_client.delete_collection(COLLECTION_NAME)
    remote_client.delete_collection(COLLECTION_NAME)
    # endregion

    # region named
    points = [
        models.PointStruct(id=1, vector={"text": dense_doc_1}),
        models.PointStruct(id=2, vector={"text": dense_doc_2}),
    ]

    populate_dense_collection(local_client, points, vector_name="text")
    populate_dense_collection(remote_client, points, vector_name="text")

    point_vectors = [
        models.PointVectors(id=1, vector={"text": dense_doc_2}),
        models.PointVectors(id=2, vector={"text": dense_doc_3}),
    ]

    local_client.update_vectors(COLLECTION_NAME, point_vectors)
    remote_client.update_vectors(COLLECTION_NAME, point_vectors)
    current_vectors = local_kwargs["points"]
    assert all([isinstance(vector.vector["text"], list) for vector in current_vectors])

    compare_collections(
        local_client,
        remote_client,
        num_vectors=10,
        collection_name=COLLECTION_NAME,
    )

    local_client.delete_collection(COLLECTION_NAME)
    remote_client.delete_collection(COLLECTION_NAME)
    # endregion


@pytest.mark.parametrize("prefer_grpc", [True, False])
def test_query_points(prefer_grpc):
    local_client = QdrantClient(":memory:")
    if not local_client._FASTEMBED_INSTALLED:
        pytest.skip("FastEmbed is not installed, skipping")
    remote_client = QdrantClient(prefer_grpc=prefer_grpc)
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
        models.PointStruct(id=i, vector={"text": doc})
        for i, doc in enumerate(
            [sparse_doc_1, sparse_doc_2, sparse_doc_3, sparse_doc_4, sparse_doc_5]
        )
    ]

    populate_sparse_collection(local_client, points, vector_name="text")
    populate_sparse_collection(remote_client, points, vector_name="text")

    # region non-prefetch queries
    local_client.query_points(COLLECTION_NAME, sparse_doc_1, using="text")
    remote_client.query_points(COLLECTION_NAME, sparse_doc_1, using="text")
    current_query = local_kwargs["query"]
    assert isinstance(current_query.nearest, models.SparseVector)
    doc_point = local_client.retrieve(COLLECTION_NAME, ids=[0], with_vectors=True)[0]
    # assert that we generate different embeddings for doc and query
    assert not (
        np.allclose(doc_point.vector["text"].values, current_query.nearest.values, atol=1e-3)
    )

    nearest_query = models.NearestQuery(nearest=sparse_doc_1)
    local_client.query_points(COLLECTION_NAME, nearest_query, using="text")
    remote_client.query_points(COLLECTION_NAME, nearest_query, using="text")
    current_query = local_kwargs["query"]
    assert isinstance(current_query.nearest, models.SparseVector)

    recommend_query = models.RecommendQuery(
        recommend=models.RecommendInput(
            positive=[sparse_doc_1],
            negative=[sparse_doc_1],
        )
    )
    local_client.query_points(COLLECTION_NAME, recommend_query, using="text")
    remote_client.query_points(COLLECTION_NAME, recommend_query, using="text")
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
    local_client.query_points(COLLECTION_NAME, discover_query, using="text")
    remote_client.query_points(COLLECTION_NAME, discover_query, using="text")
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
    local_client.query_points(COLLECTION_NAME, discover_query_list, using="text")
    remote_client.query_points(COLLECTION_NAME, discover_query_list, using="text")
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
    local_client.query_points(COLLECTION_NAME, context_query, using="text")
    remote_client.query_points(COLLECTION_NAME, context_query, using="text")
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
    local_client.query_points(COLLECTION_NAME, context_query_list, using="text")
    remote_client.query_points(COLLECTION_NAME, context_query_list, using="text")
    current_query = local_kwargs["query"]
    contexts = current_query.context
    assert all(isinstance(context.positive, models.SparseVector) for context in contexts)
    assert all(isinstance(context.negative, models.SparseVector) for context in contexts)

    local_client.delete_collection(COLLECTION_NAME)
    remote_client.delete_collection(COLLECTION_NAME)
    # endregion

    # region prefetch queries
    # endregion


@pytest.mark.skip(reason="Not implemented")
@pytest.mark.parametrize("prefer_grpc", [True, False])
def test_query_batch_points(prefer_grpc):
    pass


@pytest.mark.skip(reason="Not implemented")
@pytest.mark.parametrize("prefer_grpc", [True, False])
def test_query_points_groups(prefer_grpc):
    pass


@pytest.mark.skip(reason="Not implemented")
def test_is_query_embed():
    pass


# @pytest.mark.parametrize("prefer_grpc", [False, True])
# def test_query_embeddings_prefetch(prefer_grpc):
#     def query_is_float_list(pref: BaseModel) -> bool:
#         nearest = pref.query
#         if isinstance(nearest, models.NearestQuery):
#             nearest = nearest.nearest
#         return isinstance(nearest, list) and isinstance(nearest[0], float)
#
#     model_name = "sentence-transformers/all-MiniLM-L6-v2"
#     remote_client = QdrantClient(prefer_grpc=prefer_grpc)
#     if not remote_client._FASTEMBED_INSTALLED:
#         pytest.skip("FastEmbed is not installed, skipping")
#
#     empty_list_prefetch = models.Prefetch(query=[0.2, 0.1], prefetch=[])
#     none_prefetch = models.Prefetch(query=[0.2, 0.1], prefetch=None)
#     assert remote_client._embed_models(empty_list_prefetch).prefetch == []
#     assert remote_client._embed_models(none_prefetch).prefetch is None
#
#     nearest_query = models.NearestQuery(
#         nearest=models.Document(text="nearest on prefetch", model=model_name)
#     )
#     prefetch = models.Prefetch(query=nearest_query)
#     converted_prefetch = remote_client._embed_models(prefetch)
#     assert query_is_float_list(converted_prefetch)
#
#     nested_prefetch = models.Prefetch(
#         query=nearest_query,
#         prefetch=models.Prefetch(
#             query=[0.2, 0.3],
#             prefetch=[
#                 models.Prefetch(
#                     query=models.Document(text="nested on prefetch", model=model_name),
#                     prefetch=models.Prefetch(
#                         query=models.Document(text="deep text", model=model_name)
#                     ),
#                 ),
#                 models.Prefetch(
#                     prefetch=[
#                         models.Prefetch(
#                             query=models.Document(text="another deep text", model=model_name)
#                         ),
#                         models.Prefetch(
#                             query=models.Document(text="yet another deep text", model=model_name)
#                         ),
#                         models.Prefetch(query=[0.2, 0.4]),
#                     ]
#                 ),
#             ],
#         ),
#     )
#     converted_nested_prefetch = remote_client._embed_models(nested_prefetch)
#     assert query_is_float_list(converted_nested_prefetch)  # nearest_query check
#
#     child_prefetch = converted_nested_prefetch.prefetch
#     grandchild_prefetches = child_prefetch.prefetch
#     assert query_is_float_list(grandchild_prefetches[0])  # "nested on prefetch" check
#     assert query_is_float_list(grandchild_prefetches[0].prefetch)  # "deep text" check
#     assert query_is_float_list(grandchild_prefetches[1].prefetch[0])  # "another deep text" check
#     assert query_is_float_list(grandchild_prefetches[1].prefetch[1])  # yet another deep text check
#
#
# def test_query_batch_points():
#     major, minor, patch, dev = read_version()
#     if major is not None and (major, minor, patch) < (1, 10, 0):
#         pytest.skip("Works as of qdrant 1.11.0")
#
#     local_client = QdrantClient(":memory:")
#     if not local_client._FASTEMBED_INSTALLED:
#         pytest.skip("FastEmbed is not installed, skipping")
#     remote_client = QdrantClient()
#     model_name = "sentence-transformers/all-MiniLM-L6-v2"
#     dim = 384
#     collection_name = "test-doc-embed"
#     local_client.create_collection(
#         collection_name,
#         vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE),
#     )
#     if remote_client.collection_exists(collection_name):
#         remote_client.delete_collection(collection_name)
#     remote_client.create_collection(
#         collection_name,
#         vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE),
#     )
#
#     texts = [
#         "It's a short document",
#         "Another short document",
#         "Document to check query requests",
#         "A nonsense document",
#     ]
#     points = [
#         models.PointStruct(id=i, vector=models.Document(text=text, model=model_name))
#         for i, text in enumerate(texts)
#     ]
#     # upload data
#     local_client.upsert(collection_name, points=points)
#     remote_client.upsert(collection_name, points=points)
#     query_requests = [
#         models.QueryRequest(
#             query=models.NearestQuery(
#                 nearest=models.Document(text="It's a short query", model=model_name)
#             )
#         ),
#         models.QueryRequest(
#             query=models.NearestQuery(nearest=[random.random() for _ in range(dim)])
#         ),
#     ]
#
#     compare_client_results(
#         remote_client,
#         local_client,
#         lambda c: c.query_batch_points(collection_name, requests=query_requests),
#     )
