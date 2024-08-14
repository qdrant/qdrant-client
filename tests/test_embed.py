from qdrant_client import models, grpc
from qdrant_client.embed.utils import (
    inspect_query_types,
    inspect_query_and_prefetch_types,
    inspect_prefetch_types,
    inspect_points,
)


def test_inspect_query_types():
    assert not inspect_query_types(1)
    assert not inspect_query_types("1")
    assert not inspect_query_types([1.0, 2.0, 3.0])
    assert not inspect_query_types([[1.0, 2.0, 3.0]])
    assert not inspect_query_types(models.SparseVector(indices=[0, 1], values=[2.0, 3.0]))
    assert not inspect_query_types(models.NearestQuery(nearest=[1.0, 2.0]))
    assert not inspect_query_types(
        models.RecommendQuery(
            recommend=models.RecommendInput(positive=[[1.0, 2.0]], negative=[[-1.0, -2.0]])
        )
    )
    assert not inspect_query_types(
        models.DiscoverQuery(
            discover=models.DiscoverInput(
                target=[3.0, 4.0],
                context=models.ContextPair(positive=[1.0, 2.0], negative=[-1.0, -2.0]),
            )
        )
    )
    assert not inspect_query_types(
        models.ContextQuery(
            context=[models.ContextPair(positive=[1.0, 2.0], negative=[-1.0, -2.0])]
        )
    )
    assert not inspect_query_types(None)

    assert models.Document(text="123", model="Qdrant/bm42")


def test_inspect_prefetch_types():
    none_prefetch = models.Prefetch(query=None, prefetch=None)
    assert not inspect_prefetch_types(none_prefetch)

    vector_prefetch = models.Prefetch(query=[1.0, 2.0])
    assert not inspect_prefetch_types(vector_prefetch)

    doc_prefetch = models.Prefetch(query=models.Document(text="123", model="Qdrant/bm42"))
    assert inspect_prefetch_types(doc_prefetch)

    nested_prefetch = models.Prefetch(
        query=None,
        prefetch=models.Prefetch(query=models.Document(text="123", model="Qdrant/bm42")),
    )
    assert inspect_prefetch_types(nested_prefetch)

    vector_and_doc_prefetch = models.Prefetch(
        query=[1.0, 2.0],
        prefetch=models.Prefetch(query=models.Document(text="123", model="Qdrant/bm42")),
    )
    assert inspect_prefetch_types(vector_and_doc_prefetch)


def test_inspect_query_and_prefetch_types():
    none_query = None
    none_prefetch = None
    query = models.Document(text="123", model="Qdrant/bm42")
    prefetch = models.Prefetch(query=models.Document(text="123", model="Qdrant/bm42"))

    assert not inspect_query_and_prefetch_types(none_query, none_prefetch)
    assert inspect_query_and_prefetch_types(query, none_prefetch)
    assert inspect_query_and_prefetch_types(none_query, prefetch)
    assert inspect_query_and_prefetch_types(query, prefetch)


def test_inspect_points():
    vector_batch = models.Batch(ids=[1, 2], vectors=[[1.0, 2.0], [3.0, 4.0]])
    assert not inspect_points(vector_batch)

    document_batch = models.Batch(
        ids=[1, 2],
        vectors=[
            models.Document(text="123", model="Qdrant/bm42"),
            models.Document(text="324", model="Qdrant/bm42"),
        ],
    )
    assert inspect_points(document_batch)

    vector_points = [models.PointStruct(id=1, vector=[1.0, 2.0])]
    assert not inspect_points(vector_points)

    document_points = [
        models.PointStruct(id=1, vector=models.Document(text="123", model="Qdrant/bm42"))
    ]
    assert inspect_points(document_points)

    mixed_points = [
        models.PointStruct(id=1, vector=[1.0, 2.0]),
        models.PointStruct(id=1, vector=models.Document(text="123", model="Qdrant/bm42")),
    ]
    assert inspect_points(mixed_points)

    grpc_point = [
        grpc.PointStruct(
            id=grpc.PointId(num=3), vectors=grpc.Vectors(vector=grpc.Vector(data=[1.0, 2.0]))
        )
    ]
    assert not inspect_points(grpc_point)

    dict_batch = models.Batch(ids=[1, 2], vectors={"dense": [[1.0, 2.0]]})
    assert not inspect_points(dict_batch)

    dict_doc_batch = models.Batch(
        ids=[1, 2], vectors={"dense": [models.Document(text="123", model="Qdrant/bm42")]}
    )
    assert inspect_points(dict_doc_batch)

    multiple_keys_batch = models.Batch(
        ids=[1, 2], vectors={"dense": [[1.0, 2.0]], "dense-two": [[3.0, 4.0]]}
    )
    assert not inspect_points(multiple_keys_batch)

    multiple_keys_mixed_types_batch = models.Batch(
        ids=[1, 2],
        vectors={
            "dense": [[1.0, 2.0]],
            "dense-two": [models.Document(text="123", model="Qdrant/bm42")],
        },
    )
    assert inspect_points(multiple_keys_mixed_types_batch)
