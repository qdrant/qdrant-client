from qdrant_client import models, grpc
from qdrant_client.embed.utils import (
    inspect_query_types,
    inspect_query_and_prefetch_types,
    inspect_prefetch_types,
    inspect_points,
    inspect_query_requests,
    inspect_update_operations,
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


def test_inspect_query_requests():
    vector_only_query_request = models.QueryRequest(
        query=[0.2, 0.3],
    )

    assert not inspect_query_requests([vector_only_query_request])

    vector_only_prefetch_request = models.QueryRequest(prefetch=models.Prefetch(query=[0.2, 0.1]))

    assert not inspect_query_requests([vector_only_prefetch_request])

    document_only_query_request = models.QueryRequest(
        query=models.Document(text="123", model="Qdrant/bm42"),
    )

    assert inspect_query_requests([document_only_query_request])

    document_only_prefetch_request = models.QueryRequest(
        prefetch=models.Prefetch(query=models.Document(text="123", model="Qdrant/bm42"))
    )

    assert inspect_query_requests([document_only_prefetch_request])

    assert inspect_query_requests([vector_only_query_request, document_only_query_request])


def test_inspect_update_operations():
    non_relevant_ops = [
        models.DeleteOperation(delete=models.PointIdsList(points=[1, 2, 3])),
        models.SetPayloadOperation(set_payload=models.SetPayload(payload={"a": 2})),
        models.OverwritePayloadOperation(overwrite_payload=models.SetPayload(payload={"b": 3})),
        models.DeletePayloadOperation(delete_payload=models.DeletePayload(keys=["a", "c"])),
        models.ClearPayloadOperation(clear_payload=models.PointIdsList(points=[1, 4, 5])),
        models.DeleteVectorsOperation(delete_vectors=models.DeleteVectors(vector=["dense"])),
    ]
    assert not inspect_update_operations(non_relevant_ops)

    plain_points_batch = models.PointsBatch(
        batch=models.Batch(ids=[1, 2], vectors=[[0.1, 0.2], [0.3, 0.4]])
    )
    plain_point_structs = models.PointsList(
        points=[
            models.PointStruct(id=1, vector=[1.0, 2.0]),
            models.PointStruct(id=2, vector=[1.0, 3.0]),
        ]
    )
    doc_points_batch = models.PointsBatch(
        batch=models.Batch(
            ids=[1, 2],
            vectors=[
                models.Document(text="123", model="Qdrant/bm42"),
                models.Document(text="321", model="Qdrant/bm42"),
            ],
        ),
    )
    doc_point_struct = models.PointsList(
        points=[
            models.PointStruct(id=1, vector=models.Document(text="123", model="Qdrant/bm42")),
            models.PointStruct(id=2, vector=models.Document(text="321", model="Qdrant/bm42")),
        ]
    )

    plain_batch_upsert_op = models.UpsertOperation(upsert=plain_points_batch)
    plain_structs_upsert_op = models.UpsertOperation(upsert=plain_point_structs)
    doc_batch_upsert_op = models.UpsertOperation(upsert=doc_points_batch)
    doc_structs_upsert_op = models.UpsertOperation(upsert=doc_point_struct)

    assert not inspect_update_operations([plain_batch_upsert_op])
    assert not inspect_update_operations([plain_structs_upsert_op])
    assert inspect_update_operations([doc_batch_upsert_op])
    assert inspect_update_operations([doc_structs_upsert_op])

    assert inspect_update_operations([plain_batch_upsert_op, doc_structs_upsert_op])

    plain_point_vectors = models.PointVectors(id=1, vector=[0.2, 0.3])
    doc_point_vectors = models.PointVectors(
        id=2, vector=models.Document(text="123", model="Qdrant/bm42")
    )

    plain_point_vectors_update_op = models.UpdateVectorsOperation(
        update_vectors=models.UpdateVectors(points=[plain_point_vectors])
    )
    doc_point_vectors_update_op = models.UpdateVectorsOperation(
        update_vectors=models.UpdateVectors(points=[doc_point_vectors])
    )

    assert not inspect_update_operations([plain_point_vectors_update_op])
    assert inspect_update_operations([doc_point_vectors_update_op])

    assert inspect_update_operations([plain_point_vectors_update_op, doc_point_vectors_update_op])
