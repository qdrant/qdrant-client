import numpy as np

from qdrant_client import models, grpc
from qdrant_client.embed.type_inspector import Inspector


def test_inspect_query_types():
    inspector = Inspector()

    # region negative cases
    # region ExtendedPointId
    assert not inspector.inspect(1)  # type: ignore
    assert not inspector.inspect("1")  # type: ignore
    # endregion

    # region plain vectors
    assert not inspector.inspect([1.0, 2.0, 3.0])
    assert not inspector.inspect([[1.0, 2.0, 3.0]])
    assert not inspector.inspect(models.SparseVector(indices=[0, 1], values=[2.0, 3.0]))
    assert not inspector.inspect(np.array([1.0, 2.0, 3.0]))
    assert not inspector.inspect(np.array([[1.0, 2.0, 3.0]]))
    # endregion

    # region NearestQuery
    assert not inspector.inspect(models.NearestQuery(nearest=[1.0, 2.0]))
    assert not inspector.inspect(models.NearestQuery(nearest=[[1.0, 2.0], [3.0, 4.0]]))
    assert not inspector.inspect(models.NearestQuery(nearest=1))
    assert not inspector.inspect(models.NearestQuery(nearest="1"))
    # endregion

    # region RecommendQuery
    assert not inspector.inspect(
        models.RecommendQuery(
            recommend=models.RecommendInput(positive=[[1.0, 2.0]], negative=[[-1.0, -2.0]])
        )
    )
    # endregion

    # region DiscoverQuery
    assert not inspector.inspect(
        models.DiscoverQuery(
            discover=models.DiscoverInput(
                target=[3.0, 4.0],
                context=models.ContextPair(positive=[1.0, 2.0], negative=[-1.0, -2.0]),
            )
        )
    )
    # endregion

    # region ContextQuery
    assert not inspector.inspect(
        models.ContextQuery(
            context=[models.ContextPair(positive=[1.0, 2.0], negative=[-1.0, -2.0])]
        )
    )
    # endregion

    # region Non-vector queries
    assert not inspector.inspect(models.OrderByQuery(order_by="1"))
    assert not inspector.inspect(
        models.OrderByQuery(order_by=models.OrderBy(key="1", direction="asc"))
    )

    assert not inspector.inspect(models.FusionQuery(fusion=models.Fusion.DBSF))

    assert not inspector.inspect(models.SampleQuery(sample=models.Sample.RANDOM))
    # endregion negative cases

    # region positive cases
    doc = models.Document(text="123", model="Qdrant/bm25")
    assert inspector.inspect(doc)

    assert inspector.inspect(models.NearestQuery(nearest=doc))

    # region RecommendQuery
    assert inspector.inspect(
        models.RecommendQuery(
            recommend=models.RecommendInput(positive=[doc], negative=[[-1.0, -2.0]])
        )
    )
    assert inspector.inspect(
        models.RecommendQuery(
            recommend=models.RecommendInput(positive=[[1.0, 2.0]], negative=[doc])
        )
    )
    # endregion

    # region DiscoverQuery
    assert inspector.inspect(
        models.DiscoverQuery(
            discover=models.DiscoverInput(
                target=doc,
                context=models.ContextPair(positive=[1.0, 2.0], negative=[-1.0, -2.0]),
            )
        )
    )
    assert inspector.inspect(
        models.DiscoverQuery(
            discover=models.DiscoverInput(
                target=[3.0, 4.0],
                context=models.ContextPair(positive=doc, negative=[-1.0, -2.0]),
            )
        )
    )

    assert inspector.inspect(
        models.DiscoverQuery(
            discover=models.DiscoverInput(
                target=[3.0, 4.0],
                context=models.ContextPair(positive=[1.0, 2.0], negative=doc),
            )
        )
    )
    # endregion

    # region ContextQuery
    assert inspector.inspect(
        models.ContextQuery(context=models.ContextPair(positive=doc, negative=[-1.0, -2.0]))
    )
    assert inspector.inspect(
        models.ContextQuery(context=models.ContextPair(positive=[1.0, 2.0], negative=doc))
    )
    assert inspector.inspect(
        models.ContextQuery(context=[models.ContextPair(positive=doc, negative=[-1.0, -2.0])])
    )
    assert inspector.inspect(
        models.ContextQuery(context=[models.ContextPair(positive=[1.0, 2.0], negative=doc)])
    )
    # endregion
    # endregion positive cases


def test_inspect_prefetch_types():
    inspector = Inspector()

    # region negative cases
    none_prefetch = models.Prefetch(query=None, prefetch=None)
    assert not inspector.inspect(none_prefetch)

    vector_prefetch = models.Prefetch(query=[1.0, 2.0])
    assert not inspector.inspect(vector_prefetch)

    deep_nested_prefetch_wo_doc = models.Prefetch(
        query=[[0.1, 0.2]],
        prefetch=models.Prefetch(
            query=[[0.2, 0.3]],
            prefetch=models.Prefetch(
                query=[[0.3, 0.4]], prefetch=models.Prefetch(query=[0.2, 0.3])
            ),
        ),
    )
    assert not inspector.inspect(deep_nested_prefetch_wo_doc)
    assert not inspector.inspect([None, deep_nested_prefetch_wo_doc])
    # endregion

    # region positive cases
    doc = models.Document(text="123", model="Qdrant/bm25")

    doc_prefetch = models.Prefetch(query=doc)
    assert inspector.inspect(doc_prefetch)

    nested_prefetch = models.Prefetch(
        query=None,
        prefetch=models.Prefetch(query=doc),
    )
    assert inspector.inspect(nested_prefetch)

    vector_and_doc_prefetch = models.Prefetch(
        query=[1.0, 2.0],
        prefetch=models.Prefetch(query=doc),
    )
    assert inspector.inspect(vector_and_doc_prefetch)

    deep_nested_prefetch = models.Prefetch(
        query=[[0.1, 0.2]],
        prefetch=models.Prefetch(
            query=[[0.2, 0.3]],
            prefetch=models.Prefetch(query=[[0.3, 0.4]], prefetch=models.Prefetch(query=doc)),
        ),
    )
    assert inspector.inspect(deep_nested_prefetch)
    assert inspector.inspect([None, deep_nested_prefetch])
    # endregion


def test_inspect_query_requests():
    inspector = Inspector()

    # region negative cases
    vector = [0.2, 0.3]
    nearest_query = models.NearestQuery(nearest=vector)

    query_request_vector = models.QueryRequest(
        query=vector,
    )
    assert not inspector.inspect(query_request_vector)

    query_request_nearest_vector = models.QueryRequest(
        query=nearest_query,
    )
    assert not inspector.inspect(query_request_nearest_vector)

    vector_only_prefetch_request = models.QueryRequest(prefetch=models.Prefetch(query=[0.2, 0.1]))
    assert not inspector.inspect([vector_only_prefetch_request])

    deep_nested_prefetch_vector = models.Prefetch(
        query=[[0.1, 0.2]],
        prefetch=models.Prefetch(
            query=[[0.2, 0.3]],
            prefetch=models.Prefetch(
                query=[[0.3, 0.4]], prefetch=models.Prefetch(query=[0.2, 0.3])
            ),
        ),
    )
    deep_nested_prefetch_vector_request = models.QueryRequest(
        prefetch=deep_nested_prefetch_vector,
    )
    assert not inspector.inspect(deep_nested_prefetch_vector_request)

    query_groups_request_vector = models.QueryGroupsRequest(
        query=nearest_query,
        group_by="k",
    )
    assert not inspector.inspect(query_groups_request_vector)

    query_groups_request_prefetch_vector = models.QueryGroupsRequest(
        prefetch=models.Prefetch(query=nearest_query),
        group_by="k",
    )
    assert not inspector.inspect(query_groups_request_prefetch_vector)

    query_groups_request_deep_nested_prefetch_vector = models.QueryGroupsRequest(
        prefetch=deep_nested_prefetch_vector,
        group_by="k",
    )
    assert not inspector.inspect(query_groups_request_deep_nested_prefetch_vector)

    query_batch_request_vector = models.QueryRequestBatch(searches=[query_request_vector])
    assert not inspector.inspect(query_batch_request_vector)

    query_batch_request_nearest_vector = models.QueryRequestBatch(
        searches=[query_request_nearest_vector]
    )
    assert not inspector.inspect(query_batch_request_nearest_vector)

    query_batch_request_prefetch_vector = models.QueryRequestBatch(
        searches=[vector_only_prefetch_request]
    )
    assert not inspector.inspect(query_batch_request_prefetch_vector)

    query_batch_request_deep_nested_prefetch_vector = models.QueryRequestBatch(
        searches=[deep_nested_prefetch_vector_request]
    )
    assert not inspector.inspect(query_batch_request_deep_nested_prefetch_vector)
    # endregion

    # region positive cases
    doc = models.Document(text="123", model="Qdrant/bm25")

    document_only_query_request = models.QueryRequest(
        query=doc,
    )
    assert inspector.inspect([document_only_query_request])

    document_only_prefetch_request = models.QueryRequest(prefetch=models.Prefetch(query=doc))
    assert inspector.inspect([document_only_prefetch_request])
    assert inspector.inspect([query_request_vector, document_only_query_request])

    deep_nested_prefetch_doc = models.Prefetch(
        query=[[0.1, 0.2]],
        prefetch=models.Prefetch(
            query=[[0.2, 0.3]],
            prefetch=models.Prefetch(query=[[0.3, 0.4]], prefetch=models.Prefetch(query=doc)),
        ),
    )
    deep_nested_prefetch_doc_request = models.QueryRequest(
        prefetch=deep_nested_prefetch_doc,
    )
    assert inspector.inspect(deep_nested_prefetch_doc_request)

    query_groups_request_doc = models.QueryGroupsRequest(
        query=doc,
        group_by="k",
    )
    assert inspector.inspect(query_groups_request_doc)

    query_groups_request_prefetch_doc = models.QueryGroupsRequest(
        prefetch=models.Prefetch(query=doc),
        group_by="k",
    )
    assert inspector.inspect(query_groups_request_prefetch_doc)

    query_groups_request_deep_nested_prefetch_doc = models.QueryGroupsRequest(
        prefetch=deep_nested_prefetch_doc,
        group_by="k",
    )
    assert inspector.inspect(query_groups_request_deep_nested_prefetch_doc)

    query_batch_request_doc = models.QueryRequestBatch(searches=[document_only_query_request])
    assert inspector.inspect(query_batch_request_doc)

    query_batch_request_prefetch_doc = models.QueryRequestBatch(
        searches=[document_only_prefetch_request]
    )

    assert inspector.inspect(query_batch_request_prefetch_doc)
    assert inspector.inspect([query_batch_request_vector, query_batch_request_doc])

    query_batch_request_deep_nested_prefetch_doc = models.QueryRequestBatch(
        searches=[deep_nested_prefetch_doc_request]
    )
    assert inspector.inspect(query_batch_request_deep_nested_prefetch_doc)
    # endregion


def test_inspect_upsert_points():
    inspector = Inspector()

    # region negative cases
    vector_batch = models.Batch(ids=[1, 2], vectors=[[1.0, 2.0], [3.0, 4.0]])
    assert not inspector.inspect(vector_batch)

    vector_points = [
        models.PointStruct(id=1, vector=[1.0, 2.0]),
        models.PointStruct(id=2, vector=[3.0, 3.0]),
    ]
    assert not inspector.inspect(vector_points)

    grpc_points = [
        grpc.PointStruct(
            id=grpc.PointId(num=3), vectors=grpc.Vectors(vector=grpc.Vector(data=[1.0, 2.0]))
        ),
        grpc.PointStruct(
            id=grpc.PointId(num=4), vectors=grpc.Vectors(vector=grpc.Vector(data=[3.0, 3.0]))
        ),
    ]
    assert not inspector.inspect(grpc_points)

    multiple_keys_batch = models.Batch(
        ids=[1, 2], vectors={"dense": [[1.0, 2.0]], "dense-two": [[3.0, 4.0]]}
    )
    assert not inspector.inspect(multiple_keys_batch)

    dict_vector_points = [
        models.PointStruct(id=1, vector={"dense": [1.0, 2.0]}),
        models.PointStruct(id=2, vector={"dense": [2.0, 3.0]}),
    ]
    assert not inspector.inspect(dict_vector_points)

    multiple_keys_points = [
        models.PointStruct(id=1, vector={"dense": [1.0, 2.0], "dense-two": [3.0, 4.0]}),
        models.PointStruct(id=2, vector={"dense": [2.0, 3.0]}),
    ]
    assert not inspector.inspect(multiple_keys_points)
    # endregion negative cases

    # region positive cases
    doc_1 = models.Document(text="123", model="Qdrant/bm25")
    doc_2 = models.Document(text="321", model="Qdrant/bm25")

    document_batch = models.Batch(
        ids=[1, 2],
        vectors=[
            doc_1,
            doc_2,
        ],
    )
    assert inspector.inspect(document_batch)

    document_points = [
        models.PointStruct(id=1, vector=doc_1),
        models.PointStruct(id=2, vector=doc_2),
    ]
    assert inspector.inspect(document_points)

    mixed_points_doc_first = [
        models.PointStruct(id=1, vector=doc_1),
        models.PointStruct(id=2, vector=[0.2, 0.3]),
    ]
    assert inspector.inspect(mixed_points_doc_first)

    mixed_points_doc_second = [
        models.PointStruct(id=1, vector=[0.2, 0.3]),
        models.PointStruct(id=2, vector=doc_2),
    ]
    assert inspector.inspect(mixed_points_doc_second)

    dict_doc_batch = models.Batch(ids=[1], vectors={"dense": [doc_1]})
    assert inspector.inspect(dict_doc_batch)

    dict_mixed_batch = models.Batch(
        ids=[1, 2], vectors={"dense": [[1.0, 2.0]], "dense-two": [doc_1]}
    )
    assert inspector.inspect(dict_mixed_batch)
    # endregion


def test_inspect_update_operations():
    inspector = Inspector()

    # region negative cases
    non_relevant_ops = [
        models.DeleteOperation(delete=models.PointIdsList(points=[1, 2, 3])),
        models.SetPayloadOperation(set_payload=models.SetPayload(payload={"a": 2})),
        models.OverwritePayloadOperation(overwrite_payload=models.SetPayload(payload={"b": 3})),
        models.DeletePayloadOperation(delete_payload=models.DeletePayload(keys=["a", "c"])),
        models.ClearPayloadOperation(clear_payload=models.PointIdsList(points=[1, 4, 5])),
        models.DeleteVectorsOperation(delete_vectors=models.DeleteVectors(vector=["dense"])),
    ]
    assert not inspector.inspect(non_relevant_ops)

    plain_points_batch = models.PointsBatch(
        batch=models.Batch(ids=[1, 2], vectors=[[0.1, 0.2], [0.3, 0.4]])
    )
    assert not inspector.inspect(plain_points_batch)

    plain_point_list = models.PointsList(
        points=[
            models.PointStruct(id=1, vector=[1.0, 2.0]),
            models.PointStruct(id=2, vector=[1.0, 3.0]),
        ]
    )
    assert not inspector.inspect(plain_point_list)

    plain_point_vectors = models.PointVectors(id=1, vector=[0.2, 0.3])
    assert not inspector.inspect(plain_point_vectors)

    plain_batch_upsert_op = models.UpsertOperation(upsert=plain_points_batch)
    assert not inspector.inspect([plain_batch_upsert_op])

    plain_structs_upsert_op = models.UpsertOperation(upsert=plain_point_list)
    assert not inspector.inspect([plain_structs_upsert_op])

    plain_point_vectors_update_op = models.UpdateVectorsOperation(
        update_vectors=models.UpdateVectors(points=[plain_point_vectors])
    )
    assert not inspector.inspect([plain_point_vectors_update_op])
    # endregion

    # region positive cases
    doc_1 = models.Document(text="123", model="Qdrant/bm25")
    doc_2 = models.Document(text="321", model="Qdrant/bm25")

    doc_points_batch = models.PointsBatch(
        batch=models.Batch(
            ids=[1, 2],
            vectors=[
                doc_1,
                doc_2,
            ],
        ),
    )
    assert inspector.inspect(doc_points_batch)

    doc_points_list = models.PointsList(
        points=[
            models.PointStruct(id=1, vector=doc_1),
            models.PointStruct(id=2, vector=doc_2),
        ]
    )
    assert inspector.inspect(doc_points_list)

    mixed_points_list = models.PointsList(
        points=[
            models.PointStruct(id=1, vector=[0.2, 0.3]),
            models.PointStruct(id=2, vector=doc_2),
        ]
    )
    assert inspector.inspect(mixed_points_list)

    doc_point_vectors = [models.PointVectors(id=2, vector=doc_1)]
    assert inspector.inspect(doc_point_vectors)

    mixed_point_vectors = [
        models.PointVectors(id=2, vector=[0.2, 0.3]),
        models.PointVectors(id=3, vector=doc_2),
    ]
    assert inspector.inspect(mixed_point_vectors)

    doc_batch_upsert_op = models.UpsertOperation(upsert=doc_points_batch)
    assert inspector.inspect([doc_batch_upsert_op])

    doc_points_list_upsert_op = models.UpsertOperation(upsert=doc_points_list)
    assert inspector.inspect([doc_points_list_upsert_op])

    mixed_points_list_upsert_op = models.UpsertOperation(upsert=mixed_points_list)
    assert inspector.inspect([mixed_points_list_upsert_op])

    assert inspector.inspect([plain_batch_upsert_op, doc_points_list_upsert_op])

    doc_point_vectors_update_op = models.UpdateVectorsOperation(
        update_vectors=models.UpdateVectors(points=doc_point_vectors)
    )
    assert inspector.inspect([doc_point_vectors_update_op])
    assert inspector.inspect([plain_point_vectors_update_op, doc_point_vectors_update_op])

    mixed_point_vectors_update_op = models.UpdateVectorsOperation(
        update_vectors=models.UpdateVectors(points=mixed_point_vectors)
    )
    assert inspector.inspect([mixed_point_vectors_update_op])
    # endregion
