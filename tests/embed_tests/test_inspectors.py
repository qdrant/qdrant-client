import numpy as np

from qdrant_client import models, grpc
from qdrant_client.embed.type_inspector import Inspector
from qdrant_client.embed.embed_inspector import InspectorEmbed


def test_inspect_query_types():
    inspector = Inspector()
    inspector_embed = InspectorEmbed()

    # region negative cases
    # region ExtendedPointId
    assert not inspector.inspect(1)  # type: ignore
    assert inspector_embed.inspect(1) == []  # type: ignore
    assert not inspector.inspect("1")  # type: ignore
    assert inspector_embed.inspect("1") == []  # type: ignore
    # endregion

    # region plain vectors
    vec = [1.0, 2.0, 3.0]
    assert not inspector.inspect(vec)
    assert inspector_embed.inspect(vec) == []

    multi_vec = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    assert not inspector.inspect(multi_vec)
    assert inspector_embed.inspect(multi_vec) == []

    sparse_vec = models.SparseVector(indices=[0, 1], values=[2.0, 3.0])
    assert not inspector.inspect(sparse_vec)
    assert inspector_embed.inspect(sparse_vec) == []

    np_vec = np.array([1.0, 2.0, 3.0])
    assert not inspector.inspect(np_vec)
    assert inspector_embed.inspect(np_vec) == []

    np_multi_vec = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    assert not inspector.inspect(np_multi_vec)
    assert inspector_embed.inspect(np_multi_vec) == []
    # endregion

    # region NearestQuery
    nq_id = models.NearestQuery(nearest=1)
    assert not inspector.inspect(nq_id)
    assert inspector_embed.inspect(nq_id) == []

    nq_str_id = models.NearestQuery(nearest="1")
    assert not inspector.inspect(nq_str_id)
    assert inspector_embed.inspect(nq_str_id) == []

    nq_vec = models.NearestQuery(nearest=vec)
    assert not inspector.inspect(nq_vec)
    assert inspector_embed.inspect(nq_vec) == []

    nq_multi_vec = models.NearestQuery(nearest=multi_vec)
    assert not inspector.inspect(nq_multi_vec)
    assert inspector_embed.inspect(nq_multi_vec) == []

    nq_sparse_vec = models.NearestQuery(nearest=sparse_vec)
    assert not inspector.inspect(nq_sparse_vec)
    assert inspector_embed.inspect(nq_sparse_vec) == []
    # endregion

    # region RecommendQuery
    rq_vec = models.RecommendQuery(recommend=models.RecommendInput(positive=[vec], negative=[vec]))
    assert not inspector.inspect(rq_vec)
    assert inspector_embed.inspect(rq_vec) == []
    # endregion

    # region DiscoverQuery
    dq_vec = models.DiscoverQuery(
        discover=models.DiscoverInput(
            target=vec,
            context=models.ContextPair(positive=vec, negative=vec),
        )
    )
    assert not inspector.inspect(dq_vec)
    assert inspector_embed.inspect(dq_vec) == []
    # endregion

    # region ContextQuery
    cq_vec = models.ContextQuery(context=models.ContextPair(positive=vec, negative=vec))
    assert not inspector.inspect(cq_vec)
    assert inspector_embed.inspect(cq_vec) == []
    # endregion

    # region Non-vector queries
    order_by_plain_query = models.OrderByQuery(order_by="1")
    assert not inspector.inspect(order_by_plain_query)
    assert inspector_embed.inspect(order_by_plain_query) == []

    order_by_query = models.OrderByQuery(order_by=models.OrderBy(key="1", direction="asc"))
    assert not inspector.inspect(order_by_query)
    assert inspector_embed.inspect(order_by_query) == []

    fusion_query = models.FusionQuery(fusion=models.Fusion.DBSF)
    assert not inspector.inspect(fusion_query)
    assert inspector_embed.inspect(fusion_query) == []

    sample_query = models.SampleQuery(sample=models.Sample.RANDOM)
    assert not inspector.inspect(sample_query)
    assert inspector_embed.inspect(sample_query) == []
    # endregion negative cases

    # region positive cases
    doc = models.Document(text="123", model="Qdrant/bm25")
    assert inspector.inspect(doc)
    assert inspector_embed.inspect(doc) == []

    nq_doc = models.NearestQuery(nearest=doc)
    assert inspector.inspect(nq_doc)
    paths = inspector_embed.inspect(nq_doc)
    assert len(paths) == 1 and paths[0].as_str_list() == ["nearest"]

    # region RecommendQuery
    rq_doc = models.RecommendQuery(recommend=models.RecommendInput(positive=[doc], negative=[vec]))
    assert inspector.inspect(rq_doc)
    paths = inspector_embed.inspect(rq_doc)
    assert len(paths) == 1 and paths[0].as_str_list() == ["recommend.positive"]

    rq_doc_1 = models.RecommendQuery(
        recommend=models.RecommendInput(positive=[vec], negative=[doc])
    )
    assert inspector.inspect(rq_doc_1)
    paths = inspector_embed.inspect(rq_doc_1)
    assert len(paths) == 1 and paths[0].as_str_list() == ["recommend.negative"]

    rq_doc_2 = models.RecommendQuery(
        recommend=models.RecommendInput(positive=[doc], negative=[doc])
    )
    assert inspector.inspect(rq_doc_2)
    paths = inspector_embed.inspect(rq_doc_2)
    assert len(paths) == 1 and set(paths[0].as_str_list()) == {
        "recommend.positive",
        "recommend.negative",
    }
    # endregion

    # region DiscoverQuery
    dq_target_doc = models.DiscoverQuery(
        discover=models.DiscoverInput(
            target=doc,
            context=models.ContextPair(positive=[vec], negative=[vec]),
        )
    )
    assert inspector.inspect(dq_target_doc)
    paths = inspector_embed.inspect(dq_target_doc)
    assert len(paths) == 1 and paths[0].as_str_list() == ["discover.target"]

    dq_pos_context_doc = models.DiscoverQuery(
        discover=models.DiscoverInput(
            target=vec,
            context=models.ContextPair(positive=doc, negative=vec),
        )
    )
    assert inspector.inspect(dq_pos_context_doc)
    paths = inspector_embed.inspect(dq_pos_context_doc)
    assert len(paths) == 1 and paths[0].as_str_list() == ["discover.context.positive"]

    dq_neg_context_doc = models.DiscoverQuery(
        discover=models.DiscoverInput(
            target=vec,
            context=models.ContextPair(positive=vec, negative=doc),
        )
    )
    assert inspector.inspect(dq_neg_context_doc)
    paths = inspector_embed.inspect(dq_neg_context_doc)
    assert len(paths) == 1 and paths[0].as_str_list() == ["discover.context.negative"]
    # endregion

    # region ContextQuery
    cq_pos_doc = models.ContextQuery(context=models.ContextPair(positive=doc, negative=vec))
    assert inspector.inspect(cq_pos_doc)
    paths = inspector_embed.inspect(cq_pos_doc)
    assert len(paths) == 1 and paths[0].as_str_list() == ["context.positive"]

    cq_neg_doc = models.ContextQuery(context=models.ContextPair(positive=vec, negative=doc))
    assert inspector.inspect(cq_neg_doc)
    paths = inspector_embed.inspect(cq_neg_doc)
    assert len(paths) == 1 and paths[0].as_str_list() == ["context.negative"]

    cq_both_doc = models.ContextQuery(context=models.ContextPair(positive=doc, negative=doc))
    assert inspector.inspect(cq_both_doc)
    paths = inspector_embed.inspect(cq_both_doc)
    assert len(paths) == 1 and set(paths[0].as_str_list()) == {
        "context.positive",
        "context.negative",
    }

    cq_list_pos_doc = models.ContextQuery(context=[models.ContextPair(positive=doc, negative=vec)])
    assert inspector.inspect(cq_list_pos_doc)
    paths = inspector_embed.inspect(cq_list_pos_doc)
    assert len(paths) == 1 and paths[0].as_str_list() == ["context.positive"]

    cq_list_neg_doc = models.ContextQuery(context=[models.ContextPair(positive=vec, negative=doc)])
    assert inspector.inspect(cq_list_neg_doc)
    paths = inspector_embed.inspect(cq_list_neg_doc)
    assert len(paths) == 1 and paths[0].as_str_list() == ["context.negative"]

    cq_list_both_doc = models.ContextQuery(
        context=[models.ContextPair(positive=doc, negative=doc)]
    )
    assert inspector.inspect(cq_list_both_doc)
    paths = inspector_embed.inspect(cq_list_both_doc)
    assert len(paths) == 1 and set(paths[0].as_str_list()) == {
        "context.positive",
        "context.negative",
    }
    # endregion
    # endregion positive cases


def test_inspect_prefetch_types():
    inspector = Inspector()
    inspector_embed = InspectorEmbed()

    # region negative cases
    none_prefetch = models.Prefetch(query=None, prefetch=None)
    assert not inspector.inspect(none_prefetch)
    assert inspector_embed.inspect(none_prefetch) == []

    vector_prefetch = models.Prefetch(query=[1.0, 2.0])
    assert not inspector.inspect(vector_prefetch)
    assert inspector_embed.inspect(vector_prefetch) == []

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
    assert inspector_embed.inspect(deep_nested_prefetch_wo_doc) == []
    assert not inspector.inspect([None, deep_nested_prefetch_wo_doc])
    assert inspector_embed.inspect([None, deep_nested_prefetch_wo_doc]) == []
    # endregion

    # region positive cases
    doc = models.Document(text="123", model="Qdrant/bm25")

    doc_prefetch = models.Prefetch(query=doc)
    assert inspector.inspect(doc_prefetch)
    paths = inspector_embed.inspect(doc_prefetch)
    assert len(paths) == 1 and paths[0].as_str_list() == ["query"]

    nested_prefetch = models.Prefetch(
        query=None,
        prefetch=models.Prefetch(query=doc),
    )
    assert inspector.inspect(nested_prefetch)
    paths = inspector_embed.inspect(nested_prefetch)
    assert len(paths) == 1 and paths[0].as_str_list() == ["prefetch.query"]

    vector_and_doc_prefetch = models.Prefetch(
        query=[1.0, 2.0],
        prefetch=models.Prefetch(query=doc),
    )
    assert inspector.inspect(vector_and_doc_prefetch)
    paths = inspector_embed.inspect(vector_and_doc_prefetch)
    assert len(paths) == 1 and paths[0].as_str_list() == ["prefetch.query"]

    deep_nested_prefetch = models.Prefetch(
        query=[[0.1, 0.2]],
        prefetch=models.Prefetch(
            query=[[0.2, 0.3]],
            prefetch=models.Prefetch(query=[[0.3, 0.4]], prefetch=models.Prefetch(query=doc)),
        ),
    )
    assert inspector.inspect(deep_nested_prefetch)
    paths = inspector_embed.inspect(deep_nested_prefetch)
    assert len(paths) == 1 and paths[0].as_str_list() == ["prefetch.prefetch.prefetch.query"]

    assert inspector.inspect([None, deep_nested_prefetch])
    paths = inspector_embed.inspect([None, deep_nested_prefetch])
    assert len(paths) == 1 and paths[0].as_str_list() == [
        "prefetch.prefetch.prefetch.query"
    ]  # todo: should return a list per model

    # endregion


def test_inspect_query_requests():
    inspector = Inspector()
    inspector_embed = InspectorEmbed()

    # region negative cases
    vector = [0.2, 0.3]
    nearest_query = models.NearestQuery(nearest=vector)

    query_request_vector = models.QueryRequest(
        query=vector,
    )
    assert not inspector.inspect(query_request_vector)
    assert inspector_embed.inspect(query_request_vector) == []

    query_request_nearest_vector = models.QueryRequest(
        query=nearest_query,
    )
    assert not inspector.inspect(query_request_nearest_vector)
    assert inspector_embed.inspect(query_request_nearest_vector) == []

    vector_only_prefetch_request = models.QueryRequest(prefetch=models.Prefetch(query=[0.2, 0.1]))
    assert not inspector.inspect([vector_only_prefetch_request])
    assert inspector_embed.inspect([vector_only_prefetch_request]) == []

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
    assert inspector_embed.inspect(deep_nested_prefetch_vector_request) == []

    query_groups_request_vector = models.QueryGroupsRequest(
        query=nearest_query,
        group_by="k",
    )
    assert not inspector.inspect(query_groups_request_vector)
    assert inspector_embed.inspect(query_groups_request_vector) == []

    query_groups_request_prefetch_vector = models.QueryGroupsRequest(
        prefetch=models.Prefetch(query=nearest_query),
        group_by="k",
    )
    assert not inspector.inspect(query_groups_request_prefetch_vector)
    assert inspector_embed.inspect(query_groups_request_prefetch_vector) == []

    query_groups_request_deep_nested_prefetch_vector = models.QueryGroupsRequest(
        prefetch=deep_nested_prefetch_vector,
        group_by="k",
    )
    assert not inspector.inspect(query_groups_request_deep_nested_prefetch_vector)
    assert inspector_embed.inspect(query_groups_request_deep_nested_prefetch_vector) == []

    query_batch_request_vector = models.QueryRequestBatch(searches=[query_request_vector])
    assert not inspector.inspect(query_batch_request_vector)
    assert inspector_embed.inspect(query_batch_request_vector) == []

    query_batch_request_nearest_vector = models.QueryRequestBatch(
        searches=[query_request_nearest_vector]
    )
    assert not inspector.inspect(query_batch_request_nearest_vector)
    assert inspector_embed.inspect(query_batch_request_nearest_vector) == []

    query_batch_request_prefetch_vector = models.QueryRequestBatch(
        searches=[vector_only_prefetch_request]
    )
    assert not inspector.inspect(query_batch_request_prefetch_vector)
    assert inspector_embed.inspect(query_batch_request_prefetch_vector) == []

    query_batch_request_deep_nested_prefetch_vector = models.QueryRequestBatch(
        searches=[deep_nested_prefetch_vector_request]
    )
    assert not inspector.inspect(query_batch_request_deep_nested_prefetch_vector)
    assert inspector_embed.inspect(query_batch_request_deep_nested_prefetch_vector) == []
    # endregion

    # region positive cases
    doc = models.Document(text="123", model="Qdrant/bm25")

    document_only_query_request = models.QueryRequest(
        query=doc,
    )
    assert inspector.inspect([document_only_query_request])
    paths = inspector_embed.inspect([document_only_query_request])
    assert len(paths) == 1 and paths[0].as_str_list() == ["query"]

    document_only_prefetch_request = models.QueryRequest(prefetch=models.Prefetch(query=doc))
    assert inspector.inspect([document_only_prefetch_request])
    paths = inspector_embed.inspect([document_only_prefetch_request])
    assert len(paths) == 1 and paths[0].as_str_list() == ["prefetch.query"]

    assert inspector.inspect([query_request_vector, document_only_query_request])
    paths = inspector_embed.inspect([query_request_vector, document_only_query_request])
    assert len(paths) == 1 and paths[0].as_str_list() == ["query"]

    deep_nested_prefetch_doc = models.Prefetch(
        query=[[0.1, 0.2]],
        prefetch=models.Prefetch(
            query=[[0.2, 0.3]],
            prefetch=models.Prefetch(query=[[0.3, 0.4]], prefetch=models.Prefetch(query=doc)),
        ),
    )
    assert inspector.inspect(deep_nested_prefetch_doc)
    paths = inspector_embed.inspect(deep_nested_prefetch_doc)
    assert len(paths) == 1 and paths[0].as_str_list() == ["prefetch.prefetch.prefetch.query"]

    deep_nested_prefetch_doc_request = models.QueryRequest(
        prefetch=deep_nested_prefetch_doc,
    )
    assert inspector.inspect(deep_nested_prefetch_doc_request)
    paths = inspector_embed.inspect(deep_nested_prefetch_doc_request)
    assert len(paths) == 1 and paths[0].as_str_list() == [
        "prefetch.prefetch.prefetch.prefetch.query"
    ]

    query_groups_request_doc = models.QueryGroupsRequest(
        query=doc,
        group_by="k",
    )
    assert inspector.inspect(query_groups_request_doc)
    paths = inspector_embed.inspect(query_groups_request_doc)
    assert len(paths) == 1 and paths[0].as_str_list() == ["query"]

    query_groups_request_prefetch_doc = models.QueryGroupsRequest(
        prefetch=models.Prefetch(query=doc),
        group_by="k",
    )
    assert inspector.inspect(query_groups_request_prefetch_doc)
    paths = inspector_embed.inspect(query_groups_request_prefetch_doc)
    assert len(paths) == 1 and paths[0].as_str_list() == ["prefetch.query"]

    query_groups_request_deep_nested_prefetch_doc = models.QueryGroupsRequest(
        prefetch=deep_nested_prefetch_doc,
        group_by="k",
    )
    assert inspector.inspect(query_groups_request_deep_nested_prefetch_doc)
    paths = inspector_embed.inspect(query_groups_request_deep_nested_prefetch_doc)
    assert len(paths) == 1 and paths[0].as_str_list() == [
        "prefetch.prefetch.prefetch.prefetch.query"
    ]

    query_batch_request_doc = models.QueryRequestBatch(searches=[document_only_query_request])
    assert inspector.inspect(query_batch_request_doc)
    paths = inspector_embed.inspect(query_batch_request_doc)
    assert len(paths) == 1 and paths[0].as_str_list() == ["searches.query"]

    query_batch_request_prefetch_doc = models.QueryRequestBatch(
        searches=[document_only_prefetch_request]
    )
    assert inspector.inspect(query_batch_request_prefetch_doc)
    paths = inspector_embed.inspect(query_batch_request_prefetch_doc)
    assert len(paths) == 1 and paths[0].as_str_list() == ["searches.prefetch.query"]
    assert inspector.inspect([query_batch_request_vector, query_batch_request_doc])
    paths = inspector_embed.inspect([query_batch_request_vector, query_batch_request_doc])
    assert len(paths) == 1 and paths[0].as_str_list() == ["searches.query"]

    query_batch_request_deep_nested_prefetch_doc = models.QueryRequestBatch(
        searches=[deep_nested_prefetch_doc_request]
    )
    assert inspector.inspect(query_batch_request_deep_nested_prefetch_doc)
    paths = inspector_embed.inspect(query_batch_request_deep_nested_prefetch_doc)
    assert len(paths) == 1 and paths[0].as_str_list() == [
        "searches.prefetch.prefetch.prefetch.prefetch.query"
    ]
    # endregion


def test_inspect_upsert_points():
    inspector = Inspector()
    inspector_embed = InspectorEmbed()

    # region negative cases
    vector_batch = models.Batch(ids=[1, 2], vectors=[[1.0, 2.0], [3.0, 4.0]])
    assert not inspector.inspect(vector_batch)
    assert inspector_embed.inspect(vector_batch) == []

    vector_points = [
        models.PointStruct(id=1, vector=[1.0, 2.0]),
        models.PointStruct(id=2, vector=[3.0, 3.0]),
    ]
    assert not inspector.inspect(vector_points)
    assert inspector_embed.inspect(vector_points) == []

    grpc_points = [
        grpc.PointStruct(
            id=grpc.PointId(num=3), vectors=grpc.Vectors(vector=grpc.Vector(data=[1.0, 2.0]))
        ),
        grpc.PointStruct(
            id=grpc.PointId(num=4), vectors=grpc.Vectors(vector=grpc.Vector(data=[3.0, 3.0]))
        ),
    ]
    assert not inspector.inspect(grpc_points)
    assert inspector_embed.inspect(grpc_points) == []

    multiple_keys_batch = models.Batch(
        ids=[1, 2], vectors={"dense": [[1.0, 2.0]], "dense-two": [[3.0, 4.0]]}
    )
    assert not inspector.inspect(multiple_keys_batch)
    assert inspector_embed.inspect(multiple_keys_batch) == []

    dict_vector_points = [
        models.PointStruct(id=1, vector={"dense": [1.0, 2.0]}),
        models.PointStruct(id=2, vector={"dense": [2.0, 3.0]}),
    ]
    assert not inspector.inspect(dict_vector_points)
    assert inspector_embed.inspect(dict_vector_points) == []

    multiple_keys_points = [
        models.PointStruct(id=1, vector={"dense": [1.0, 2.0], "dense-two": [3.0, 4.0]}),
        models.PointStruct(id=2, vector={"dense": [2.0, 3.0]}),
    ]
    assert not inspector.inspect(multiple_keys_points)
    assert inspector_embed.inspect(multiple_keys_points) == []
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
    paths = inspector_embed.inspect(document_batch)
    assert len(paths) == 1 and paths[0].as_str_list() == ["vectors"]

    document_points = [
        models.PointStruct(id=1, vector=doc_1),
        models.PointStruct(id=2, vector=doc_2),
    ]
    assert inspector.inspect(document_points)
    paths = inspector_embed.inspect(document_points)
    assert len(paths) == 1 and paths[0].as_str_list() == ["vector"]

    mixed_points_doc_first = [
        models.PointStruct(id=1, vector=doc_1),
        models.PointStruct(id=2, vector=[0.2, 0.3]),
    ]
    assert inspector.inspect(mixed_points_doc_first)
    paths = inspector_embed.inspect(mixed_points_doc_first)
    assert len(paths) == 1 and paths[0].as_str_list() == ["vector"]

    mixed_points_doc_second = [
        models.PointStruct(id=1, vector=[0.2, 0.3]),
        models.PointStruct(id=2, vector=doc_2),
    ]
    assert inspector.inspect(mixed_points_doc_second)
    paths = inspector_embed.inspect(mixed_points_doc_second)
    assert len(paths) == 1 and paths[0].as_str_list() == ["vector"]

    dict_doc_batch = models.Batch(ids=[1], vectors={"dense": [doc_1]})
    assert inspector.inspect(dict_doc_batch)
    paths = inspector_embed.inspect(dict_doc_batch)
    assert len(paths) == 1 and paths[0].as_str_list() == ["vectors"]

    dict_mixed_batch = models.Batch(
        ids=[1, 2], vectors={"dense": [[1.0, 2.0]], "dense-two": [doc_1]}
    )
    assert inspector.inspect(dict_mixed_batch)
    paths = inspector_embed.inspect(dict_mixed_batch)
    assert len(paths) == 1 and paths[0].as_str_list() == ["vectors"]
    # endregion


def test_inspect_update_operations():
    inspector = Inspector()
    inspector_embed = InspectorEmbed()

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
    assert inspector_embed.inspect(non_relevant_ops) == []

    plain_points_batch = models.PointsBatch(
        batch=models.Batch(ids=[1, 2], vectors=[[0.1, 0.2], [0.3, 0.4]])
    )
    assert not inspector.inspect(plain_points_batch)
    assert inspector_embed.inspect(plain_points_batch) == []

    plain_point_list = models.PointsList(
        points=[
            models.PointStruct(id=1, vector=[1.0, 2.0]),
            models.PointStruct(id=2, vector=[1.0, 3.0]),
        ]
    )
    assert not inspector.inspect(plain_point_list)
    assert inspector_embed.inspect(plain_point_list) == []

    plain_point_vectors = models.PointVectors(id=1, vector=[0.2, 0.3])
    assert not inspector.inspect(plain_point_vectors)
    assert inspector_embed.inspect(plain_point_vectors) == []

    plain_batch_upsert_op = models.UpsertOperation(upsert=plain_points_batch)
    assert not inspector.inspect([plain_batch_upsert_op])
    assert inspector_embed.inspect([plain_batch_upsert_op]) == []

    plain_structs_upsert_op = models.UpsertOperation(upsert=plain_point_list)
    assert not inspector.inspect([plain_structs_upsert_op])
    assert inspector_embed.inspect([plain_structs_upsert_op]) == []

    plain_point_vectors_update_op = models.UpdateVectorsOperation(
        update_vectors=models.UpdateVectors(points=[plain_point_vectors])
    )
    assert not inspector.inspect([plain_point_vectors_update_op])
    assert inspector_embed.inspect([plain_point_vectors_update_op]) == []
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
    paths = inspector_embed.inspect(doc_points_batch)
    assert len(paths) == 1 and paths[0].as_str_list() == ["batch.vectors"]

    doc_points_list = models.PointsList(
        points=[
            models.PointStruct(id=1, vector=doc_1),
            models.PointStruct(id=2, vector=doc_2),
        ]
    )
    assert inspector.inspect(doc_points_list)
    paths = inspector_embed.inspect(doc_points_list)
    assert len(paths) == 1 and paths[0].as_str_list() == ["points.vector"]

    mixed_points_list = models.PointsList(
        points=[
            models.PointStruct(id=1, vector=[0.2, 0.3]),
            models.PointStruct(id=2, vector=doc_2),
        ]
    )
    assert inspector.inspect(mixed_points_list)
    paths = inspector_embed.inspect(mixed_points_list)
    assert len(paths) == 1 and paths[0].as_str_list() == ["points.vector"]

    doc_point_vectors = [models.PointVectors(id=2, vector=doc_1)]
    assert inspector.inspect(doc_point_vectors)
    paths = inspector_embed.inspect(doc_point_vectors)
    assert len(paths) == 1 and paths[0].as_str_list() == ["vector"]

    mixed_point_vectors = [
        models.PointVectors(id=2, vector=[0.2, 0.3]),
        models.PointVectors(id=3, vector=doc_2),
    ]
    assert inspector.inspect(mixed_point_vectors)
    paths = inspector_embed.inspect(mixed_point_vectors)
    assert len(paths) == 1 and paths[0].as_str_list() == ["vector"]

    doc_batch_upsert_op = models.UpsertOperation(upsert=doc_points_batch)
    assert inspector.inspect([doc_batch_upsert_op])
    paths = inspector_embed.inspect([doc_batch_upsert_op])
    assert len(paths) == 1 and paths[0].as_str_list() == ["upsert.batch.vectors"]

    doc_points_list_upsert_op = models.UpsertOperation(upsert=doc_points_list)
    assert inspector.inspect([doc_points_list_upsert_op])
    paths = inspector_embed.inspect([doc_points_list_upsert_op])
    assert len(paths) == 1 and paths[0].as_str_list() == ["upsert.points.vector"]

    mixed_points_list_upsert_op = models.UpsertOperation(upsert=mixed_points_list)
    assert inspector.inspect([mixed_points_list_upsert_op])
    paths = inspector_embed.inspect([mixed_points_list_upsert_op])
    assert len(paths) == 1 and paths[0].as_str_list() == ["upsert.points.vector"]

    assert inspector.inspect([plain_batch_upsert_op, doc_points_list_upsert_op])
    paths = inspector_embed.inspect([plain_batch_upsert_op, doc_points_list_upsert_op])
    assert len(paths) == 1 and paths[0].as_str_list() == ["upsert.points.vector"]

    doc_point_vectors_update_op = models.UpdateVectorsOperation(
        update_vectors=models.UpdateVectors(points=doc_point_vectors)
    )
    assert inspector.inspect([doc_point_vectors_update_op])
    paths = inspector_embed.inspect([doc_point_vectors_update_op])
    assert len(paths) == 1 and paths[0].as_str_list() == ["update_vectors.points.vector"]
    assert inspector.inspect([plain_point_vectors_update_op, doc_point_vectors_update_op])
    paths = inspector_embed.inspect([plain_point_vectors_update_op, doc_point_vectors_update_op])
    assert len(paths) == 1 and paths[0].as_str_list() == ["update_vectors.points.vector"]

    mixed_point_vectors_update_op = models.UpdateVectorsOperation(
        update_vectors=models.UpdateVectors(points=mixed_point_vectors)
    )
    assert inspector.inspect([mixed_point_vectors_update_op])
    paths = inspector_embed.inspect([mixed_point_vectors_update_op])
    assert len(paths) == 1 and paths[0].as_str_list() == ["update_vectors.points.vector"]
    # endregion
