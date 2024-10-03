from qdrant_client import models, grpc
from qdrant_client.embed.type_inspector import Inspector


def test_inspect_query_types():
    inspector = Inspector()
    assert not inspector.inspect(1)
    assert not inspector.inspect("1")
    assert not inspector.inspect([1.0, 2.0, 3.0])
    assert not inspector.inspect([[1.0, 2.0, 3.0]])
    assert not inspector.inspect(models.SparseVector(indices=[0, 1], values=[2.0, 3.0]))
    assert not inspector.inspect(models.NearestQuery(nearest=[1.0, 2.0]))
    assert not inspector.inspect(
        models.RecommendQuery(
            recommend=models.RecommendInput(positive=[[1.0, 2.0]], negative=[[-1.0, -2.0]])
        )
    )
    assert not inspector.inspect(
        models.DiscoverQuery(
            discover=models.DiscoverInput(
                target=[3.0, 4.0],
                context=models.ContextPair(positive=[1.0, 2.0], negative=[-1.0, -2.0]),
            )
        )
    )
    assert not inspector.inspect(
        models.ContextQuery(
            context=[models.ContextPair(positive=[1.0, 2.0], negative=[-1.0, -2.0])]
        )
    )
    assert not inspector.inspect(None)

    doc = models.Document(text="123", model="Qdrant/bm25")
    assert doc

    assert inspector.inspect(models.NearestQuery(nearest=doc))
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
    assert inspector.inspect(
        models.ContextQuery(context=[models.ContextPair(positive=doc, negative=[-1.0, -2.0])])
    )
    assert inspector.inspect(
        models.ContextQuery(context=[models.ContextPair(positive=[1.0, 2.0], negative=doc)])
    )


def test_inspect_prefetch_types():
    inspector = Inspector()
    none_prefetch = models.Prefetch(query=None, prefetch=None)
    assert not inspector.inspect(none_prefetch)

    vector_prefetch = models.Prefetch(query=[1.0, 2.0])
    assert not inspector.inspect(vector_prefetch)

    doc_prefetch = models.Prefetch(query=models.Document(text="123", model="Qdrant/bm25"))
    assert inspector.inspect(doc_prefetch)

    nested_prefetch = models.Prefetch(
        query=None,
        prefetch=models.Prefetch(query=models.Document(text="123", model="Qdrant/bm25")),
    )
    assert inspector.inspect(nested_prefetch)

    vector_and_doc_prefetch = models.Prefetch(
        query=[1.0, 2.0],
        prefetch=models.Prefetch(query=models.Document(text="123", model="Qdrant/bm25")),
    )
    assert inspector.inspect(vector_and_doc_prefetch)


def test_inspect_query_and_prefetch_types():
    inspector = Inspector()
    none_query = None
    none_prefetch = None
    query = models.Document(text="123", model="Qdrant/bm25")
    prefetch = models.Prefetch(query=models.Document(text="123", model="Qdrant/bm25"))

    assert not inspector.inspect([none_query, none_prefetch])
    assert inspector.inspect([query, none_prefetch])
    assert inspector.inspect([none_query, prefetch])
    assert inspector.inspect([query, prefetch])


def test_inspect_points():
    inspector = Inspector()
    vector_batch = models.Batch(ids=[1, 2], vectors=[[1.0, 2.0], [3.0, 4.0]])
    assert not inspector.inspect(vector_batch)

    document_batch = models.Batch(
        ids=[1, 2],
        vectors=[
            models.Document(text="123", model="Qdrant/bm25"),
            models.Document(text="324", model="Qdrant/bm25"),
        ],
    )
    assert inspector.inspect(document_batch)

    vector_points = [models.PointStruct(id=1, vector=[1.0, 2.0])]
    assert not inspector.inspect(vector_points)

    document_points = [
        models.PointStruct(id=1, vector=models.Document(text="123", model="Qdrant/bm25"))
    ]
    assert inspector.inspect(document_points)

    mixed_points = [
        models.PointStruct(id=1, vector=[1.0, 2.0]),
        models.PointStruct(id=1, vector=models.Document(text="123", model="Qdrant/bm25")),
    ]
    assert inspector.inspect(mixed_points)

    grpc_point = [
        grpc.PointStruct(
            id=grpc.PointId(num=3), vectors=grpc.Vectors(vector=grpc.Vector(data=[1.0, 2.0]))
        )
    ]
    assert not inspector.inspect(grpc_point)

    dict_batch = models.Batch(ids=[1, 2], vectors={"dense": [[1.0, 2.0]]})
    assert not inspector.inspect(dict_batch)

    dict_doc_batch = models.Batch(
        ids=[1, 2], vectors={"dense": [models.Document(text="123", model="Qdrant/bm25")]}
    )
    assert inspector.inspect(dict_doc_batch)

    multiple_keys_batch = models.Batch(
        ids=[1, 2], vectors={"dense": [[1.0, 2.0]], "dense-two": [[3.0, 4.0]]}
    )
    assert not inspector.inspect(multiple_keys_batch)

    multiple_keys_mixed_types_batch = models.Batch(
        ids=[1, 2],
        vectors={
            "dense": [[1.0, 2.0]],
            "dense-two": [models.Document(text="123", model="Qdrant/bm25")],
        },
    )
    assert inspector.inspect(multiple_keys_mixed_types_batch)


def test_inspect_query_requests():
    inspector = Inspector()
    vector_only_query_request = models.QueryRequest(
        query=[0.2, 0.3],
    )

    assert not inspector.inspect([vector_only_query_request])

    vector_only_prefetch_request = models.QueryRequest(prefetch=models.Prefetch(query=[0.2, 0.1]))

    assert not inspector.inspect([vector_only_prefetch_request])

    document_only_query_request = models.QueryRequest(
        query=models.Document(text="123", model="Qdrant/bm25"),
    )

    assert inspector.inspect([document_only_query_request])

    document_only_prefetch_request = models.QueryRequest(
        prefetch=models.Prefetch(query=models.Document(text="123", model="Qdrant/bm25"))
    )

    assert inspector.inspect([document_only_prefetch_request])

    assert inspector.inspect([vector_only_query_request, document_only_query_request])


def test_inspect_update_operations():
    inspector = Inspector()
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
                models.Document(text="123", model="Qdrant/bm25"),
                models.Document(text="321", model="Qdrant/bm25"),
            ],
        ),
    )
    doc_point_struct = models.PointsList(
        points=[
            models.PointStruct(id=1, vector=models.Document(text="123", model="Qdrant/bm25")),
            models.PointStruct(id=2, vector=models.Document(text="321", model="Qdrant/bm25")),
        ]
    )

    plain_batch_upsert_op = models.UpsertOperation(upsert=plain_points_batch)
    plain_structs_upsert_op = models.UpsertOperation(upsert=plain_point_structs)
    doc_batch_upsert_op = models.UpsertOperation(upsert=doc_points_batch)
    doc_structs_upsert_op = models.UpsertOperation(upsert=doc_point_struct)

    assert not inspector.inspect([plain_batch_upsert_op])
    assert not inspector.inspect([plain_structs_upsert_op])
    assert inspector.inspect([doc_batch_upsert_op])
    assert inspector.inspect([doc_structs_upsert_op])

    assert inspector.inspect([plain_batch_upsert_op, doc_structs_upsert_op])

    plain_point_vectors = models.PointVectors(id=1, vector=[0.2, 0.3])
    doc_point_vectors = models.PointVectors(
        id=2, vector=models.Document(text="123", model="Qdrant/bm25")
    )

    plain_point_vectors_update_op = models.UpdateVectorsOperation(
        update_vectors=models.UpdateVectors(points=[plain_point_vectors])
    )
    doc_point_vectors_update_op = models.UpdateVectorsOperation(
        update_vectors=models.UpdateVectors(points=[doc_point_vectors])
    )

    assert not inspector.inspect([plain_point_vectors_update_op])
    assert inspector.inspect([doc_point_vectors_update_op])

    assert inspector.inspect([plain_point_vectors_update_op, doc_point_vectors_update_op])


def test_inspector():
    inspector = Inspector()

    doc = models.Document(text="123", model="Qd")
    assert inspector.inspect(doc)

    s = [
        models.PointVectors(id=3, vector=[0.2, 0.3]),
        models.PointVectors(id=2, vector=models.Document(text="123", model="Qd")),
    ]
    assert inspector.inspect(s)

    c = models.ContextPair(positive=[0.2, 0.3], negative=[0.3, 0.4])
    assert not inspector.inspect(c)

    c2 = models.ContextPair(
        positive=[0.2, 0.3], negative=models.Document(text="sdas", model="qwert")
    )
    assert inspector.inspect(c2)

    cq = models.ContextQuery(context=c)
    assert not inspector.inspect(cq)

    cq2 = models.ContextQuery(context=c2)
    assert inspector.inspect(cq2)

    cl = [c, c2]
    cq3 = models.ContextQuery(context=cl)
    assert inspector.inspect(cq3)

    di = models.DiscoverInput(target=[0.2, 0.3], context=c)
    assert not inspector.inspect(di)

    di2 = models.DiscoverInput(target=models.Document(text="qwert", model="scx"), context=c)
    assert inspector.inspect(di2)

    di3 = models.DiscoverInput(target=models.Document(text="dasd", model="aaa"), context=c2)
    assert inspector.inspect(di3)

    dq = models.DiscoverQuery(discover=di)
    assert not inspector.inspect(dq)

    dq2 = models.DiscoverQuery(discover=di2)
    assert inspector.inspect(dq2)

    dq3 = models.DiscoverQuery(discover=di3)
    assert inspector.inspect(dq3)

    nq = models.NearestQuery(nearest=[[0.2, 0.3]])
    assert not inspector.inspect(nq)

    nq2 = models.NearestQuery(nearest=[0.3, 0.4])
    assert not inspector.inspect(nq2)

    nq3 = models.NearestQuery(nearest=models.Document(text="qwdas", model="xczxc"))
    assert inspector.inspect(nq3)

    ps = models.PointStruct(id=2, vector=[0.2, 0.3])
    assert not inspector.inspect(ps)

    ps2 = models.PointStruct(id=2, vector=models.Document(text="qwert", model="scx"))
    assert inspector.inspect(ps2)

    pv = models.PointVectors(id=2, vector=[0.2, 0.3])
    assert not inspector.inspect(pv)

    pv2 = models.PointVectors(id=2, vector=models.Document(text="qwert", model="scx"))
    assert inspector.inspect(pv2)

    pb = models.PointsBatch(batch=models.Batch(ids=[1, 2, 3], vectors=[[0.2, 0.3]]))
    assert not inspector.inspect(pb)

    pb2 = models.PointsBatch(
        batch=models.Batch(ids=[1, 2, 3], vectors=[models.Document(text="qwert", model="scx")])
    )

    assert inspector.inspect(pb2)

    pb3 = models.PointsBatch(batch=models.Batch(ids=[1, 2, 3], vectors=[[[0.2, 0.3], [0.4, 0.5]]]))
    assert not inspector.inspect(pb3)

    pb4 = models.PointsBatch(
        batch=models.Batch(ids=[1, 2, 3], vectors={"Q": [[[0.3, 0.4], [0.2, 0.3]]]})
    )
    assert not inspector.inspect(pb4)

    pb5 = models.PointsBatch(
        batch=models.Batch(
            ids=[1, 2, 3],
            vectors={
                "Q": [
                    models.Document(text="qwert", model="scx"),
                    models.Document(text="qwert", model="scx"),
                ]
            },
        )
    )
    assert inspector.inspect(pb5)

    pl = models.PointsList(points=[models.PointStruct(id=1, vector=[0.2, 0.3])])
    assert not inspector.inspect(pl)
    pl2 = models.PointsList(
        points=[models.PointStruct(id=2, vector=models.Document(text="asdf", model="wesd"))]
    )
    assert inspector.inspect(pl2)
    pl3 = models.PointsList(
        points=[
            models.PointStruct(id=1, vector=[0.2, 0.3]),
            models.PointStruct(id=2, vector=models.Document(text="asdf", model="wesd")),
        ]
    )
    assert inspector.inspect(pl3)
    pl4 = models.PointsList(
        points=[models.PointStruct(id=1, vector={"W": [0.3, 0.4], "E": [0.2, 0.1]})]
    )
    assert not inspector.inspect(pl4)
    pl5 = models.PointsList(
        points=[models.PointStruct(id=1, vector={"D": models.Document(text="aas", model="dzcz")})]
    )
    assert inspector.inspect(pl5)
    pl6 = models.PointsList(
        points=[
            models.PointStruct(
                id=1,
                vector={
                    "W": [0.3, 0.4],
                    "E": [0.2, 0.1],
                    "D": models.Document(text="aas", model="dzcz"),
                },
            )
        ]
    )
    assert inspector.inspect(pl6)

    ri = models.RecommendInput()
    assert not inspector.inspect(ri)

    ri1 = models.RecommendInput(positive=[[0.2, 0.3]])
    assert not inspector.inspect(ri1)
    ri2 = models.RecommendInput(negative=[[0.2, 0.3]])
    assert not inspector.inspect(ri2)
    ri3 = models.RecommendInput(
        positive=[[0.2, 0.3]], negative=[models.Document(text="wrap", model="wrap2")]
    )
    assert inspector.inspect(ri3)
    ri4 = models.RecommendInput(positive=[[[0.2, 0.3]]])
    assert not inspector.inspect(ri4)
    ri5 = models.RecommendInput(
        positive=[[0.2, 0.3]], negative=[[0.2, 0.3], models.Document(text="wrap", model="wrap2")]
    )
    assert inspector.inspect(ri5)

    rq = models.RecommendQuery(recommend=ri)
    assert not inspector.inspect(rq)

    rq1 = models.RecommendQuery(recommend=ri1)
    assert not inspector.inspect(rq1)

    rq2 = models.RecommendQuery(recommend=ri2)
    assert not inspector.inspect(rq2)

    rq3 = models.RecommendQuery(recommend=ri3)
    assert inspector.inspect(rq3)

    rq4 = models.RecommendQuery(recommend=ri4)
    assert not inspector.inspect(rq4)

    rq5 = models.RecommendQuery(recommend=ri5)
    assert inspector.inspect(rq5)

    uv = models.UpdateVectors(points=[pv])
    assert not inspector.inspect(uv)

    uv1 = models.UpdateVectors(
        points=[models.PointVectors(id=2, vector=models.Document(text="qwert", model="scx"))]
    )
    assert inspector.inspect(pv2)

    uv2 = models.UpdateVectors(
        points=[
            models.PointVectors(id=3, vector={"W": models.Document(text="qwert", model="scx")})
        ]
    )
    assert inspector.inspect(uv2)

    uv3 = models.UpdateVectors(
        points=[models.PointVectors(id=3, vector={"W": [0.3, 0.4], "E": [0.2, 0.1]})]
    )
    assert not inspector.inspect(uv3)

    uvo = models.UpdateVectorsOperation(update_vectors=uv)
    assert not inspector.inspect(uvo)

    uvo1 = models.UpdateVectorsOperation(update_vectors=uv1)
    assert inspector.inspect(uvo1)

    uvo2 = models.UpdateVectorsOperation(update_vectors=uv2)
    assert inspector.inspect(uvo2)

    uvo3 = models.UpdateVectorsOperation(update_vectors=uv3)
    assert not inspector.inspect(uvo3)

    uopb = models.UpsertOperation(upsert=pb)
    assert not inspector.inspect(uopb)

    uopb2 = models.UpsertOperation(upsert=pb2)
    assert inspector.inspect(uopb2)

    uopb3 = models.UpsertOperation(upsert=pb3)
    assert not inspector.inspect(uopb3)

    uopb4 = models.UpsertOperation(upsert=pb4)
    assert not inspector.inspect(uopb4)

    uopb5 = models.UpsertOperation(upsert=pb5)
    assert inspector.inspect(uopb5)

    uopl = models.UpsertOperation(upsert=pl)
    assert not inspector.inspect(uopl)

    uopl2 = models.UpsertOperation(upsert=pl2)
    assert inspector.inspect(uopl2)

    uopl3 = models.UpsertOperation(upsert=pl3)
    assert inspector.inspect(uopl3)

    uopl4 = models.UpsertOperation(upsert=pl4)
    assert not inspector.inspect(uopl4)

    uopl5 = models.UpsertOperation(upsert=pl5)
    assert inspector.inspect(uopl5)

    uopl6 = models.UpsertOperation(upsert=pl6)
    assert inspector.inspect(uopl6)

    uo = models.UpdateOperations(operations=[])
    assert not inspector.inspect(uo)

    do = models.DeleteOperation(delete=models.PointIdsList(points=[1, 2, 3]))
    uo1 = models.UpdateOperations(operations=[do])
    assert not inspector.inspect(uo1)

    uo2 = models.UpdateOperations(operations=[do, uopl5])
    assert inspector.inspect(uo2)

    uo3 = models.UpdateOperations(operations=[uopb2])
    assert inspector.inspect(uo3)

    p = models.Prefetch()
    assert not inspector.inspect(p)

    p1 = models.Prefetch(query=nq)
    assert not inspector.inspect(p1)

    p2 = models.Prefetch(query=nq3)
    assert inspector.inspect(p2)

    p3 = models.Prefetch(query=nq, prefetch=models.Prefetch(query=nq))
    assert not inspector.inspect(p3)

    p4 = models.Prefetch(query=nq, prefetch=models.Prefetch(query=nq3))
    assert inspector.inspect(p4)

    p5 = models.Prefetch(query=nq3, prefetch=models.Prefetch(query=nq))
    assert inspector.inspect(p5)

    p6 = models.Prefetch(prefetch=models.Prefetch(query=nq))
    assert not inspector.inspect(p6)

    p7 = models.Prefetch(prefetch=models.Prefetch(query=nq3))
    assert inspector.inspect(p7)

    p8 = models.Prefetch(
        prefetch=models.Prefetch(prefetch=models.Prefetch(prefetch=models.Prefetch(query=nq)))
    )
    assert not inspector.inspect(p8)

    p9 = models.Prefetch(
        prefetch=models.Prefetch(prefetch=models.Prefetch(prefetch=models.Prefetch(query=nq3)))
    )
    assert inspector.inspect(p9)

    p10 = models.Prefetch(prefetch=[models.Prefetch(query=nq3)])
    assert inspector.inspect(p10)

    p11 = models.Prefetch(prefetch=[models.Prefetch(query=nq2)])
    assert not inspector.inspect(p11)

    p12 = models.Prefetch(
        query=nq2,
        prefetch=[
            models.Prefetch(query=nq2),
            models.Prefetch(prefetch=[models.Prefetch(query=nq3)]),
        ],
    )
    assert inspector.inspect(p12)

    qgr = models.QueryGroupsRequest(
        query=nq,
        group_by="k",
    )
    assert not inspector.inspect(qgr)
    qgr1 = models.QueryGroupsRequest(query=nq3, group_by="k")
    assert inspector.inspect(qgr1)
    qgr2 = models.QueryGroupsRequest(prefetch=p, group_by="k")
    assert not inspector.inspect(qgr2)
    qgr3 = models.QueryGroupsRequest(prefetch=p1, group_by="k")
    assert not inspector.inspect(qgr3)
    qgr4 = models.QueryGroupsRequest(prefetch=p2, group_by="k")
    assert inspector.inspect(qgr4)
    qgr5 = models.QueryGroupsRequest(prefetch=p3, group_by="k")
    assert not inspector.inspect(qgr5)
    qgr6 = models.QueryGroupsRequest(prefetch=p4, group_by="k")
    assert inspector.inspect(qgr6)
    qgr7 = models.QueryGroupsRequest(prefetch=p5, group_by="k")
    assert inspector.inspect(qgr7)
    qgr8 = models.QueryGroupsRequest(prefetch=p6, group_by="k")
    assert not inspector.inspect(qgr8)
    qgr9 = models.QueryGroupsRequest(prefetch=p7, group_by="k")
    assert inspector.inspect(qgr9)
    qgr10 = models.QueryGroupsRequest(prefetch=p8, group_by="k")
    assert not inspector.inspect(qgr10)
    qgr11 = models.QueryGroupsRequest(prefetch=p9, group_by="k")
    assert inspector.inspect(qgr11)
    qgr12 = models.QueryGroupsRequest(prefetch=p10, group_by="k")
    assert inspector.inspect(qgr12)
    qgr13 = models.QueryGroupsRequest(prefetch=p11, group_by="k")
    assert not inspector.inspect(qgr13)

    qr = models.QueryRequest(
        query=nq,
    )
    assert not inspector.inspect(qr)
    qr1 = models.QueryRequest(
        query=nq3,
    )
    assert inspector.inspect(qr1)
    qr2 = models.QueryRequest(
        prefetch=p,
    )
    assert not inspector.inspect(qr2)
    qr3 = models.QueryRequest(
        prefetch=p1,
    )
    assert not inspector.inspect(qr3)
    qr4 = models.QueryRequest(
        prefetch=p2,
    )
    assert inspector.inspect(qr4)
    qr5 = models.QueryRequest(
        prefetch=p3,
    )
    assert not inspector.inspect(qr5)
    qr6 = models.QueryRequest(
        prefetch=p4,
    )
    assert inspector.inspect(qr6)
    qr7 = models.QueryRequest(
        prefetch=p5,
    )
    assert inspector.inspect(qr7)
    qr8 = models.QueryRequest(
        prefetch=p6,
    )
    assert not inspector.inspect(qr8)
    qr9 = models.QueryRequest(
        prefetch=p7,
    )
    assert inspector.inspect(qr9)
    qr10 = models.QueryRequest(
        prefetch=p8,
    )
    assert not inspector.inspect(qr10)
    qr11 = models.QueryRequest(
        prefetch=p9,
    )
    assert inspector.inspect(qr11)
    qr12 = models.QueryRequest(
        prefetch=p10,
    )
    assert inspector.inspect(qr12)
    qr13 = models.QueryRequest(
        prefetch=p11,
    )
    assert not inspector.inspect(qr13)

    qrb = models.QueryRequestBatch(searches=[])
    assert not inspector.inspect(qrb)

    qrb1 = models.QueryRequestBatch(searches=[qr1])
    assert inspector.inspect(qrb1)

    qrb2 = models.QueryRequestBatch(searches=[qr2])
    assert not inspector.inspect(qrb2)

    qrb3 = models.QueryRequestBatch(searches=[qr3])
    assert not inspector.inspect(qrb3)

    qrb4 = models.QueryRequestBatch(searches=[qr4])
    assert inspector.inspect(qrb4)

    qrb5 = models.QueryRequestBatch(searches=[qr5])
    assert not inspector.inspect(qrb5)

    qrb6 = models.QueryRequestBatch(searches=[qr6])
    assert inspector.inspect(qrb6)

    qrb7 = models.QueryRequestBatch(searches=[qr7])
    assert inspector.inspect(qrb7)

    qrb8 = models.QueryRequestBatch(searches=[qr8])
    assert not inspector.inspect(qrb8)

    qrb9 = models.QueryRequestBatch(searches=[qr9])
    assert inspector.inspect(qrb9)

    qrb10 = models.QueryRequestBatch(searches=[qr10])
    assert not inspector.inspect(qrb10)  # takes a lot of time

    qrb11 = models.QueryRequestBatch(searches=[qr11])
    assert inspector.inspect(qrb11)

    qrb12 = models.QueryRequestBatch(searches=[qr12])
    assert inspector.inspect(qrb12)

    qrb13 = models.QueryRequestBatch(searches=[qr13])
    assert not inspector.inspect(qrb13)

    qrb14 = models.QueryRequestBatch(searches=[qr, qr2, qr3])
    assert not inspector.inspect(qrb14)

    qrb15 = models.QueryRequestBatch(searches=[qr, qr2, qr3, qr4, qr5])
    assert inspector.inspect(qrb15)
