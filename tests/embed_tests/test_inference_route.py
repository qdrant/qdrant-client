from qdrant_client import models, QdrantClient


def test_inference_flag():
    def arg_checker(*args, **kwargs):
        return args, kwargs

    remote_client = QdrantClient(cloud_inference=True)

    remote_client._client.upsert = arg_checker
    remote_client._client.query_points = arg_checker
    remote_client._client.update_vectors = arg_checker
    remote_client._client.query_points_groups = arg_checker
    remote_client._client.query_batch_points = arg_checker

    cn = "qwerty"
    doc = models.Document(text="my text", model="sentence-transformers/all-MiniLM-L6-v2")
    vec = [0.2, 0.3]

    _, kw = remote_client.upsert(collection_name=cn, points=[models.PointStruct(id=1, vector=doc)])
    assert kw["_cloud_inference"]

    _, kw = remote_client.upsert(collection_name=cn, points=[models.PointStruct(id=1, vector=vec)])
    assert not kw["_cloud_inference"]

    _, kw = remote_client.query_points(collection_name=cn, query=models.NearestQuery(nearest=doc))
    assert kw["_cloud_inference"]

    _, kw = remote_client.query_points(collection_name=cn, query=models.NearestQuery(nearest=vec))
    assert not kw["_cloud_inference"]

    _, kw = remote_client.query_points_groups(
        collection_name=cn, group_by="text", query=models.NearestQuery(nearest=doc)
    )
    assert kw["_cloud_inference"]

    _, kw = remote_client.query_points_groups(
        collection_name=cn, group_by="text", query=models.NearestQuery(nearest=vec)
    )
    assert not kw["_cloud_inference"]

    _, kw = remote_client.query_batch_points(
        collection_name=cn, requests=[models.QueryRequest(query=models.NearestQuery(nearest=doc))]
    )
    assert kw["_cloud_inference"]

    _, kw = remote_client.query_batch_points(
        collection_name=cn, requests=[models.QueryRequest(query=models.NearestQuery(nearest=vec))]
    )
    assert not kw["_cloud_inference"]

    _, kw = remote_client.update_vectors(
        collection_name=cn, points=[models.PointVectors(id=1, vector=doc)]
    )
    assert kw["_cloud_inference"]

    _, kw = remote_client.update_vectors(
        collection_name=cn, points=[models.PointVectors(id=1, vector=vec)]
    )
    assert not kw["_cloud_inference"]

    remote_client.cloud_inference = False

    _, kw = remote_client.upsert(
        collection_name="qwerty", points=[models.PointStruct(id=1, vector=doc)]
    )
    assert not kw["_cloud_inference"]

    _, kw = remote_client.upsert(
        collection_name="qwerty", points=[models.PointStruct(id=1, vector=vec)]
    )
    assert not kw["_cloud_inference"]

    _, kw = remote_client.query_points(collection_name=cn, query=models.NearestQuery(nearest=doc))
    assert not kw["_cloud_inference"]

    _, kw = remote_client.query_points(collection_name=cn, query=models.NearestQuery(nearest=vec))
    assert not kw["_cloud_inference"]

    _, kw = remote_client.query_points_groups(
        collection_name=cn, group_by="text", query=models.NearestQuery(nearest=doc)
    )
    assert not kw["_cloud_inference"]

    _, kw = remote_client.query_points_groups(
        collection_name=cn, group_by="text", query=models.NearestQuery(nearest=vec)
    )
    assert not kw["_cloud_inference"]

    _, kw = remote_client.query_batch_points(
        collection_name=cn, requests=[models.QueryRequest(query=models.NearestQuery(nearest=doc))]
    )
    assert not kw["_cloud_inference"]

    _, kw = remote_client.query_batch_points(
        collection_name=cn, requests=[models.QueryRequest(query=models.NearestQuery(nearest=vec))]
    )
    assert not kw["_cloud_inference"]

    _, kw = remote_client.update_vectors(
        collection_name=cn, points=[models.PointVectors(id=1, vector=doc)]
    )
    assert not kw["_cloud_inference"]

    _, kw = remote_client.update_vectors(
        collection_name=cn, points=[models.PointVectors(id=1, vector=vec)]
    )
    assert not kw["_cloud_inference"]
