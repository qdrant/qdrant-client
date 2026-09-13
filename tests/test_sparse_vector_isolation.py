import pytest

from qdrant_client import QdrantClient, models


@pytest.fixture
def client():
    client = QdrantClient(":memory:")
    client.create_collection(
        "test", vectors_config={}, sparse_vectors_config={"text": models.SparseVectorParams()}
    )
    try:
        yield client
    finally:
        client.close()


def assert_original_vector(client):
    record = client.retrieve("test", [1], with_vectors=True)[0]
    assert record.vector == {"text": models.SparseVector(indices=[0], values=[1.0])}
    results = client.query_points(
        "test", query=models.SparseVector(indices=[0], values=[1.0]), using="text"
    ).points
    assert [(point.id, point.score) for point in results] == [(1, 1.0)]


@pytest.mark.parametrize("operation", ["insert", "upsert", "update_vectors"])
def test_sparse_vector_inputs_are_detached(client, operation):
    if operation != "insert":
        client.upsert(
            "test",
            [
                models.PointStruct(
                    id=1, vector={"text": models.SparseVector(indices=[2], values=[2.0])}
                )
            ],
        )
    vector = {"text": models.SparseVector(indices=[0], values=[1.0])}
    if operation == "update_vectors":
        point = models.PointVectors(id=1, vector=vector)
        client.update_vectors("test", [point])
    else:
        point = models.PointStruct(id=1, vector=vector)
        client.upsert("test", [point])

    point.vector["text"].indices[0] = 1
    point.vector["text"].values[0] = 99.0
    assert_original_vector(client)


@pytest.mark.parametrize("operation", ["retrieve", "scroll", "query"])
def test_sparse_vector_results_are_detached(client, operation):
    client.upsert(
        "test",
        [
            models.PointStruct(
                id=1, vector={"text": models.SparseVector(indices=[0], values=[1.0])}
            )
        ],
    )
    if operation == "retrieve":
        point = client.retrieve("test", [1], with_vectors=True)[0]
    elif operation == "scroll":
        point = client.scroll("test", with_vectors=True)[0][0]
    else:
        point = client.query_points(
            "test",
            query=models.SparseVector(indices=[0], values=[1.0]),
            using="text",
            with_vectors=True,
        ).points[0]

    point.vector["text"].indices[0] = 1
    point.vector["text"].values[0] = 99.0
    assert_original_vector(client)
