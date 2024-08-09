import pytest

from qdrant_client import QdrantClient, models


@pytest.fixture
def qdrant() -> QdrantClient:
    return QdrantClient(":memory:")


def test_dense_in_memory_key_filter_returns_results(qdrant: QdrantClient):
    qdrant.create_collection(
        collection_name="test_collection",
        vectors_config=models.VectorParams(size=4, distance=models.Distance.DOT),
    )

    operation_info = qdrant.upsert(
        collection_name="test_collection",
        wait=True,
        points=[
            models.PointStruct(id=1, vector=[0.05, 0.61, 0.76, 0.74], payload={"city": "Berlin"}),
            models.PointStruct(
                id=2,
                vector=[0.19, 0.81, 0.75, 0.11],
                payload={"city": ["Berlin", "London"]},
            ),
            models.PointStruct(
                id=3,
                vector=[0.36, 0.55, 0.47, 0.94],
                payload={"city": ["Berlin", "Moscow"]},
            ),
            models.PointStruct(
                id=4,
                vector=[0.18, 0.01, 0.85, 0.80],
                payload={"city": ["London", "Moscow"]},
            ),
            models.PointStruct(id=5, vector=[0.24, 0.18, 0.22, 0.44], payload={"count": [0]}),
            models.PointStruct(id=6, vector=[0.35, 0.08, 0.11, 0.44]),
        ],
    )

    assert operation_info.operation_id == 0
    assert operation_info.status == models.UpdateStatus.COMPLETED

    search_result = qdrant.search(
        collection_name="test_collection",
        query_vector=[0.2, 0.1, 0.9, 0.7],
        query_filter=models.Filter(
            must=[models.FieldCondition(key="city", match=models.MatchValue(value="London"))]
        ),
        limit=3,
    )

    assert [r.id for r in search_result] == [4, 2]


def test_sparse_in_memory_key_filter_returns_results(qdrant: QdrantClient):
    qdrant.create_collection(
        collection_name="test_collection",
        vectors_config={},
        sparse_vectors_config={"text": models.SparseVectorParams()},
    )

    operation_info = qdrant.upsert(
        collection_name="test_collection",
        wait=True,
        points=[
            models.PointStruct(
                id=1,
                vector={
                    "text": models.SparseVector(
                        indices=[0, 1, 2, 3], values=[0.05, 0.61, 0.76, 0.74]
                    )
                },
                payload={"city": "Berlin"},
            ),
            models.PointStruct(
                id=2,
                vector={
                    "text": models.SparseVector(
                        indices=[0, 1, 2, 3], values=[0.19, 0.81, 0.75, 0.11]
                    )
                },
                payload={"city": ["Berlin", "London"]},
            ),
            models.PointStruct(
                id=3,
                vector={
                    "text": models.SparseVector(
                        indices=[0, 1, 2, 3], values=[0.36, 0.55, 0.47, 0.94]
                    )
                },
                payload={"city": ["Berlin", "Moscow"]},
            ),
            models.PointStruct(
                id=4,
                vector={
                    "text": models.SparseVector(
                        indices=[0, 1, 2, 3], values=[0.18, 0.01, 0.85, 0.80]
                    )
                },
                payload={"city": ["London", "Moscow"]},
            ),
        ],
    )

    assert operation_info.operation_id == 0
    assert operation_info.status == models.UpdateStatus.COMPLETED

    search_result = qdrant.search(
        collection_name="test_collection",
        query_vector=models.NamedSparseVector(
            name="text",
            vector=models.SparseVector(indices=[0, 1, 2, 3], values=[0.2, 0.1, 0.9, 0.7]),
        ),
        query_filter=models.Filter(
            must=[models.FieldCondition(key="city", match=models.MatchValue(value="London"))]
        ),
        limit=3,
    )

    assert [r.id for r in search_result] == [4, 2]
