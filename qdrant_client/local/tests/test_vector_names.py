from pathlib import Path

import pytest

from qdrant_client import QdrantClient, models


@pytest.mark.parametrize("persistent", [False, True])
def test_delete_sparse_vector_name_with_unnamed_dense_vector(
    persistent: bool, tmp_path: Path
) -> None:
    path = str(tmp_path / "collection") if persistent else None
    client = QdrantClient(path=path) if persistent else QdrantClient(":memory:")
    try:
        dense_config = models.VectorParams(size=2, distance=models.Distance.DOT)
        client.create_collection(
            "test",
            vectors_config=dense_config,
            sparse_vectors_config={"text": models.SparseVectorParams()},
        )
        client.upsert(
            "test",
            [
                models.PointStruct(
                    id=1,
                    vector={
                        "": [1.0, 0.0],
                        "text": models.SparseVector(indices=[1], values=[1.0]),
                    },
                    payload={"title": "example"},
                )
            ],
        )

        client.delete_vector_name("test", "text")

        if persistent:
            client.close()
            client = QdrantClient(path=path)

        params = client.get_collection("test").config.params
        assert params.vectors == dense_config
        assert "text" not in (params.sparse_vectors or {})
        point = client.retrieve("test", [1], with_vectors=True)[0]
        assert point.vector == [1.0, 0.0]
        assert point.payload == {"title": "example"}
        assert client.query_points("test", query=[1.0, 0.0]).points[0].id == 1
    finally:
        client.close()


def test_delete_unnamed_dense_vector_is_still_rejected() -> None:
    client = QdrantClient(":memory:")
    try:
        dense_config = models.VectorParams(size=2, distance=models.Distance.DOT)
        client.create_collection(
            "test",
            vectors_config=dense_config,
            sparse_vectors_config={"text": models.SparseVectorParams()},
        )

        with pytest.raises(ValueError, match="Cannot delete the unnamed vector"):
            client.delete_vector_name("test", "")

        params = client.get_collection("test").config.params
        assert params.vectors == dense_config
        assert "text" in params.sparse_vectors
    finally:
        client.close()
