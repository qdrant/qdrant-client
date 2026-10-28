"""Local collection creation rejects names shared by dense and sparse vectors."""

from pathlib import Path

import pytest

from qdrant_client import AsyncQdrantClient, QdrantClient, models


@pytest.fixture(params=[False, True], ids=["dense", "multivector"])
def vector_params(request: pytest.FixtureRequest) -> models.VectorParams:
    """Both dense vector kinds share the namespace with sparse vectors."""
    return models.VectorParams(
        size=2,
        distance=models.Distance.DOT,
        multivector_config=(
            models.MultiVectorConfig(comparator=models.MultiVectorComparator.MAX_SIM)
            if request.param
            else None
        ),
    )


@pytest.mark.parametrize("persistent", [False, True], ids=["memory", "disk"])
def test_local_collection_rejects_vector_name_collision(
    vector_params: models.VectorParams, persistent: bool, tmp_path: Path
) -> None:
    """Reject the schema before creating a collection or its storage directory."""
    client = QdrantClient(path=str(tmp_path)) if persistent else QdrantClient(":memory:")
    try:
        with pytest.raises(ValueError, match="Dense and sparse vector names must be unique"):
            client.create_collection(
                "example",
                vectors_config={"shared": vector_params},
                sparse_vectors_config={"shared": models.SparseVectorParams()},
            )
        assert not client.collection_exists("example")
        assert not (tmp_path / "collection" / "example").exists()

        assert client.create_collection(
            "example",
            vectors_config={"dense": vector_params},
            sparse_vectors_config={"sparse": models.SparseVectorParams()},
        )
        vector = [[1.0, 2.0]] if vector_params.multivector_config else [1.0, 2.0]
        client.upsert(
            "example",
            [
                models.PointStruct(
                    id=1,
                    vector={
                        "dense": vector,
                        "sparse": models.SparseVector(indices=[1], values=[3.0]),
                    },
                )
            ],
        )
        stored_vectors = client.retrieve("example", [1], with_vectors=True)[0].vector
        assert isinstance(stored_vectors, dict)
        assert set(stored_vectors) == {"dense", "sparse"}
    finally:
        client.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("persistent", [False, True], ids=["memory", "disk"])
async def test_async_local_collection_rejects_vector_name_collision(
    vector_params: models.VectorParams, persistent: bool, tmp_path: Path
) -> None:
    """The async client rejects the same schema without leaving a partial collection."""
    client = AsyncQdrantClient(path=str(tmp_path)) if persistent else AsyncQdrantClient(":memory:")
    try:
        with pytest.raises(ValueError, match="Dense and sparse vector names must be unique"):
            await client.create_collection(
                "example",
                vectors_config={"shared": vector_params},
                sparse_vectors_config={"shared": models.SparseVectorParams()},
            )
        assert not await client.collection_exists("example")
        assert not (tmp_path / "collection" / "example").exists()

        assert await client.create_collection(
            "example",
            vectors_config={"dense": vector_params},
            sparse_vectors_config={"sparse": models.SparseVectorParams()},
        )
        vector = [[1.0, 2.0]] if vector_params.multivector_config else [1.0, 2.0]
        await client.upsert(
            "example",
            [
                models.PointStruct(
                    id=1,
                    vector={
                        "dense": vector,
                        "sparse": models.SparseVector(indices=[1], values=[3.0]),
                    },
                )
            ],
        )
        stored_vectors = (await client.retrieve("example", [1], with_vectors=True))[0].vector
        assert isinstance(stored_vectors, dict)
        assert set(stored_vectors) == {"dense", "sparse"}
    finally:
        await client.close()


def test_unnamed_dense_vector_can_coexist_with_named_sparse_vector() -> None:
    """A single unnamed dense vector has no collision with a named sparse vector."""
    client = QdrantClient(":memory:")
    try:
        assert client.create_collection(
            "example",
            vectors_config=models.VectorParams(size=2, distance=models.Distance.DOT),
            sparse_vectors_config={"sparse": models.SparseVectorParams()},
        )
    finally:
        client.close()
