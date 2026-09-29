import pytest

from qdrant_client import AsyncQdrantClient, QdrantClient, models


@pytest.mark.parametrize("persistent", [False, True])
def test_local_delete_collection_reports_missing(tmp_path, persistent):
    client = (
        QdrantClient(path=str(tmp_path)) if persistent else QdrantClient(":memory:")
    )
    try:
        client.create_collection(
            "existing",
            vectors_config=models.VectorParams(size=2, distance=models.Distance.DOT),
        )
        assert client.delete_collection("missing") is False
        assert client.collection_exists("existing")
        assert client.delete_collection("existing") is True
        assert client.delete_collection("existing") is False
    finally:
        client.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("persistent", [False, True])
async def test_async_local_delete_collection_reports_missing(tmp_path, persistent):
    client = (
        AsyncQdrantClient(path=str(tmp_path))
        if persistent
        else AsyncQdrantClient(":memory:")
    )
    try:
        await client.create_collection(
            "existing",
            vectors_config=models.VectorParams(size=2, distance=models.Distance.DOT),
        )
        assert await client.delete_collection("missing") is False
        assert await client.collection_exists("existing")
        assert await client.delete_collection("existing") is True
        assert await client.delete_collection("existing") is False
    finally:
        await client.close()
