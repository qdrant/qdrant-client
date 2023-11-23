import random
import tempfile
import pytest

import numpy as np
from typing import Optional

import qdrant_client
import qdrant_client.http.models as rest
from qdrant_client._pydantic_compat import construct

default_collection_name = "example"


def ingest_dense_vector_data(
    vector_size: int = 1500,
    path: Optional[str] = None,
    collection_name: str = default_collection_name,
):  # vector_size < 433: works, vector_size >= 433: crashes
    lines = [x for x in range(10)]

    embeddings = np.random.randn(len(lines), vector_size).tolist()
    client = qdrant_client.QdrantClient(path=path)

    client.recreate_collection(
        collection_name,
        vectors_config=rest.VectorParams(
            size=vector_size,
            distance=rest.Distance.COSINE,
        ),
    )

    client.upsert(
        collection_name=collection_name,
        points=construct(
            rest.Batch,
            ids=random.sample(range(100), len(lines)),
            vectors=embeddings,
        ),
    )


def ingest_sparse_vector_data(
        max_vector_size: int = 100,
        path: Optional[str] = None,
        collection_name: str = default_collection_name,
):  # vector_size < 433: works, vector_size >= 433: crashes
    lines = [x for x in range(10)]

    embeddings = np.random.randn(len(lines), vector_size).tolist()
    client = qdrant_client.QdrantClient(path=path)

    client.recreate_collection(
        collection_name,
        vectors_config=rest.VectorParams(
            size=vector_size,
            distance=rest.Distance.COSINE,
        ),
    )

    client.upsert(
        collection_name=collection_name,
        points=construct(
            rest.Batch,
            ids=random.sample(range(100), len(lines)),
            vectors=embeddings,
        ),
    )


def test_prevent_parallel_access():
    with tempfile.TemporaryDirectory() as tmpdir:
        _client = qdrant_client.QdrantClient(path=tmpdir)

        with pytest.raises(Exception) as e:
            _client2 = qdrant_client.QdrantClient(path=tmpdir)

        assert "already accessed by another instance" in str(e)


def test_local_persistence():
    with tempfile.TemporaryDirectory() as tmpdir:
        ingest_dense_vector_data(path=tmpdir)
        client = qdrant_client.QdrantClient(path=tmpdir)
        assert client.count(default_collection_name).count == 10
        del client

        ingest_dense_vector_data(path=tmpdir)
        client = qdrant_client.QdrantClient(path=tmpdir)
        assert client.count(default_collection_name).count == 10
        del client

        ingest_dense_vector_data(path=tmpdir)
        ingest_dense_vector_data(path=tmpdir, collection_name="example_2")
        client = qdrant_client.QdrantClient(path=tmpdir)
        assert client.count(default_collection_name).count == 10
        assert client.count("example_2").count == 10
