import random
import tempfile

import numpy as np

import qdrant_client
import qdrant_client.http.models as rest

default_collection_name = "example"


def ingest_data(
    vector_size=1500,
    path=None,
    collection_name=default_collection_name,
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
        points=rest.Batch.construct(
            ids=random.sample(range(100), len(lines)),
            vectors=embeddings,
        ),
    )


def test_prevent_parallel_access():
    with tempfile.TemporaryDirectory() as tmpdir:
        client = qdrant_client.QdrantClient(path=tmpdir)

        try:
            client2 = qdrant_client.QdrantClient(path=tmpdir)
            assert False
        except Exception as e:
            error_message = str(e)
            assert "already accessed by another instance" in error_message


def test_local_persistence():
    with tempfile.TemporaryDirectory() as tmpdir:
        ingest_data(path=tmpdir)
        client = qdrant_client.QdrantClient(path=tmpdir)
        assert 10 == client.count(default_collection_name).count
        del client

        ingest_data(path=tmpdir)
        client = qdrant_client.QdrantClient(path=tmpdir)
        assert 10 == client.count(default_collection_name).count
        del client

        ingest_data(path=tmpdir)
        ingest_data(path=tmpdir, collection_name="example_2")
        client = qdrant_client.QdrantClient(path=tmpdir)
        assert 10 == client.count(default_collection_name).count
        assert 10 == client.count("example_2").count
