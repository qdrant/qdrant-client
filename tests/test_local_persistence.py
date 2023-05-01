import random
import tempfile

import numpy as np

import qdrant_client
import qdrant_client.http.models as rest

default_collection_name = "example"


def ingest_data(
    vector_size=433, path=None
):  # vector_size < 433: works, vector_size >= 433: crashes
    lines = [x for x in range(10)]

    embeddings = np.random.randn(len(lines), vector_size).tolist()
    client = qdrant_client.QdrantClient(path=path)

    client.recreate_collection(
        default_collection_name,
        vectors_config=rest.VectorParams(
            size=vector_size,
            distance=rest.Distance.COSINE,
        ),
    )

    client.upsert(
        collection_name=default_collection_name,
        points=rest.Batch.construct(
            ids=[random.randint(0, 100) for _ in range(len(lines))],
            vectors=embeddings,
        ),
    )


def test_local_persistence():
    with tempfile.TemporaryDirectory() as tmpdir:
        ingest_data(path=tmpdir)
        ingest_data(path=tmpdir)
