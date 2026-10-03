"""Collection configs must share nothing with the caller's objects.

Local mode used to keep the passed vector params by reference and return them from
`get_collection` as is, so reusing a config as a template for another collection, or editing
a returned one, changed the metric and the rankings of an existing collection.
"""

import pytest

from qdrant_client import QdrantClient, models


def query_scores(client: QdrantClient, collection_name: str) -> list[tuple[int, float]]:
    points = client.query_points(collection_name, query=[1.0, 0.0], using="dense").points
    return [(point.id, point.score) for point in points]


@pytest.mark.parametrize("source", ["input", "info"])
def test_reused_vectors_config(source: str) -> None:
    client = QdrantClient(":memory:")
    vectors_config = {
        "dense": models.VectorParams(
            size=2, distance=models.Distance.DOT, hnsw_config=models.HnswConfigDiff(m=16)
        )
    }
    sparse_vectors_config = {
        "sparse": models.SparseVectorParams(index=models.SparseIndexParams(on_disk=False))
    }
    client.create_collection(
        "first", vectors_config=vectors_config, sparse_vectors_config=sparse_vectors_config
    )
    points = [
        models.PointStruct(id=1, vector={"dense": [6.0, 0.0]}),
        models.PointStruct(id=2, vector={"dense": [1.0, 2.0]}),
    ]
    client.upsert("first", points)

    if source == "info":
        params = client.get_collection("first").config.params
        vectors_config, sparse_vectors_config = params.vectors, params.sparse_vectors

    vectors_config["dense"].distance = models.Distance.EUCLID
    vectors_config["dense"].hnsw_config.m = 32
    sparse_vectors_config["sparse"].index.on_disk = True
    client.create_collection(
        "second", vectors_config=vectors_config, sparse_vectors_config=sparse_vectors_config
    )
    client.upsert("second", points)

    first = client.get_collection("first").config.params
    assert first.vectors["dense"].distance == models.Distance.DOT
    assert first.vectors["dense"].hnsw_config.m == 16
    assert first.sparse_vectors["sparse"].index.on_disk is False
    assert query_scores(client, "first") == [(1, 6.0), (2, 1.0)]

    second = client.get_collection("second").config.params
    assert second.vectors["dense"].distance == models.Distance.EUCLID
    assert second.vectors["dense"].hnsw_config.m == 32
    assert second.sparse_vectors["sparse"].index.on_disk is True
    assert query_scores(client, "second") == [(2, 2.0), (1, 5.0)]


def test_updated_config() -> None:
    client = QdrantClient(":memory:")
    client.create_collection(
        "test",
        vectors_config={},
        sparse_vectors_config={"sparse": models.SparseVectorParams()},
        metadata={"kept": "value"},
    )
    sparse_params = models.SparseVectorParams(modifier=models.Modifier.IDF)
    metadata = {"added": {"nested": 1}}
    client.update_collection(
        "test", sparse_vectors_config={"sparse": sparse_params}, metadata=metadata
    )

    sparse_params.modifier = None
    metadata["added"]["nested"] = 2
    client.get_collection("test").config.metadata["added"]["nested"] = 3

    config = client.get_collection("test").config
    assert config.params.sparse_vectors["sparse"].modifier == models.Modifier.IDF
    assert config.metadata == {"kept": "value", "added": {"nested": 1}}
