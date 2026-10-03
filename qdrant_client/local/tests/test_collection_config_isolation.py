import json
from pathlib import Path
from typing import Any

import pytest

from qdrant_client import AsyncQdrantClient, QdrantClient, models


def dense_points(named: bool) -> list[models.PointStruct]:
    return [
        models.PointStruct(
            id=1,
            vector={"dense": [6.0, 0.0], "other": [2.0, 0.0]} if named else [6.0, 0.0],
        ),
        models.PointStruct(
            id=2,
            vector={"dense": [1.0, 2.0], "other": [3.0, 0.0]} if named else [1.0, 2.0],
        ),
    ]


def dense_config(named: bool) -> models.VectorParams | dict[str, models.VectorParams]:
    params = models.VectorParams(size=2, distance=models.Distance.DOT)
    return (
        {"dense": params, "other": models.VectorParams(size=2, distance=models.Distance.DOT)}
        if named
        else params
    )


def vector_params(config: Any, named: bool) -> Any:
    return config["dense"] if named else config


def query_values(result: models.QueryResponse) -> list[tuple[int | str, float]]:
    return [(point.id, point.score) for point in result.points]


def disk_distance(path: Path, name: str, named: bool) -> str:
    vectors = json.loads((path / "meta.json").read_text())["collections"][name]["vectors"]
    return vector_params(vectors, named)["distance"]


@pytest.mark.parametrize("named", [False, True], ids=["unnamed", "named"])
@pytest.mark.parametrize("boundary", ["input", "info"])
@pytest.mark.parametrize("persistent", [False, True], ids=["memory", "disk"])
def test_local_collection_config_reuse(
    tmp_path: Path, named: bool, boundary: str, persistent: bool
) -> None:
    """Adapting a second collection's config must not change the first's metric or scores."""
    client = QdrantClient(path=str(tmp_path)) if persistent else QdrantClient(":memory:")
    config = dense_config(named)
    using = "dense" if named else None
    dot = (models.Distance.DOT, [(1, 6.0), (2, 1.0)])
    euclid = (models.Distance.EUCLID, [(2, 2.0), (1, 5.0)])

    def snapshot(name: str) -> tuple:
        params = client.get_collection(name).config.params.vectors
        return (
            vector_params(params, named).distance,
            query_values(client.query_points(name, query=[1.0, 0.0], using=using, limit=2)),
        )

    try:
        client.create_collection("first", vectors_config=config)
        client.upsert("first", dense_points(named))
        assert snapshot("first") == dot
        if boundary == "info":
            config = client.get_collection("first").config.params.vectors

        vector_params(config, named).distance = models.Distance.EUCLID
        observed = {"after_edit": snapshot("first")}
        if persistent:
            assert disk_distance(tmp_path, "first", named) == models.Distance.DOT

        client.create_collection("second", vectors_config=config)
        client.upsert("second", dense_points(named))
        observed["after_create"] = snapshot("first")
        assert snapshot("second") == euclid
        if named:
            for name in ("first", "second"):
                assert query_values(
                    client.query_points(name, query=[1.0, 0.0], using="other", limit=2)
                ) == [(2, 3.0), (1, 2.0)]

        if persistent:
            saved_distance = disk_distance(tmp_path, "first", named)
            assert disk_distance(tmp_path, "second", named) == models.Distance.EUCLID
            client.close()
            client = QdrantClient(path=str(tmp_path))
            observed["after_reopen"] = snapshot("first")
            assert snapshot("second") == euclid
            # Collect live and reopened values before asserting, so the regression also
            # exercises the save that used to persist the unintended metric change.
            observed["saved_distance"] = saved_distance
        expected = {stage: dot for stage in observed if stage != "saved_distance"}
        if persistent:
            expected["saved_distance"] = models.Distance.DOT
        assert observed == expected
    finally:
        client.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("named", [False, True], ids=["unnamed", "named"])
@pytest.mark.parametrize("boundary", ["input", "info"])
async def test_async_local_collection_config_reuse(
    tmp_path: Path, named: bool, boundary: str
) -> None:
    client = AsyncQdrantClient(path=str(tmp_path))
    config = dense_config(named)
    using = "dense" if named else None
    dot = (models.Distance.DOT, [(1, 6.0), (2, 1.0)])
    euclid = (models.Distance.EUCLID, [(2, 2.0), (1, 5.0)])

    async def snapshot(name: str) -> tuple:
        params = (await client.get_collection(name)).config.params.vectors
        return (
            vector_params(params, named).distance,
            query_values(await client.query_points(name, query=[1.0, 0.0], using=using, limit=2)),
        )

    try:
        await client.create_collection("first", vectors_config=config)
        await client.upsert("first", dense_points(named))
        assert await snapshot("first") == dot
        if boundary == "info":
            config = (await client.get_collection("first")).config.params.vectors
        vector_params(config, named).distance = models.Distance.EUCLID
        observed = {"after_edit": await snapshot("first")}
        assert disk_distance(tmp_path, "first", named) == models.Distance.DOT
        await client.create_collection("second", vectors_config=config)
        await client.upsert("second", dense_points(named))
        observed["after_create"] = await snapshot("first")
        assert await snapshot("second") == euclid
        if named:
            for name in ("first", "second"):
                assert query_values(
                    await client.query_points(name, query=[1.0, 0.0], using="other", limit=2)
                ) == [(2, 3.0), (1, 2.0)]
        observed["saved_distance"] = disk_distance(tmp_path, "first", named)
        assert disk_distance(tmp_path, "second", named) == models.Distance.EUCLID
        await client.close()
        client = AsyncQdrantClient(path=str(tmp_path))
        observed["after_reopen"] = await snapshot("first")
        assert await snapshot("second") == euclid
        assert observed == {
            "after_edit": dot,
            "after_create": dot,
            "saved_distance": models.Distance.DOT,
            "after_reopen": dot,
        }
    finally:
        await client.close()


@pytest.mark.parametrize("named", [False, True], ids=["unnamed", "named"])
@pytest.mark.parametrize("reuse", [False, True], ids=["fresh", "unchanged_reuse"])
def test_local_collection_config_reuse_controls(tmp_path: Path, named: bool, reuse: bool) -> None:
    client = QdrantClient(path=str(tmp_path))
    config = dense_config(named)
    using = "dense" if named else None
    try:
        client.create_collection("first", vectors_config=config)
        client.upsert("first", dense_points(named))
        second_config = config if reuse else dense_config(named)
        if not reuse:
            vector_params(second_config, named).distance = models.Distance.EUCLID
        client.create_collection("second", vectors_config=second_config)
        client.upsert("second", dense_points(named))
        client.close()
        client = QdrantClient(path=str(tmp_path))
        assert query_values(
            client.query_points("first", query=[1.0, 0.0], using=using, limit=2)
        ) == [(1, 6.0), (2, 1.0)]
        assert query_values(
            client.query_points("second", query=[1.0, 0.0], using=using, limit=2)
        ) == ([(1, 6.0), (2, 1.0)] if reuse else [(2, 2.0), (1, 5.0)])
        assert disk_distance(tmp_path, "first", named) == models.Distance.DOT
    finally:
        client.close()


@pytest.mark.parametrize("boundary", ["input", "info"])
def test_local_nested_vector_config_isolation(boundary: str) -> None:
    vectors = {
        "dense": models.VectorParams(
            size=2,
            distance=models.Distance.DOT,
            hnsw_config=models.HnswConfigDiff(m=16),
            quantization_config=models.ScalarQuantization(
                scalar=models.ScalarQuantizationConfig(type=models.ScalarType.INT8, quantile=0.99)
            ),
        ),
        "multi": models.VectorParams(
            size=2,
            distance=models.Distance.DOT,
            multivector_config=models.MultiVectorConfig(
                comparator=models.MultiVectorComparator.MAX_SIM
            ),
            hnsw_config=models.HnswConfigDiff(m=8),
        ),
    }
    sparse = {"sparse": models.SparseVectorParams(index=models.SparseIndexParams(on_disk=False))}
    client = QdrantClient(":memory:")
    try:
        client.create_collection("first", vectors_config=vectors, sparse_vectors_config=sparse)
        if boundary == "info":
            params = client.get_collection("first").config.params
            vectors, sparse = params.vectors, params.sparse_vectors
        vectors["dense"].hnsw_config.m = 32
        vectors["dense"].quantization_config.scalar.quantile = 0.5
        vectors["multi"].hnsw_config.m = 24
        sparse["sparse"].index.on_disk = True
        client.create_collection("second", vectors_config=vectors, sparse_vectors_config=sparse)
        first = client.get_collection("first").config.params
        second = client.get_collection("second").config.params
        assert first.vectors["dense"].hnsw_config.m == 16
        assert first.vectors["dense"].quantization_config.scalar.quantile == 0.99
        assert first.vectors["multi"].hnsw_config.m == 8
        assert first.sparse_vectors["sparse"].index.on_disk is False
        assert second.vectors["dense"].hnsw_config.m == 32
        assert second.vectors["dense"].quantization_config.scalar.quantile == 0.5
        assert second.vectors["multi"].hnsw_config.m == 24
        assert second.sparse_vectors["sparse"].index.on_disk is True
    finally:
        client.close()


def test_local_config_updates_and_vector_schema_persistence(tmp_path: Path) -> None:
    client = QdrantClient(path=str(tmp_path))
    try:
        client.create_collection(
            "first",
            vectors_config=models.VectorParams(size=2, distance=models.Distance.DOT),
            sparse_vectors_config={"sparse": models.SparseVectorParams()},
            metadata={"kept": "value"},
        )
        client.upsert(
            "first",
            [
                models.PointStruct(
                    id=1,
                    vector={
                        "": [2.0, 0.0],
                        "sparse": models.SparseVector(indices=[1], values=[2.0]),
                    },
                )
            ],
        )
        client.update_collection(
            "first",
            sparse_vectors_config={
                "sparse": models.SparseVectorParams(
                    modifier=models.Modifier.IDF, index=models.SparseIndexParams(on_disk=True)
                )
            },
            metadata={"added": "value"},
        )
        dense = models.DenseVectorNameConfig(
            dense=models.DenseVectorConfig(size=2, distance=models.Distance.EUCLID)
        )
        multi = models.DenseVectorNameConfig(
            dense=models.DenseVectorConfig(
                size=2,
                distance=models.Distance.DOT,
                multivector_config=models.MultiVectorConfig(
                    comparator=models.MultiVectorComparator.MAX_SIM
                ),
            )
        )
        sparse = models.SparseVectorNameConfig(sparse=models.SparseVectorConfig())
        for name, config in (("added", dense), ("multi", multi), ("temporary", sparse)):
            client.create_vector_name("first", name, config)
        client.update_vectors(
            "first",
            [models.PointVectors(id=1, vector={"added": [4.0, 0.0], "multi": [[3.0, 0.0]]})],
        )
        client.delete_vector_name("first", "temporary")
        client.create_vector_name("first", "removed", dense)
        client.delete_vector_name("first", "removed")
        before = client.get_collection("first").config
        sparse_scores = query_values(
            client.query_points(
                "first", using="sparse", query=models.SparseVector(indices=[1], values=[1.0])
            )
        )
        assert sparse_scores[0][1] == pytest.approx(2.0 * 0.28768207245178085)
        client.close()
        client = QdrantClient(path=str(tmp_path))
        after = client.get_collection("first").config
        assert after == before
        assert after.metadata == {"kept": "value", "added": "value"}
        assert set(after.params.vectors) == {"", "added", "multi"}
        assert set(after.params.sparse_vectors) == {"sparse"}
        assert after.params.sparse_vectors["sparse"].index.on_disk is True
        assert query_values(client.query_points("first", query=[1.0, 0.0])) == [(1, 2.0)]
        assert query_values(client.query_points("first", using="added", query=[1.0, 0.0])) == [
            (1, 3.0)
        ]
        assert query_values(client.query_points("first", using="multi", query=[[1.0, 0.0]])) == [
            (1, 3.0)
        ]
        assert (
            query_values(
                client.query_points(
                    "first", using="sparse", query=models.SparseVector(indices=[1], values=[1.0])
                )
            )
            == sparse_scores
        )
    finally:
        client.close()
