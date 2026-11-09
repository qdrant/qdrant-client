from collections.abc import Iterator
from pathlib import Path

import pytest

from qdrant_client import AsyncQdrantClient, QdrantClient, models

# Explicit lookup aliases retain their spelling when checking query-point exclusion.
QUERY_CASES = [
    ("items", "items", [2]),
    ("active", "items", [2]),
    ("items", "active", [1, 2]),
    ("active", "active", [1, 2]),
    ("active", "second", [1, 2]),
    ("different", "items", [1, 2]),
    ("active", None, [2]),
]


@pytest.fixture(params=["memory", "persistent", "reopened"])
def client(request: pytest.FixtureRequest, tmp_path: Path) -> Iterator[QdrantClient]:
    client = (
        QdrantClient(":memory:") if request.param == "memory" else QdrantClient(path=str(tmp_path))
    )
    for name in ("items", "different"):
        client.create_collection(
            name, vectors_config=models.VectorParams(size=2, distance=models.Distance.DOT)
        )
        client.upsert(
            name,
            [
                models.PointStruct(id=1, vector=[1.0, 0.0], payload={"group": 1}),
                models.PointStruct(id=2, vector=[0.8, 0.2], payload={"group": 2}),
            ],
        )
    client.update_collection_aliases(
        [
            models.CreateAliasOperation(
                create_alias=models.CreateAlias(alias_name=alias, collection_name="items")
            )
            for alias in ("active", "second")
        ]
    )
    if request.param == "reopened":
        client.close()
        client = QdrantClient(path=str(tmp_path))
    try:
        yield client
    finally:
        client.close()


@pytest.mark.parametrize("target,source,expected", QUERY_CASES)
def test_query_id_exclusion_with_aliases(
    client: QdrantClient, target: str, source: str | None, expected: list[int]
) -> None:
    lookup = models.LookupLocation(collection=source) if source is not None else None
    result = client.query_points(target, query=1, lookup_from=lookup)
    assert [point.id for point in result.points] == expected
    batch = client.query_batch_points(target, [models.QueryRequest(query=1, lookup_from=lookup)])
    assert [point.id for point in batch[0].points] == expected
    groups = client.query_points_groups(target, query=1, lookup_from=lookup, group_by="group")
    assert sorted(hit.id for group in groups.groups for hit in group.hits) == expected


@pytest.mark.asyncio
@pytest.mark.parametrize("target,source,expected", QUERY_CASES)
async def test_async_query_id_exclusion_with_aliases(
    tmp_path: Path, target: str, source: str | None, expected: list[int]
) -> None:
    client = AsyncQdrantClient(path=str(tmp_path))
    try:
        for name in ("items", "different"):
            await client.create_collection(
                name, vectors_config=models.VectorParams(size=2, distance=models.Distance.DOT)
            )
            await client.upsert(
                name,
                [
                    models.PointStruct(id=1, vector=[1.0, 0.0], payload={"group": 1}),
                    models.PointStruct(id=2, vector=[0.8, 0.2], payload={"group": 2}),
                ],
            )
        await client.update_collection_aliases(
            [
                models.CreateAliasOperation(
                    create_alias=models.CreateAlias(alias_name=alias, collection_name="items")
                )
                for alias in ("active", "second")
            ]
        )
        lookup = models.LookupLocation(collection=source) if source is not None else None
        result = await client.query_points(target, query=1, lookup_from=lookup)
        assert [point.id for point in result.points] == expected
        batch = await client.query_batch_points(
            target, [models.QueryRequest(query=1, lookup_from=lookup)]
        )
        assert [point.id for point in batch[0].points] == expected
        groups = await client.query_points_groups(
            target, query=1, lookup_from=lookup, group_by="group"
        )
        assert sorted(hit.id for group in groups.groups for hit in group.hits) == expected
    finally:
        await client.close()
