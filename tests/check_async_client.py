import asyncio
import random

import grpc.aio._call
import numpy as np

import qdrant_client.http.exceptions
from qdrant_client import models
from qdrant_client.async_qdrant_client import AsyncQdrantClient

COLLECTION_NAME = "async_test_collection"


async def main():
    client = AsyncQdrantClient(prefer_grpc=True)
    collection_params = dict(
        collection_name=COLLECTION_NAME,
        vectors_config=models.VectorParams(size=10, distance=models.Distance.EUCLID),
    )
    try:
        await client.create_collection(**collection_params)
    except (
        qdrant_client.http.exceptions.UnexpectedResponse,
        grpc.aio._call.AioRpcError,
    ):
        print("collection already exists")
        await client.delete_collection(COLLECTION_NAME)
        await client.create_collection(**collection_params)
    await client.recreate_collection(**collection_params)

    print(await client.get_collection(COLLECTION_NAME))
    print(await client.get_collections())

    await client.update_collection(
        COLLECTION_NAME, hnsw_config=models.HnswConfigDiff(m=32, ef_construct=120)
    )

    alias_name = COLLECTION_NAME + "_alias"
    await client.update_collection_aliases(
        change_aliases_operations=[
            models.CreateAliasOperation(
                create_alias=models.CreateAlias(
                    collection_name=COLLECTION_NAME, alias_name=alias_name
                )
            )
        ]
    )
    print(await client.get_aliases())
    print(await client.get_collection_aliases(COLLECTION_NAME))
    await client.update_collection_aliases(
        change_aliases_operations=[
            models.DeleteAliasOperation(delete_alias=models.DeleteAlias(alias_name=alias_name))
        ]
    )
    assert (await client.get_aliases()).aliases == []

    await client.upsert(
        collection_name=COLLECTION_NAME,
        points=[
            models.PointStruct(
                id=i,
                vector=np.random.rand(10).tolist(),
                payload={"random_dig": random.randint(1, 100)},
            )
            for i in range(100)
        ],
    )
    assert (await client.count(COLLECTION_NAME)).count == 100

    assert len((await client.scroll(COLLECTION_NAME, limit=2))[0]) == 2

    assert (
        len(
            await client.search(
                COLLECTION_NAME,
                query_vector=np.random.rand(10).tolist(),  # type: ignore
                limit=10,
            )
        )
        == 10
    )

    assert (
        len(
            await client.search_batch(
                COLLECTION_NAME,
                requests=[
                    models.SearchRequest(vector=np.random.rand(10).tolist(), limit=10)
                    for _ in range(3)
                ],
            )
        )
        == 3
    )

    assert (
        len(
            (
                await client.search_groups(
                    COLLECTION_NAME,
                    query_vector=np.random.rand(10).tolist(),  # type: ignore
                    limit=4,
                    group_by="random_dig",
                )
            ).groups
        )
        == 4
    )

    assert len(await client.recommend(COLLECTION_NAME, positive=[0], limit=5)) == 5
    assert (
        len(
            (
                await client.recommend_groups(
                    COLLECTION_NAME, positive=[1], group_by="random_dig", limit=6
                )
            ).groups
        )
        == 6
    )
    assert (
        len(
            (
                await client.recommend_batch(
                    COLLECTION_NAME,
                    requests=[models.RecommendRequest(positive=[2], limit=7)],
                )
            )[0]
        )
        == 7
    )

    assert len(await client.retrieve(COLLECTION_NAME, ids=[3, 5])) == 2

    await client.create_payload_index(
        COLLECTION_NAME,
        field_name="random_dig",
        field_schema=models.PayloadSchemaType.INTEGER,
    )
    assert "random_dig" in (await client.get_collection(COLLECTION_NAME)).payload_schema

    await client.delete_payload_index(COLLECTION_NAME, field_name="random_dig")
    assert "random_dig" not in (await client.get_collection(COLLECTION_NAME)).payload_schema

    assert not (await client.lock_storage(reason="test")).write
    assert (await client.get_locks()).write
    assert (await client.unlock_storage()).write
    assert not (await client.get_locks()).write

    assert isinstance(await client.create_snapshot(COLLECTION_NAME), models.SnapshotDescription)
    snapshots = await client.list_snapshots(COLLECTION_NAME)
    assert len(snapshots) == 1
    # await client.upsert(COLLECTION_NAME, points=[models.PointStruct(id=101, vector=np.random.rand(10).tolist())])
    # assert (await client.get_collection(COLLECTION_NAME)).vectors_count == 101
    # await client.recover_snapshot(collection_name=COLLECTION_NAME, location=...)
    # assert (await client.get_collection(COLLECTION_NAME)).vectors_count == 100
    await client.delete_snapshot(COLLECTION_NAME, snapshot_name=snapshots[0].name)
    assert len(await client.list_snapshots(COLLECTION_NAME)) == 0

    assert isinstance(await client.create_full_snapshot(), models.SnapshotDescription)
    snapshots = await client.list_full_snapshots()
    assert len(snapshots) == 1

    await client.delete_full_snapshot(snapshot_name=snapshots[0].name)
    assert len(await client.list_full_snapshots()) == 0

    assert isinstance(
        await client.create_shard_snapshot(COLLECTION_NAME, shard_id=0),
        models.SnapshotDescription,
    )
    snapshots = await client.list_shard_snapshots(COLLECTION_NAME, shard_id=0)
    assert len(snapshots) == 1
    # await client.upsert(COLLECTION_NAME, points=[models.PointStruct(id=101, vector=np.random.rand(10).tolist())])
    # assert (await client.get_collection(COLLECTION_NAME)).vectors_count == 101
    # await client.recover_shard_snapshot(collection_name=COLLECTION_NAME, location=..., shard_id=0)
    # assert (await client.get_collection(COLLECTION_NAME)).vectors_count == 100
    await client.delete_shard_snapshot(
        COLLECTION_NAME, snapshot_name=snapshots[0].name, shard_id=0
    )
    assert len(await client.list_shard_snapshots(COLLECTION_NAME, shard_id=0)) == 0

    await client.delete_vectors(COLLECTION_NAME, vectors=[""], points=[0])
    assert (await client.retrieve(COLLECTION_NAME, ids=[0]))[0].vector is None

    await client.update_vectors(
        COLLECTION_NAME,
        points=[models.PointVectors(id=0, vector=[1.0] * 10)],
    )
    assert (await client.retrieve(COLLECTION_NAME, ids=[0], with_vectors=True))[0].vector == [
        1.0
    ] * 10

    await client.delete(COLLECTION_NAME, points_selector=[0])
    assert (await client.count(COLLECTION_NAME)).count == 99

    await client.batch_update_points(
        COLLECTION_NAME,
        update_operations=[
            models.UpsertOperation(
                upsert=models.PointsList(points=[models.PointStruct(id=0, vector=[1.0] * 10)])
            )
        ],
    )
    assert (await client.count(COLLECTION_NAME)).count == 100

    await client.set_payload(COLLECTION_NAME, payload={"added_payload": "zero"}, points=[0])
    assert (await client.retrieve(COLLECTION_NAME, ids=[0], with_payload=["added_payload"]))[
        0
    ].payload == {"added_payload": "zero"}
    await client.overwrite_payload(
        COLLECTION_NAME, payload={"overwritten": True, "rand_digit": 2023}, points=[1]
    )
    assert (await client.retrieve(COLLECTION_NAME, ids=[1]))[0].payload == {
        "overwritten": True,
        "rand_digit": 2023,
    }

    await client.delete_payload(COLLECTION_NAME, keys=["added_payload"], points=[0])
    assert (await client.retrieve(COLLECTION_NAME, ids=[0], with_payload=["added_payload"]))[
        0
    ].payload == {}
    await client.clear_payload(COLLECTION_NAME, points_selector=[1])
    assert (await client.retrieve(COLLECTION_NAME, ids=[1]))[0].payload == {}

    # region teardown
    await client.delete_collection(COLLECTION_NAME)
    collections = await client.get_collections()

    assert all(collection.name != COLLECTION_NAME for collection in collections.collections)
    await client.close()
    # endregion


if __name__ == "__main__":
    asyncio.run(main())
