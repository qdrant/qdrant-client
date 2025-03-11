import asyncio
import random
import time
import uuid

import grpc.aio._call
import numpy as np
import pytest

import qdrant_client.http.exceptions
from qdrant_client import models
from qdrant_client.async_qdrant_client import AsyncQdrantClient
from tests.utils import read_version

NUM_VECTORS = 100
NUM_QUERIES = 100
DIM = 32
COLLECTION_NAME = "async_test_collection"


@pytest.mark.asyncio
@pytest.mark.parametrize("prefer_grpc", [True, False])
async def test_async_qdrant_client(prefer_grpc: bool):
    collection_name = f"{COLLECTION_NAME}_{uuid.uuid4().hex}"
    major, minor, patch, dev = read_version()

    client = AsyncQdrantClient(prefer_grpc=prefer_grpc, timeout=15)
    collection_params = dict(
        collection_name=collection_name,
        vectors_config=models.VectorParams(size=10, distance=models.Distance.EUCLID),
    )
    try:
        await client.create_collection(**collection_params)
    except (
        qdrant_client.http.exceptions.UnexpectedResponse,
        grpc.aio._call.AioRpcError,
    ):
        await client.delete_collection(collection_name)
        await client.create_collection(**collection_params)

    await client.get_collection(collection_name)
    await client.get_collections()
    if dev or None in (major, minor, patch) or (major, minor, patch) >= (1, 8, 0):
        await client.collection_exists(collection_name)

    await client.update_collection(
        collection_name, hnsw_config=models.HnswConfigDiff(m=32, ef_construct=120)
    )

    alias_name = collection_name + "_alias"
    await client.update_collection_aliases(
        change_aliases_operations=[
            models.CreateAliasOperation(
                create_alias=models.CreateAlias(
                    collection_name=collection_name, alias_name=alias_name
                )
            )
        ]
    )
    await client.get_aliases()
    await client.get_collection_aliases(collection_name)
    await client.update_collection_aliases(
        change_aliases_operations=[
            models.DeleteAliasOperation(delete_alias=models.DeleteAlias(alias_name=alias_name))
        ]
    )
    assert (await client.get_aliases()).aliases == []

    await client.upsert(
        collection_name=collection_name,
        points=[
            models.PointStruct(
                id=i,
                vector=np.random.rand(10).tolist(),
                payload={"random_dig": random.randint(1, 100)},
            )
            for i in range(100)
        ],
    )
    assert (await client.count(collection_name)).count == 100

    assert len((await client.scroll(collection_name, limit=2))[0]) == 2

    assert (
        len(
            await client.search(
                collection_name,
                query_vector=np.random.rand(10).tolist(),  # type: ignore
                limit=10,
            )
        )
        == 10
    )

    assert (
        len(
            await client.search_batch(
                collection_name,
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
                    collection_name,
                    query_vector=np.random.rand(10).tolist(),  # type: ignore
                    limit=4,
                    group_by="random_dig",
                )
            ).groups
        )
        == 4
    )

    assert len(await client.recommend(collection_name, positive=[0], limit=5)) == 5
    assert (
        len(
            (
                await client.recommend_groups(
                    collection_name, positive=[1], group_by="random_dig", limit=6
                )
            ).groups
        )
        == 6
    )
    assert (
        len(
            (
                await client.recommend_batch(
                    collection_name,
                    requests=[models.RecommendRequest(positive=[2], limit=7)],
                )
            )[0]
        )
        == 7
    )

    if dev or None in (major, minor, patch) or (major, minor, patch) >= (1, 10, 0):
        assert (
            len(
                (
                    await client.query_points(collection_name, query=np.random.rand(10).tolist())
                ).points
            )
            == 10
        )
        query_responses = await client.query_batch_points(
            collection_name, requests=[models.QueryRequest(query=np.random.rand(10).tolist())]
        )
        assert len(query_responses) == 1 and len(query_responses[0].points) == 10

    assert len(await client.retrieve(collection_name, ids=[3, 5])) == 2

    await client.create_payload_index(
        collection_name,
        field_name="random_dig",
        field_schema=models.PayloadSchemaType.INTEGER,
    )
    assert "random_dig" in (await client.get_collection(collection_name)).payload_schema

    await client.delete_payload_index(collection_name, field_name="random_dig")
    assert "random_dig" not in (await client.get_collection(collection_name)).payload_schema

    assert not (await client.lock_storage(reason="test")).write
    assert (await client.get_locks()).write
    assert (await client.unlock_storage()).write
    assert not (await client.get_locks()).write

    assert isinstance(await client.create_snapshot(collection_name), models.SnapshotDescription)
    snapshots = await client.list_snapshots(collection_name)
    assert len(snapshots) == 1

    # recover snapshot location is unknown
    # await client.upsert(collection_name, points=[models.PointStruct(id=101, vector=np.random.rand(10).tolist())])
    # assert (await client.get_collection(collection_name)).vectors_count == 101
    # await client.recover_snapshot(collection_name=collection_name, location=...)
    # assert (await client.get_collection(collection_name)).vectors_count == 100

    await client.delete_snapshot(collection_name, snapshot_name=snapshots[0].name, wait=True)

    assert len(await client.list_snapshots(collection_name)) == 0

    assert isinstance(await client.create_full_snapshot(), models.SnapshotDescription)
    snapshots = await client.list_full_snapshots()
    assert len(snapshots) == 1

    await client.delete_full_snapshot(snapshot_name=snapshots[0].name, wait=True)

    assert len(await client.list_full_snapshots()) == 0

    assert isinstance(
        await client.create_shard_snapshot(collection_name, shard_id=0),
        models.SnapshotDescription,
    )
    snapshots = await client.list_shard_snapshots(collection_name, shard_id=0)
    assert len(snapshots) == 1

    # recover snapshot location is unknown
    # await client.upsert(collection_name, points=[models.PointStruct(id=101, vector=np.random.rand(10).tolist())])
    # assert (await client.get_collection(collection_name)).vectors_count == 101
    # await client.recover_shard_snapshot(collection_name=collection_name, location=..., shard_id=0)
    # assert (await client.get_collection(collection_name)).vectors_count == 100

    await client.delete_shard_snapshot(
        collection_name, snapshot_name=snapshots[0].name, shard_id=0
    )
    time.sleep(
        0.5
    )  # wait param is not propagated https://github.com/qdrant/qdrant-client/issues/254
    assert len(await client.list_shard_snapshots(collection_name, shard_id=0)) == 0

    await client.delete_vectors(collection_name, vectors=[""], points=[0])
    assert (await client.retrieve(collection_name, ids=[0]))[0].vector is None

    await client.update_vectors(
        collection_name,
        points=[models.PointVectors(id=0, vector=[1.0] * 10)],
    )
    assert (await client.retrieve(collection_name, ids=[0], with_vectors=True))[0].vector == [
        1.0
    ] * 10

    await client.delete(collection_name, points_selector=[0])
    assert (await client.count(collection_name)).count == 99

    await client.batch_update_points(
        collection_name,
        update_operations=[
            models.UpsertOperation(
                upsert=models.PointsList(points=[models.PointStruct(id=0, vector=[1.0] * 10)])
            )
        ],
    )
    assert (await client.count(collection_name)).count == 100

    await client.set_payload(collection_name, payload={"added_payload": "zero"}, points=[0])
    assert (await client.retrieve(collection_name, ids=[0], with_payload=["added_payload"]))[
        0
    ].payload == {"added_payload": "zero"}
    await client.overwrite_payload(
        collection_name, payload={"overwritten": True, "rand_digit": 2023}, points=[1]
    )
    assert (await client.retrieve(collection_name, ids=[1]))[0].payload == {
        "overwritten": True,
        "rand_digit": 2023,
    }

    await client.delete_payload(collection_name, keys=["added_payload"], points=[0])
    assert (await client.retrieve(collection_name, ids=[0], with_payload=["added_payload"]))[
        0
    ].payload == {}
    await client.clear_payload(collection_name, points_selector=[1])
    assert (await client.retrieve(collection_name, ids=[1]))[0].payload == {}

    # region teardown
    await client.delete_collection(collection_name)
    collections = await client.get_collections()

    assert all(collection.name != collection_name for collection in collections.collections)
    await client.close()
    # endregion


@pytest.mark.asyncio
async def test_async_qdrant_client_local():
    collection_name = f"{COLLECTION_NAME}_{uuid.uuid4().hex}"
    major, minor, patch, dev = read_version()
    client = AsyncQdrantClient(":memory:")

    collection_params = dict(
        collection_name=collection_name,
        vectors_config=models.VectorParams(size=10, distance=models.Distance.EUCLID),
    )
    if await client.collection_exists(collection_name):
        await client.delete_collection(collection_name)
    await client.create_collection(**collection_params)

    await client.get_collection(collection_name)
    await client.get_collections()
    if dev or None in (major, minor, patch) or (major, minor, patch) >= (1, 8, 0):
        await client.collection_exists(collection_name)
    await client.update_collection(
        collection_name, hnsw_config=models.HnswConfigDiff(m=32, ef_construct=120)
    )

    alias_name = collection_name + "_alias"
    await client.update_collection_aliases(
        change_aliases_operations=[
            models.CreateAliasOperation(
                create_alias=models.CreateAlias(
                    collection_name=collection_name, alias_name=alias_name
                )
            )
        ]
    )
    await client.get_aliases()
    await client.get_collection_aliases(collection_name)
    await client.update_collection_aliases(
        change_aliases_operations=[
            models.DeleteAliasOperation(delete_alias=models.DeleteAlias(alias_name=alias_name))
        ]
    )
    assert await client.get_aliases()

    await client.upsert(
        collection_name=collection_name,
        points=[
            models.PointStruct(
                id=i,
                vector=np.random.rand(10).tolist(),
                payload={"random_dig": random.randint(1, 100)},
            )
            for i in range(100)
        ],
    )
    assert (await client.count(collection_name)).count == 100

    assert len((await client.scroll(collection_name, limit=2))[0]) == 2

    assert (
        len(
            await client.search(
                collection_name,
                query_vector=np.random.rand(10).tolist(),  # type: ignore
                limit=10,
            )
        )
        == 10
    )

    assert (
        len(
            await client.search_batch(
                collection_name,
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
                    collection_name,
                    query_vector=np.random.rand(10).tolist(),  # type: ignore
                    limit=4,
                    group_by="random_dig",
                )
            ).groups
        )
        == 4
    )

    if dev or None in (major, minor, patch) or (major, minor, patch) >= (1, 10, 0):
        assert (
            len(
                (
                    await client.query_points(collection_name, query=np.random.rand(10).tolist())
                ).points
            )
            == 10
        )
        query_responses = await client.query_batch_points(
            collection_name, requests=[models.QueryRequest(query=np.random.rand(10).tolist())]
        )
        assert len(query_responses) == 1 and len(query_responses[0].points) == 10

    assert len(await client.recommend(collection_name, positive=[0], limit=5)) == 5
    assert (
        len(
            (
                await client.recommend_groups(
                    collection_name, positive=[1], group_by="random_dig", limit=6
                )
            ).groups
        )
        == 6
    )
    assert (
        len(
            (
                await client.recommend_batch(
                    collection_name,
                    requests=[models.RecommendRequest(positive=[2], limit=7)],
                )
            )[0]
        )
        == 7
    )

    assert len(await client.retrieve(collection_name, ids=[3, 5])) == 2

    await client.create_payload_index(
        collection_name,
        field_name="random_dig",
        field_schema=models.PayloadSchemaType.INTEGER,
    )

    await client.delete_payload_index(collection_name, field_name="random_dig")

    assert await client.get_locks()

    assert len(await client.list_snapshots(collection_name)) == 0
    assert len(await client.list_full_snapshots()) == 0

    snapshots = await client.list_shard_snapshots(collection_name, shard_id=0)
    assert len(snapshots) == 0

    assert (await client.retrieve(collection_name, ids=[0]))[0].vector is None

    await client.update_vectors(
        collection_name,
        points=[models.PointVectors(id=0, vector=[1.0] * 10)],
    )
    assert (await client.retrieve(collection_name, ids=[0], with_vectors=True))[0].vector == [
        1.0
    ] * 10

    await client.delete(collection_name, points_selector=[0])
    assert (await client.count(collection_name)).count == 99

    await client.batch_update_points(
        collection_name,
        update_operations=[
            models.UpsertOperation(
                upsert=models.PointsList(points=[models.PointStruct(id=0, vector=[1.0] * 10)])
            )
        ],
    )
    assert (await client.count(collection_name)).count == 100

    await client.set_payload(collection_name, payload={"added_payload": "zero"}, points=[0])
    assert (await client.retrieve(collection_name, ids=[0], with_payload=["added_payload"]))[
        0
    ].payload == {"added_payload": "zero"}
    await client.overwrite_payload(
        collection_name, payload={"overwritten": True, "rand_digit": 2023}, points=[1]
    )
    assert (await client.retrieve(collection_name, ids=[1]))[0].payload == {
        "overwritten": True,
        "rand_digit": 2023,
    }

    await client.delete_payload(collection_name, keys=["added_payload"], points=[0])
    assert (await client.retrieve(collection_name, ids=[0], with_payload=["added_payload"]))[
        0
    ].payload == {}
    await client.clear_payload(collection_name, points_selector=[1])
    assert (await client.retrieve(collection_name, ids=[1]))[0].payload == {}

    # region teardown
    if await client.collection_exists(collection_name):
        await client.delete_collection(collection_name)
    collections = await client.get_collections()

    assert all(collection.name != collection_name for collection in collections.collections)
    await client.close()
    # endregion


@pytest.mark.asyncio
async def test_async_auth():
    """Test that the auth token provider is called and the token in all modes."""
    token = ""
    call_num = 0

    async def async_auth_token_provider():
        nonlocal token
        nonlocal call_num
        await asyncio.sleep(0.1)
        token = f"token_{call_num}"
        call_num += 1
        return token

    client = AsyncQdrantClient(timeout=3, auth_token_provider=async_auth_token_provider)
    await client.get_collections()
    assert token == "token_0"

    await client.get_collections()
    assert token == "token_1"

    token = ""
    call_num = 0

    client = AsyncQdrantClient(
        prefer_grpc=True, timeout=3, auth_token_provider=async_auth_token_provider
    )
    await client.get_collections()
    assert token == "token_0"

    await client.get_collections()
    assert token == "token_1"

    await client.unlock_storage()
    assert token == "token_2"

    sync_token = ""
    call_num = 0

    def auth_token_provider():
        nonlocal sync_token
        nonlocal call_num
        sync_token = f"token_{call_num}"
        call_num += 1
        return sync_token

    # Additional sync request is sent during client init to check compatibility
    client = AsyncQdrantClient(
        timeout=3, check_compatibility=False, auth_token_provider=auth_token_provider
    )
    await client.get_collections()
    assert sync_token == "token_0"

    await client.get_collections()
    assert sync_token == "token_1"

    sync_token = ""
    call_num = 0

    def auth_token_provider():
        nonlocal sync_token
        nonlocal call_num
        sync_token = f"token_{call_num}"
        call_num += 1
        return sync_token

    # Additional sync request is sent during client init to check compatibility
    client = AsyncQdrantClient(timeout=3, auth_token_provider=auth_token_provider)
    await client.get_collections()
    assert sync_token == "token_1"

    await client.get_collections()
    assert sync_token == "token_2"

    sync_token = ""
    call_num = 0

    client = AsyncQdrantClient(
        prefer_grpc=True,
        timeout=3,
        check_compatibility=False,
        auth_token_provider=auth_token_provider,
    )
    await client.get_collections()
    assert sync_token == "token_0"

    await client.get_collections()
    assert sync_token == "token_1"

    await client.unlock_storage()
    assert sync_token == "token_2"


@pytest.mark.asyncio
@pytest.mark.parametrize("prefer_grpc", [False, True])
async def test_custom_sharding(prefer_grpc: bool):
    collection_name = f"{COLLECTION_NAME}_{uuid.uuid4().hex}"
    client = AsyncQdrantClient(prefer_grpc=prefer_grpc)

    if await client.collection_exists(collection_name):
        await client.delete_collection(collection_name=collection_name)
    await client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(size=DIM, distance=models.Distance.DOT),
        sharding_method=models.ShardingMethod.CUSTOM,
    )

    await client.create_shard_key(collection_name=collection_name, shard_key="cats")
    await client.create_shard_key(collection_name=collection_name, shard_key="dogs")

    collection_info = await client.get_collection(collection_name)

    assert collection_info.config.params.shard_number == 1
    # assert collection_info.config.params.sharding_method == models.ShardingMethod.CUSTOM  # todo: fix in grpc
