import json
from collections import UserList
from unittest.mock import AsyncMock, Mock, PropertyMock

import httpx
import pytest

from qdrant_client import AsyncQdrantClient, QdrantClient, grpc, models
from qdrant_client.async_qdrant_remote import AsyncQdrantRemote
from qdrant_client.conversions.conversion import RestToGrpc
from qdrant_client.qdrant_remote import QdrantRemote


def point_list():
    return [
        models.PointStruct(id=1, vector=[1.0, 2.0], payload={"name": "first"}),
        models.PointStruct(id=2, vector=[3.0, 4.0], payload={"name": "second"}),
    ]


@pytest.mark.parametrize("sequence_type", [list, tuple, UserList])
def test_local_upsert_accepts_point_sequences(sequence_type):
    client = QdrantClient(":memory:")
    try:
        client.create_collection(
            "test",
            vectors_config=models.VectorParams(size=2, distance=models.Distance.DOT),
        )
        points = point_list()
        result = client.upsert("test", points=sequence_type(points))
        assert result.status == models.UpdateStatus.COMPLETED
        records = client.retrieve("test", ids=[1, 2], with_vectors=True)
        assert [(record.id, record.vector, record.payload) for record in records] == [
            (point.id, point.vector, point.payload) for point in points
        ]
        client.upsert("test", points=sequence_type([]))
        assert client.count("test").count == 2
    finally:
        client.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("sequence_type", [list, tuple, UserList])
async def test_async_local_upsert_accepts_point_sequences(sequence_type):
    client = AsyncQdrantClient(":memory:")
    try:
        await client.create_collection(
            "test",
            vectors_config=models.VectorParams(size=2, distance=models.Distance.DOT),
        )
        points = point_list()
        result = await client.upsert("test", points=sequence_type(points))
        assert result.status == models.UpdateStatus.COMPLETED
        records = await client.retrieve("test", ids=[1, 2], with_vectors=True)
        assert [(record.id, record.vector, record.payload) for record in records] == [
            (point.id, point.vector, point.payload) for point in points
        ]
        await client.upsert("test", points=sequence_type([]))
        assert (await client.count("test")).count == 2
    finally:
        await client.close()


@pytest.mark.parametrize("sequence_type", [list, tuple, UserList])
def test_local_upsert_rejects_invalid_sequence_before_writing(sequence_type):
    client = QdrantClient(":memory:")
    try:
        client.create_collection(
            "test",
            vectors_config=models.VectorParams(size=2, distance=models.Distance.DOT),
        )
        client.upsert("test", points=point_list()[:1])
        points = sequence_type(
            [
                models.PointStruct(id=1, vector=[5.0, 6.0], payload={"name": "changed"}),
                models.PointStruct(id=2, vector=[7.0]),
            ]
        )
        with pytest.raises(ValueError, match="expected dim: 2, got 1"):
            client.upsert("test", points=points)
        assert client.count("test").count == 1
        record = client.retrieve("test", ids=[1], with_vectors=True)[0]
        assert record.vector == [1.0, 2.0]
        assert record.payload == {"name": "first"}
    finally:
        client.close()


@pytest.mark.parametrize("points", ["", b"", bytearray()])
def test_local_upsert_rejects_text_and_byte_sequences(points):
    client = QdrantClient(":memory:")
    try:
        client.create_collection(
            "test",
            vectors_config=models.VectorParams(size=2, distance=models.Distance.DOT),
        )
        with pytest.raises(ValueError, match="Unsupported type"):
            client.upsert("test", points=points)
        assert client.count("test").count == 0
    finally:
        client.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("is_async", [False, True])
@pytest.mark.parametrize("prefer_grpc", [False, True])
@pytest.mark.parametrize("sequence_type", [list, tuple, UserList])
@pytest.mark.parametrize("grpc_input", [False, True])
@pytest.mark.parametrize("empty", [False, True])
async def test_remote_upsert_serializes_point_sequences(
    is_async, prefer_grpc, sequence_type, grpc_input, empty, mocker
):
    requests = []

    def handler(request):
        requests.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "result": {"operation_id": 1, "status": "completed"},
                "status": "ok",
                "time": 0.0,
            },
        )

    client_type = AsyncQdrantRemote if is_async else QdrantRemote
    client = client_type(
        prefer_grpc=prefer_grpc,
        check_compatibility=False,
        transport=httpx.MockTransport(handler),
    )
    points = [] if empty else point_list()
    grpc_points = [RestToGrpc.convert_point_struct(point) for point in points]
    sequence = sequence_type(grpc_points if grpc_input else points)
    if prefer_grpc:
        upsert = (AsyncMock if is_async else Mock)(
            return_value=grpc.PointsOperationResponse(
                result=grpc.UpdateResult(operation_id=1, status=grpc.UpdateStatus.Completed)
            )
        )
        mocker.patch.object(
            client_type, "grpc_points", new_callable=PropertyMock, return_value=Mock(Upsert=upsert)
        )
    try:
        result = client.upsert("test", points=sequence, update_mode=models.UpdateMode.INSERT_ONLY)
        if is_async:
            result = await result
        assert result.status == models.UpdateStatus.COMPLETED
        if prefer_grpc:
            assert upsert.call_args.args[0] == grpc.UpsertPoints(
                collection_name="test",
                wait=True,
                points=grpc_points,
                update_mode=grpc.UpdateMode.InsertOnly,
            )
        else:
            assert requests == [
                {
                    "points": [
                        {"id": point.id, "vector": point.vector, "payload": point.payload}
                        for point in points
                    ],
                    "update_mode": "insert_only",
                }
            ]
    finally:
        if is_async:
            await client.close()
        else:
            client.close()
