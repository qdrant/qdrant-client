import threading
from unittest import mock

import httpx
import pytest

from qdrant_client import models
from qdrant_client.http.api import points_api
from qdrant_client.http.api_client import AsyncApis


@pytest.mark.asyncio
async def test_upsert_serialization_does_not_run_on_event_loop_thread():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "time": 0.0,
                "status": "ok",
                "result": {"operation_id": 0, "status": "completed"},
            },
        )

    apis = AsyncApis(host="http://localhost:6333", transport=httpx.MockTransport(handler))
    loop_thread = threading.get_ident()
    serialization_threads = []
    original_encoder = points_api.jsonable_encoder

    def recording_encoder(*args, **kwargs):
        serialization_threads.append(threading.get_ident())
        return original_encoder(*args, **kwargs)

    with mock.patch.object(points_api, "jsonable_encoder", recording_encoder):
        response = await apis.points_api.upsert_points(
            collection_name="test",
            point_insert_operations=models.PointsList(
                points=[models.PointStruct(id=1, vector=[0.1, 0.2])]
            ),
        )
    await apis.aclose()

    assert response.result.status == models.UpdateStatus.COMPLETED
    assert serialization_threads and loop_thread not in serialization_threads
