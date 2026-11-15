import asyncio
import builtins
from pathlib import Path
from unittest.mock import patch

import httpx
import pytest

from qdrant_client import AsyncQdrantClient, QdrantClient, models
from qdrant_client.http.exceptions import UnexpectedResponse


class SnapshotTransport(httpx.BaseTransport, httpx.AsyncBaseTransport):
    def __init__(self, fail=False, cancel=False):
        self.fail = fail
        self.cancel = cancel
        self.requests = []
        self.chunk_sizes = []
        self.bytes_sent = 0
        self.file_bytes = 0
        self.prefix = []

    def _check_request(self, request):
        self.requests.append(request)
        assert request.method == "POST"
        assert request.url.path == "/collections/test/snapshots/upload"
        assert request.url.params == httpx.QueryParams(
            {"wait": "false", "priority": "snapshot", "checksum": "a" * 64}
        )
        assert request.headers["api-key"] == "test-key"
        assert request.headers["content-type"].startswith("multipart/form-data; boundary=")

    def _response(self, request):
        return httpx.Response(
            500 if self.fail else 202,
            json={"result": None, "status": "ok", "time": 0.1},
            request=request,
        )

    def _chunk(self, chunk):
        self.chunk_sizes.append(len(chunk))
        self.bytes_sent += len(chunk)
        if len(self.prefix) < 2:
            self.prefix.append(chunk)
        if chunk and chunk == b"x" * len(chunk):
            self.file_bytes += len(chunk)

    def handle_request(self, request):
        self._check_request(request)
        for chunk in request.stream:
            self._chunk(chunk)
        return self._response(request)

    async def handle_async_request(self, request):
        self._check_request(request)
        async for chunk in request.stream:
            self._chunk(chunk)
            if self.cancel and len(self.chunk_sizes) >= 3:
                raise asyncio.CancelledError()
        return self._response(request)


OPTIONS = {
    "wait": False,
    "priority": models.SnapshotPriority.SNAPSHOT,
    "checksum": "a" * 64,
}


def client_kwargs(transport):
    return dict(
        url="https://example.invalid",
        api_key="test-key",
        check_compatibility=False,
        prefer_grpc=True,
        transport=transport,
    )


def tracked_open(handles):
    def open_file(*args, **kwargs):
        file = builtins.open(*args, **kwargs)
        handles.append(file)
        return file

    return open_file


@pytest.mark.parametrize("as_string", [False, True])
@pytest.mark.parametrize("fail", [False, True])
def test_sync_upload_streams_and_closes(tmp_path, as_string, fail):
    path = tmp_path / "backup.snapshot"
    size = 2 * 1024 * 1024
    path.write_bytes(b"x" * size)
    transport = SnapshotTransport(fail=fail)
    handles = []
    client = QdrantClient(**client_kwargs(transport))
    try:
        with patch("qdrant_client.qdrant_remote.open", tracked_open(handles), create=True):
            if fail:
                with pytest.raises(UnexpectedResponse):
                    client.recover_from_uploaded_snapshot(
                        "test", str(path) if as_string else path, **OPTIONS
                    )
            else:
                assert (
                    client.recover_from_uploaded_snapshot(
                        "test", str(path) if as_string else path, **OPTIONS
                    )
                    is None
                )
        assert len(handles) == 1 and handles[0].closed
        assert len(transport.requests) == 1
        assert transport.bytes_sent > size
        assert transport.file_bytes == size
        assert b'name="snapshot"; filename="backup.snapshot"' in b"".join(transport.prefix)
        assert len(transport.chunk_sizes) > 32
        assert max(transport.chunk_sizes) <= 65536
    finally:
        client.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["success", "failure", "cancel"])
async def test_async_upload_streams_and_closes(tmp_path, outcome):
    path = tmp_path / "backup.snapshot"
    path.write_bytes(b"x" * (2 * 1024 * 1024))
    transport = SnapshotTransport(fail=outcome == "failure", cancel=outcome == "cancel")
    handles = []
    client = AsyncQdrantClient(**client_kwargs(transport))
    try:
        with patch("qdrant_client.async_qdrant_remote.open", tracked_open(handles), create=True):
            if outcome == "success":
                assert await client.recover_from_uploaded_snapshot("test", path, **OPTIONS) is None
            else:
                with pytest.raises(
                    asyncio.CancelledError if outcome == "cancel" else UnexpectedResponse
                ):
                    await client.recover_from_uploaded_snapshot("test", path, **OPTIONS)
        assert len(handles) == 1 and handles[0].closed
        assert len(transport.requests) == 1
        assert max(transport.chunk_sizes) <= 65536
        if outcome == "success":
            assert transport.file_bytes == path.stat().st_size
            assert b'name="snapshot"; filename="backup.snapshot"' in b"".join(transport.prefix)
    finally:
        await client.close()


def test_sync_invalid_file_does_not_send(tmp_path):
    transport = SnapshotTransport()
    client = QdrantClient(**client_kwargs(transport))
    try:
        with pytest.raises(FileNotFoundError):
            client.recover_from_uploaded_snapshot("test", tmp_path / "absent", **OPTIONS)
        with pytest.raises(AssertionError, match="Unknown arguments"):
            client.recover_from_uploaded_snapshot("test", tmp_path / "absent", nonsense=True)
        assert not transport.requests
    finally:
        client.close()


@pytest.mark.asyncio
async def test_async_invalid_file_does_not_send(tmp_path):
    transport = SnapshotTransport()
    client = AsyncQdrantClient(**client_kwargs(transport))
    try:
        with pytest.raises(FileNotFoundError):
            await client.recover_from_uploaded_snapshot("test", tmp_path / "absent", **OPTIONS)
        with pytest.raises(AssertionError, match="Unknown arguments"):
            await client.recover_from_uploaded_snapshot("test", tmp_path / "absent", nonsense=True)
        assert not transport.requests
    finally:
        await client.close()


def test_local_mode_rejects_before_opening():
    client = QdrantClient(":memory:")
    try:
        with pytest.raises(NotImplementedError, match="Snapshots are not supported"):
            client.recover_from_uploaded_snapshot("test", Path("nonexistent.snapshot"))
    finally:
        client.close()


@pytest.mark.asyncio
async def test_async_local_mode_rejects_before_opening():
    client = AsyncQdrantClient(":memory:")
    try:
        with pytest.raises(NotImplementedError, match="Snapshots are not supported"):
            await client.recover_from_uploaded_snapshot("test", Path("nonexistent.snapshot"))
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_task_cancellation_closes_upload(tmp_path):
    path = tmp_path / "backup.snapshot"
    path.write_bytes(b"x" * (2 * 1024 * 1024))
    started = asyncio.Event()

    class BlockingTransport(SnapshotTransport):
        async def handle_async_request(self, request):
            async for chunk in request.stream:
                self._chunk(chunk)
                if self.file_bytes:
                    started.set()
                    await asyncio.Future()
            raise AssertionError("upload unexpectedly completed")

    transport = BlockingTransport()
    handles = []
    client = AsyncQdrantClient(**client_kwargs(transport))
    try:
        with patch("qdrant_client.async_qdrant_remote.open", tracked_open(handles), create=True):
            task = asyncio.create_task(client.recover_from_uploaded_snapshot("test", path))
            await asyncio.wait_for(started.wait(), timeout=5)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        assert len(handles) == 1 and handles[0].closed
        assert 0 < transport.file_bytes < path.stat().st_size
    finally:
        await client.close()


@pytest.mark.parametrize("async_client", [False, True])
def test_default_options_and_success_result(tmp_path, async_client):
    path = tmp_path / "backup.snapshot"
    path.write_bytes(b"snapshot bytes")

    def respond(request):
        assert request.url.params == httpx.QueryParams({"wait": "true"})
        assert b"snapshot bytes" in request.content
        return httpx.Response(200, json={"result": True, "status": "ok", "time": 0.1})

    transport = httpx.MockTransport(respond)
    if async_client:

        async def run():
            client = AsyncQdrantClient(**client_kwargs(transport))
            try:
                assert await client.recover_from_uploaded_snapshot("test", path) is True
            finally:
                await client.close()

        asyncio.run(run())
    else:
        client = QdrantClient(**client_kwargs(transport))
        try:
            assert client.recover_from_uploaded_snapshot("test", path) is True
        finally:
            client.close()
