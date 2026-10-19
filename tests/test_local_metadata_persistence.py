import inspect
import os
import stat
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import pytest

from qdrant_client import AsyncQdrantClient, QdrantClient, models
from qdrant_client.local import async_qdrant_local, qdrant_local

Client = QdrantClient | AsyncQdrantClient


async def call(method: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    result = method(*args, **kwargs)
    return await result if inspect.isawaitable(result) else result


async def populated_client(client_class: type[Client], directory: Path) -> Client:
    client = client_class(path=str(directory))
    for name in ("docs", "backup"):
        await call(
            client.create_collection,
            name,
            vectors_config=models.VectorParams(size=2, distance=models.Distance.DOT),
            sparse_vectors_config={"sparse": models.SparseVectorParams()},
            metadata={"kept": {"nested": 1}},
        )
        await call(client.upsert, name, [models.PointStruct(id=1, vector=[1.0, 0.0])])
    await call(
        client.update_collection_aliases,
        [
            models.CreateAliasOperation(
                create_alias=models.CreateAlias(collection_name="docs", alias_name="live")
            )
        ],
    )
    return client


async def assert_unchanged(client: Client) -> None:
    assert sorted(c.name for c in (await call(client.get_collections)).collections) == [
        "backup",
        "docs",
    ]
    assert (await call(client.get_aliases)).aliases == [
        models.AliasDescription(alias_name="live", collection_name="docs")
    ]
    for name in ("docs", "backup", "live"):
        assert (await call(client.retrieve, name, [1], with_vectors=True))[0].vector == [1.0, 0.0]
        config = (await call(client.get_collection, name)).config
        assert config.metadata == {"kept": {"nested": 1}}
        assert config.params.sparse_vectors["sparse"].modifier is None


async def change(client: Client, operation: str, metadata: dict[str, Any]) -> None:
    if operation == "create":
        await call(client.create_collection, "new", vectors_config={}, metadata=metadata)
    elif operation == "update":
        await call(
            client.update_collection,
            "docs",
            sparse_vectors_config={
                "sparse": models.SparseVectorParams(modifier=models.Modifier.IDF)
            },
            metadata=metadata,
        )
    else:
        await call(
            client.update_collection_aliases,
            [
                models.CreateAliasOperation(
                    create_alias=models.CreateAlias(collection_name="backup", alias_name="live")
                )
            ],
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("client_class", [QdrantClient, AsyncQdrantClient])
@pytest.mark.parametrize("operation", ["create", "update"])
async def test_failed_metadata_serialization_preserves_store(
    client_class: type[Client], tmp_path: Path, operation: str
) -> None:
    client = await populated_client(client_class, tmp_path)
    original = (tmp_path / "meta.json").read_bytes()
    try:
        with pytest.raises(TypeError, match="not JSON serializable"):
            await change(client, operation, {"kept": {"bad": datetime(2020, 1, 1)}})
        assert (tmp_path / "meta.json").read_bytes() == original
        await assert_unchanged(client)
        # A later successful save must not resurrect the rejected change.
        await call(client.create_collection, "unrelated", vectors_config={})
        await call(client.delete_collection, "unrelated")
    finally:
        await call(client.close)
    reopened = client_class(path=str(tmp_path))
    try:
        await assert_unchanged(reopened)
        await change(reopened, operation, {"valid": True})
    finally:
        await call(reopened.close)
    reopened = client_class(path=str(tmp_path))
    try:
        name = "new" if operation == "create" else "docs"
        assert (await call(reopened.get_collection, name)).config.metadata["valid"] is True
    finally:
        await call(reopened.close)


@pytest.mark.asyncio
@pytest.mark.parametrize("client_class", [QdrantClient, AsyncQdrantClient])
@pytest.mark.parametrize("operation", ["create", "update", "alias"])
@pytest.mark.parametrize("failure", ["write", "fsync", "replace"])
async def test_failed_metadata_write_preserves_store(
    client_class: type[Client],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
    failure: str,
) -> None:
    client = await populated_client(client_class, tmp_path)
    original = (tmp_path / "meta.json").read_bytes()
    module = qdrant_local if client_class is QdrantClient else async_qdrant_local

    class PartialWrite:
        def __init__(self, stream: Any) -> None:
            self.stream = stream

        def write(self, text: str) -> None:
            self.stream.write(text[: len(text) // 2])
            self.stream.flush()
            raise OSError("injected metadata write failure")

    @contextmanager
    def failing_open(filename: str | int, mode: str = "r", **kwargs: Any) -> Any:
        with open(filename, mode, **kwargs) as stream:
            yield (
                PartialWrite(stream)
                if "w" in mode
                and (isinstance(filename, int) or Path(filename).name == "meta.json")
                else stream
            )

    def fail(*args: Any, **kwargs: Any) -> None:
        raise OSError("injected metadata write failure")

    try:
        with monkeypatch.context() as patch:
            if failure == "write":
                patch.setattr(module, "open", failing_open, raising=False)
            else:
                patch.setattr(module.os, failure, fail)
            with pytest.raises(OSError, match="injected metadata write failure"):
                await change(client, operation, {"added": True})
        assert (tmp_path / "meta.json").read_bytes() == original
        await assert_unchanged(client)
        assert sorted(item.name for item in tmp_path.iterdir()) == [
            ".lock",
            "collection",
            "meta.json",
        ]
    finally:
        await call(client.close)
    reopened = client_class(path=str(tmp_path))
    try:
        await assert_unchanged(reopened)
        await change(reopened, operation, {"added": True})
    finally:
        await call(reopened.close)


@pytest.mark.asyncio
@pytest.mark.parametrize("client_class", [QdrantClient, AsyncQdrantClient])
async def test_rejected_collection_update_preserves_earlier_parameters(
    client_class: type[Client], tmp_path: Path
) -> None:
    client = await populated_client(client_class, tmp_path)
    original = (tmp_path / "meta.json").read_bytes()
    try:
        with pytest.raises(ValueError, match="does not exist"):
            await call(
                client.update_collection,
                "docs",
                sparse_vectors_config={
                    "sparse": models.SparseVectorParams(modifier=models.Modifier.IDF),
                    "missing": models.SparseVectorParams(),
                },
            )
        assert (tmp_path / "meta.json").read_bytes() == original
        await assert_unchanged(client)
    finally:
        await call(client.close)


@pytest.mark.asyncio
@pytest.mark.parametrize("client_class", [QdrantClient, AsyncQdrantClient])
@pytest.mark.parametrize("operation", ["create", "update", "alias"])
async def test_metadata_commit_is_not_rolled_back_by_cleanup(
    client_class: type[Client], tmp_path: Path, monkeypatch: pytest.MonkeyPatch, operation: str
) -> None:
    client = await populated_client(client_class, tmp_path)
    module = qdrant_local if client_class is QdrantClient else async_qdrant_local
    real_replace = module.os.replace
    real_rmdir = module.os.rmdir
    real_unlink = module.os.unlink
    committed = False

    def replace(*args: Any, **kwargs: Any) -> None:
        nonlocal committed
        real_replace(*args, **kwargs)
        committed = True

    def rmdir(*args: Any, **kwargs: Any) -> None:
        if committed:
            raise OSError("injected cleanup failure after metadata commit")
        real_rmdir(*args, **kwargs)

    def unlink(*args: Any, **kwargs: Any) -> None:
        if committed:
            raise OSError("injected cleanup failure after metadata commit")
        real_unlink(*args, **kwargs)

    try:
        with monkeypatch.context() as patch:
            patch.setattr(module.os, "replace", replace)
            patch.setattr(module.os, "rmdir", rmdir)
            patch.setattr(module.os, "unlink", unlink)
            await change(client, operation, {"added": True})
        assert committed
    finally:
        await call(client.close)
    reopened = client_class(path=str(tmp_path))
    try:
        if operation == "alias":
            assert (await call(reopened.get_aliases)).aliases == [
                models.AliasDescription(alias_name="live", collection_name="backup")
            ]
        else:
            name = "new" if operation == "create" else "docs"
            config = (await call(reopened.get_collection, name)).config
            assert config.metadata["added"] is True
            if operation == "update":
                assert config.params.sparse_vectors["sparse"].modifier == models.Modifier.IDF
        for name in ("docs", "backup"):
            assert (await call(reopened.count, name)).count == 1
    finally:
        await call(reopened.close)


@pytest.mark.asyncio
@pytest.mark.parametrize("client_class", [QdrantClient, AsyncQdrantClient])
@pytest.mark.parametrize("cleanup_fails", [False, True])
async def test_failed_metadata_open_closes_descriptor(
    client_class: type[Client],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cleanup_fails: bool,
) -> None:
    client = await populated_client(client_class, tmp_path)
    original = (tmp_path / "meta.json").read_bytes()
    module = qdrant_local if client_class is QdrantClient else async_qdrant_local
    real_mkstemp = module.tempfile.mkstemp
    descriptors: list[int] = []

    def mkstemp(*args: Any, **kwargs: Any) -> tuple[int, str]:
        fd, filename = real_mkstemp(*args, **kwargs)
        descriptors.append(fd)
        return fd, filename

    def fail_open(*args: Any, **kwargs: Any) -> None:
        raise OSError("injected metadata open failure")

    def fail_cleanup(*args: Any, **kwargs: Any) -> None:
        raise OSError("injected secondary cleanup failure")

    try:
        with monkeypatch.context() as patch:
            patch.setattr(module.tempfile, "mkstemp", mkstemp)
            patch.setattr(module, "open", fail_open, raising=False)
            if cleanup_fails:
                patch.setattr(module.os, "unlink", fail_cleanup)
            with pytest.raises(OSError, match="injected metadata open failure"):
                await change(client, "create", {"added": True})
        assert len(descriptors) == 1
        with pytest.raises(OSError):
            module.os.fstat(descriptors[0])
        assert (tmp_path / "meta.json").read_bytes() == original
        await assert_unchanged(client)
    finally:
        await call(client.close)


@pytest.mark.asyncio
@pytest.mark.skipif(os.name != "posix", reason="POSIX file modes")
@pytest.mark.parametrize("client_class", [QdrantClient, AsyncQdrantClient])
async def test_metadata_save_keeps_file_mode(client_class: type[Client], tmp_path: Path) -> None:
    client = client_class(path=str(tmp_path))
    meta_path = tmp_path / "meta.json"
    try:
        meta_path.chmod(0o640)
        await call(client.create_collection, "docs", vectors_config={})
        assert stat.S_IMODE(meta_path.stat().st_mode) == 0o640
    finally:
        await call(client.close)


@pytest.mark.asyncio
@pytest.mark.skipif(os.name != "posix", reason="directories cannot be fsynced on Windows")
@pytest.mark.parametrize("client_class", [QdrantClient, AsyncQdrantClient])
@pytest.mark.parametrize("dir_fsync_fails", [False, True])
async def test_metadata_replace_fsyncs_directory(
    client_class: type[Client],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    dir_fsync_fails: bool,
) -> None:
    client = await populated_client(client_class, tmp_path)
    module = qdrant_local if client_class is QdrantClient else async_qdrant_local
    real_fsync = module.os.fsync
    synced_dirs: list[int] = []

    def fsync(fd: int) -> None:
        if stat.S_ISDIR(os.fstat(fd).st_mode):
            synced_dirs.append(fd)
            if dir_fsync_fails:
                raise OSError("injected directory fsync failure")
        real_fsync(fd)

    try:
        with monkeypatch.context() as patch:
            patch.setattr(module.os, "fsync", fsync)
            await change(client, "create", {"added": True})
        assert synced_dirs
        assert (await call(client.get_collection, "new")).config.metadata == {"added": True}
    finally:
        await call(client.close)
    reopened = client_class(path=str(tmp_path))
    try:
        assert (await call(reopened.get_collection, "new")).config.metadata == {"added": True}
    finally:
        await call(reopened.close)
