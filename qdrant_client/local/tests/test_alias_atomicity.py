import threading
from unittest.mock import patch
from concurrent.futures import ThreadPoolExecutor

import pytest

from qdrant_client import AsyncQdrantClient, QdrantClient, models


def _create_alias(collection_name, alias_name):
    return models.CreateAliasOperation(
        create_alias=models.CreateAlias(collection_name=collection_name, alias_name=alias_name)
    )


def _failed_alias_batch():
    return [
        _create_alias("docs_v2", "live"),
        models.RenameAliasOperation(
            rename_alias=models.RenameAlias(old_alias_name="missing", new_alias_name="other")
        ),
    ]


@pytest.mark.parametrize("persistent", [False, True])
def test_failed_alias_batch_preserves_existing_alias(tmp_path, persistent):
    location = str(tmp_path) if persistent else ":memory:"
    client = QdrantClient(path=location) if persistent else QdrantClient(location)
    client.create_collection("docs_v1", vectors_config={})
    client.create_collection("docs_v2", vectors_config={})
    client.update_collection_aliases([_create_alias("docs_v1", "live")])

    with pytest.raises(KeyError, match="missing"):
        client.update_collection_aliases(_failed_alias_batch())

    assert client.get_collection_aliases("docs_v1").aliases == [
        models.AliasDescription(alias_name="live", collection_name="docs_v1")
    ]
    assert client.get_collection_aliases("docs_v2").aliases == []
    client.close()

    if persistent:
        client = QdrantClient(path=location)
        assert client.get_collection_aliases("docs_v1").aliases == [
            models.AliasDescription(alias_name="live", collection_name="docs_v1")
        ]
        client.close()


@pytest.mark.asyncio
async def test_async_failed_alias_batch_preserves_existing_alias():
    client = AsyncQdrantClient(":memory:")
    await client.create_collection("docs_v1", vectors_config={})
    await client.create_collection("docs_v2", vectors_config={})
    await client.update_collection_aliases([_create_alias("docs_v1", "live")])

    with pytest.raises(KeyError, match="missing"):
        await client.update_collection_aliases(_failed_alias_batch())

    assert (await client.get_collection_aliases("docs_v1")).aliases == [
        models.AliasDescription(alias_name="live", collection_name="docs_v1")
    ]
    assert (await client.get_collection_aliases("docs_v2")).aliases == []
    await client.close()


@pytest.mark.asyncio
async def test_alias_cannot_target_another_alias_async():
    client = AsyncQdrantClient(":memory:")
    await client.create_collection("docs", vectors_config={})
    await client.update_collection_aliases([_create_alias("docs", "old")])
    with pytest.raises(ValueError, match="Collection old not found"):
        await client.update_collection_aliases([_create_alias("old", "new")])
    assert (await client.get_aliases()).aliases == [
        models.AliasDescription(alias_name="old", collection_name="docs")
    ]
    await client.close()


def test_alias_cannot_target_another_alias():
    client = QdrantClient(":memory:")
    client.create_collection("docs", vectors_config={})
    client.update_collection_aliases([_create_alias("docs", "old")])
    with pytest.raises(ValueError, match="Collection old not found"):
        client.update_collection_aliases([_create_alias("old", "new")])
    assert client.get_aliases().aliases == [
        models.AliasDescription(alias_name="old", collection_name="docs")
    ]
    client.close()


def test_concurrent_alias_batches_keep_both_results():
    class PausingAliases(dict):
        def __init__(self):
            super().__init__()
            self.barrier = threading.Barrier(2)

        def copy(self):
            try:
                self.barrier.wait(timeout=0.05)
            except threading.BrokenBarrierError:
                pass
            return super().copy()

    client = QdrantClient(":memory:")
    client.create_collection("docs", vectors_config={})
    for _ in range(20):
        client._client.aliases = PausingAliases()
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [
                pool.submit(client.update_collection_aliases, [_create_alias("docs", name)])
                for name in ("first", "second")
            ]
            for future in futures:
                assert future.result()
        assert {alias.alias_name for alias in client.get_aliases().aliases} == {"first", "second"}
    client.close()


@pytest.mark.parametrize("persistent", [False, True])
def test_create_alias_after_close_is_rejected(tmp_path, persistent):
    client = QdrantClient(path=str(tmp_path)) if persistent else QdrantClient(":memory:")
    client.create_collection("docs", vectors_config={})
    client.close()
    with pytest.raises(RuntimeError, match="closed"):
        client.update_collection_aliases([_create_alias("docs", "live")])
    assert client._client.aliases == {}


@pytest.mark.asyncio
async def test_async_create_alias_after_close_is_rejected():
    client = AsyncQdrantClient(":memory:")
    await client.create_collection("docs", vectors_config={})
    await client.close()
    with pytest.raises(RuntimeError, match="closed"):
        await client.update_collection_aliases([_create_alias("docs", "live")])
    assert client._client.aliases == {}


@pytest.mark.asyncio
async def test_async_save_failure_leaves_aliases_unchanged(tmp_path):
    client = AsyncQdrantClient(path=str(tmp_path))
    await client.create_collection("docs", vectors_config={})
    with patch.object(client._client, "_save", side_effect=OSError("disk full")):
        with pytest.raises(OSError, match="disk full"):
            await client.update_collection_aliases([_create_alias("docs", "live")])
    assert client._client.aliases == {}
    await client.close()


def test_save_failure_leaves_aliases_unchanged(tmp_path):
    client = QdrantClient(path=str(tmp_path))
    client.create_collection("docs", vectors_config={})
    with patch.object(client._client, "_save", side_effect=OSError("disk full")):
        with pytest.raises(OSError, match="disk full"):
            client.update_collection_aliases([_create_alias("docs", "live")])
    assert client._client.aliases == {}
    client.close()


def test_collection_delete_cannot_race_with_alias_batch():
    class PausingAliases(dict):
        copied = threading.Event()
        proceed = threading.Event()

        def copy(self):
            result = super().copy()
            self.copied.set()
            if not self.proceed.wait(timeout=2):
                raise TimeoutError("alias update was not released")
            return result

    client = QdrantClient(":memory:")
    client.create_collection("docs", vectors_config={})
    aliases = PausingAliases()
    client._client.aliases = aliases
    with ThreadPoolExecutor(max_workers=2) as pool:
        update = pool.submit(client.update_collection_aliases, [_create_alias("docs", "live")])
        assert aliases.copied.wait(timeout=2)
        deletion = pool.submit(client.delete_collection, "docs")
        aliases.proceed.set()
        assert update.result(timeout=2)
        assert deletion.result(timeout=2)
    assert client.get_aliases().aliases == []
    client.close()
