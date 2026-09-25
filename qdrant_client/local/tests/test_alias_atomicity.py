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
