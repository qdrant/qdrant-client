import uuid

from qdrant_client.client_base import QdrantBase
from qdrant_client.http import models
from tests.congruence_tests.test_common import (
    COLLECTION_NAME,
    compare_client_results,
    generate_fixtures,
    init_client,
)


class TestAliasRetriever:
    __test__ = False

    @classmethod
    def list_aliases(cls, client: QdrantBase) -> list[models.AliasDescription]:
        aliases = client.get_aliases()
        return sorted(aliases.aliases, key=lambda x: x.alias_name)

    def list_collection_aliases(
        self, client: QdrantBase, collection_name: str = COLLECTION_NAME
    ) -> list[models.AliasDescription]:
        aliases = client.get_collection_aliases(collection_name=collection_name)
        return sorted(aliases.aliases, key=lambda x: x.alias_name)


def test_alias_changes(local_client: QdrantBase, remote_client: QdrantBase):
    fixture_points = generate_fixtures(10)

    retriever = TestAliasRetriever()

    collection_name = f"{COLLECTION_NAME}_{uuid.uuid4().hex}"
    init_client(local_client, fixture_points, collection_name)
    init_client(remote_client, fixture_points, collection_name)

    alias_name = "test_alias"

    ops = [
        models.CreateAliasOperation(
            create_alias=models.CreateAlias(
                collection_name=collection_name,
                alias_name=alias_name,
            )
        )
    ]

    local_client.update_collection_aliases(change_aliases_operations=ops)
    remote_client.update_collection_aliases(change_aliases_operations=ops)

    compare_client_results(local_client, remote_client, retriever.list_aliases)
    compare_client_results(
        local_client,
        remote_client,
        retriever.list_collection_aliases,
        collection_name=collection_name,
    )

    ops = [
        models.CreateAliasOperation(
            create_alias=models.CreateAlias(
                collection_name=collection_name,
                alias_name=alias_name + "_new",
            )
        ),
        models.CreateAliasOperation(
            create_alias=models.CreateAlias(
                collection_name=collection_name,
                alias_name=alias_name + "_new2",
            )
        ),
    ]

    local_client.update_collection_aliases(change_aliases_operations=ops)
    remote_client.update_collection_aliases(change_aliases_operations=ops)

    compare_client_results(local_client, remote_client, retriever.list_aliases)
    compare_client_results(
        local_client,
        remote_client,
        retriever.list_collection_aliases,
        collection_name=collection_name,
    )

    ops = [
        models.DeleteAliasOperation(
            delete_alias=models.DeleteAlias(alias_name=alias_name + "_new")
        ),
        models.RenameAliasOperation(
            rename_alias=models.RenameAlias(
                old_alias_name=alias_name + "_new2",
                new_alias_name=alias_name + "_new3",
            )
        ),
    ]

    local_client.update_collection_aliases(change_aliases_operations=ops)
    remote_client.update_collection_aliases(change_aliases_operations=ops)

    compare_client_results(local_client, remote_client, retriever.list_aliases)
    compare_client_results(
        local_client,
        remote_client,
        retriever.list_collection_aliases,
        collection_name=collection_name,
    )
