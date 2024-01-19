from typing import List

from qdrant_client.client_base import QdrantBase
from qdrant_client.http import models
from tests.congruence_tests.test_common import (
    COLLECTION_NAME,
    compare_client_results,
    generate_fixtures,
    init_client,
    init_local,
    init_remote,
)


class TestAliasRetriever:
    __test__ = False

    def __init__(self, collection_name=COLLECTION_NAME):
        self.collection_name = collection_name

    @classmethod
    def list_aliases(cls, client: QdrantBase) -> List[models.AliasDescription]:
        aliases = client.get_aliases()
        return sorted(aliases.aliases, key=lambda x: x.alias_name)

    def list_collection_aliases(self, client: QdrantBase) -> List[models.AliasDescription]:
        aliases = client.get_collection_aliases(collection_name=self.collection_name)
        return sorted(aliases.aliases, key=lambda x: x.alias_name)


def test_alias_changes():
    fixture_points = generate_fixtures(10)

    retriever = TestAliasRetriever()

    local_client = init_local()
    init_client(local_client, fixture_points)

    remote_client = init_remote()
    init_client(remote_client, fixture_points)

    alias_name = "test_alias"

    ops = [
        models.CreateAliasOperation(
            create_alias=models.CreateAlias(
                collection_name=COLLECTION_NAME,
                alias_name=alias_name,
            )
        )
    ]

    local_client.update_collection_aliases(change_aliases_operations=ops)
    remote_client.update_collection_aliases(change_aliases_operations=ops)

    compare_client_results(local_client, remote_client, retriever.list_aliases)
    compare_client_results(local_client, remote_client, retriever.list_collection_aliases)

    ops = [
        models.CreateAliasOperation(
            create_alias=models.CreateAlias(
                collection_name=COLLECTION_NAME,
                alias_name=alias_name + "_new",
            )
        ),
        models.CreateAliasOperation(
            create_alias=models.CreateAlias(
                collection_name=COLLECTION_NAME,
                alias_name=alias_name + "_new2",
            )
        ),
    ]

    local_client.update_collection_aliases(change_aliases_operations=ops)
    remote_client.update_collection_aliases(change_aliases_operations=ops)

    compare_client_results(local_client, remote_client, retriever.list_aliases)
    compare_client_results(local_client, remote_client, retriever.list_collection_aliases)

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
    compare_client_results(local_client, remote_client, retriever.list_collection_aliases)
