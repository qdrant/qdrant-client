import pytest

from qdrant_client.client_base import QdrantBase
from qdrant_client.http import models
from qdrant_client.http.exceptions import UnexpectedResponse
from tests.congruence_tests.test_common import (
    COLLECTION_NAME,
    compare_client_results,
    generate_fixtures,
    init_client,
    init_local,
    init_remote,
)
from tests.utils import read_version


class TestAliasRetriever:
    __test__ = False

    def __init__(self, collection_name=COLLECTION_NAME):
        self.collection_name = collection_name

    @classmethod
    def list_aliases(cls, client: QdrantBase) -> list[models.AliasDescription]:
        aliases = client.get_aliases()
        return sorted(aliases.aliases, key=lambda x: x.alias_name)

    def list_collection_aliases(self, client: QdrantBase) -> list[models.AliasDescription]:
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


def test_rejected_alias_changes_leave_aliases_untouched():
    """A rejected batch of alias changes must not apply any of its operations.

    Regression: local mode applied the operations one at a time, so the ones before the
    rejected operation stayed applied, and the next save wrote them to disk. It also accepted
    an alias as the target of a new alias, and an alias named after an existing collection,
    both of which the server rejects.
    """
    major, minor, patch, dev = read_version()
    if not dev and None not in (major, minor, patch) and (major, minor, patch) < (1, 19, 2):
        pytest.skip("Alias changes are applied atomically as of qdrant 1.19.2")

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

    # the rename is rejected, so the alias created before it must not appear
    ops = [
        models.CreateAliasOperation(
            create_alias=models.CreateAlias(
                collection_name=COLLECTION_NAME,
                alias_name=alias_name + "_new",
            )
        ),
        models.RenameAliasOperation(
            rename_alias=models.RenameAlias(
                old_alias_name="missing_alias",
                new_alias_name=alias_name + "_renamed",
            )
        ),
    ]

    with pytest.raises(ValueError):
        local_client.update_collection_aliases(change_aliases_operations=ops)
    with pytest.raises(UnexpectedResponse):
        remote_client.update_collection_aliases(change_aliases_operations=ops)

    compare_client_results(local_client, remote_client, retriever.list_aliases)
    compare_client_results(local_client, remote_client, retriever.list_collection_aliases)

    # an alias must point at a collection, not at another alias
    ops = [
        models.CreateAliasOperation(
            create_alias=models.CreateAlias(
                collection_name=alias_name,
                alias_name=alias_name + "_new",
            )
        )
    ]

    with pytest.raises(ValueError):
        local_client.update_collection_aliases(change_aliases_operations=ops)
    with pytest.raises(UnexpectedResponse):
        remote_client.update_collection_aliases(change_aliases_operations=ops)

    compare_client_results(local_client, remote_client, retriever.list_aliases)
    compare_client_results(local_client, remote_client, retriever.list_collection_aliases)

    # an alias must not take the name of an existing collection
    ops = [
        models.CreateAliasOperation(
            create_alias=models.CreateAlias(
                collection_name=COLLECTION_NAME,
                alias_name=COLLECTION_NAME,
            )
        )
    ]

    with pytest.raises(ValueError):
        local_client.update_collection_aliases(change_aliases_operations=ops)
    with pytest.raises(UnexpectedResponse):
        remote_client.update_collection_aliases(change_aliases_operations=ops)

    compare_client_results(local_client, remote_client, retriever.list_aliases)
    compare_client_results(local_client, remote_client, retriever.list_collection_aliases)

    # renaming an alias to the name of an existing collection must also be rejected
    ops = [
        models.RenameAliasOperation(
            rename_alias=models.RenameAlias(
                old_alias_name=alias_name,
                new_alias_name=COLLECTION_NAME,
            )
        )
    ]

    with pytest.raises(ValueError):
        local_client.update_collection_aliases(change_aliases_operations=ops)
    with pytest.raises(UnexpectedResponse):
        remote_client.update_collection_aliases(change_aliases_operations=ops)

    compare_client_results(local_client, remote_client, retriever.list_aliases)
    compare_client_results(local_client, remote_client, retriever.list_collection_aliases)

    # the create is rejected, so the delete before it must not be applied either
    ops = [
        models.DeleteAliasOperation(delete_alias=models.DeleteAlias(alias_name=alias_name)),
        models.CreateAliasOperation(
            create_alias=models.CreateAlias(
                collection_name=alias_name,
                alias_name=alias_name + "_new",
            )
        ),
    ]

    with pytest.raises(ValueError):
        local_client.update_collection_aliases(change_aliases_operations=ops)
    with pytest.raises(UnexpectedResponse):
        remote_client.update_collection_aliases(change_aliases_operations=ops)

    compare_client_results(local_client, remote_client, retriever.list_aliases)
    compare_client_results(local_client, remote_client, retriever.list_collection_aliases)


def test_collection_cannot_take_alias_name():
    """A new collection must not take the name of an existing alias."""
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

    with pytest.raises(ValueError):
        local_client.create_collection(alias_name, vectors_config={})
    with pytest.raises(UnexpectedResponse):
        remote_client.create_collection(alias_name, vectors_config={})

    compare_client_results(local_client, remote_client, retriever.list_aliases)
    compare_client_results(local_client, remote_client, retriever.list_collection_aliases)
