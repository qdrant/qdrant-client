import pytest

from qdrant_client import QdrantClient
from qdrant_client.local.qdrant_local import QdrantLocal
from tests.congruence_tests.test_common import (
    delete_fixture_collection,
    init_local,
    init_remote,
    initialize_fixture_collection,
)


@pytest.fixture
def local_client():
    client: QdrantClient = init_local()
    initialize_fixture_collection(client)
    yield client
    delete_fixture_collection(client)


@pytest.fixture
def remote_client():
    client: QdrantClient = init_remote()
    initialize_fixture_collection(client)
    yield client
    delete_fixture_collection(client)
