import pytest

from qdrant_client import QdrantClient
from tests.congruence_tests.test_common import (
    delete_fixture_collection,
    init_local,
    init_remote,
    initialize_fixture_collection,
)


@pytest.fixture
def local_client() -> QdrantClient:
    client: QdrantClient = init_local()
    initialize_fixture_collection(client)
    yield client
    delete_fixture_collection(client)


@pytest.fixture
def remote_client() -> QdrantClient:
    client: QdrantClient = init_remote()
    initialize_fixture_collection(client)
    yield client
    delete_fixture_collection(client)
