import pytest

from qdrant_client import QdrantClient
from tests.congruence_tests.test_common import (
    delete_fixture_collection,
    init_local,
    init_remote,
    initialize_fixture_collection,
)


@pytest.fixture(scope="session")
def storage() -> str:
    return ":memory:"


@pytest.fixture(scope="session")
def prefer_grpc() -> bool:
    return False


@pytest.fixture(scope="session")
def local_client(storage: str) -> QdrantClient:
    client: QdrantClient = init_local(storage=storage)
    # initialize_fixture_collection(client)
    yield client
    delete_fixture_collection(client)


@pytest.fixture(scope="session")
def remote_client(prefer_grpc: bool) -> QdrantClient:
    client: QdrantClient = init_remote(prefer_grpc=prefer_grpc)
    # initialize_fixture_collection(client)
    yield client
    delete_fixture_collection(client)
