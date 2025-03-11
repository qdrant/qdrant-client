import uuid
import pytest

from qdrant_client import QdrantClient
from tests.congruence_tests.test_common import (
    delete_fixture_collection,
    init_local,
    init_remote,
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
    yield client
    delete_fixture_collection(client)


@pytest.fixture(scope="session")
def remote_client(prefer_grpc: bool) -> QdrantClient:
    client: QdrantClient = init_remote(prefer_grpc=prefer_grpc)
    yield client
    delete_fixture_collection(client)


@pytest.fixture(scope="session")
def collection_name() -> str:
    return f"collection_1_{uuid.uuid4().hex}"


@pytest.fixture(scope="session")
def secondary_collection_name() -> str:
    return f"collection_2_{uuid.uuid4().hex}"
