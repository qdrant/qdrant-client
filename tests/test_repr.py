import pytest

from qdrant_client import AsyncQdrantClient, QdrantClient
from qdrant_client.async_qdrant_remote import AsyncQdrantRemote
from qdrant_client.local.async_qdrant_local import AsyncQdrantLocal
from qdrant_client.local.qdrant_local import QdrantLocal
from qdrant_client.qdrant_remote import QdrantRemote


def test_local_client_repr_shows_location():
    assert repr(QdrantClient(":memory:")) == "<QdrantClient mode=local location=':memory:'>"


def test_async_local_client_repr_shows_location():
    assert (
        repr(AsyncQdrantClient(":memory:")) == "<AsyncQdrantClient mode=local location=':memory:'>"
    )


def test_persistent_local_client_repr_shows_path(tmp_path):
    path = str(tmp_path / "storage")
    assert repr(QdrantClient(path=path)) == f"<QdrantClient mode=local location={path!r}>"


def test_remote_client_repr_shows_host_and_grpc_preference():
    client = QdrantClient("localhost", port=6333, prefer_grpc=True)
    assert repr(client) == "<QdrantClient mode=remote host='localhost:6333' prefer_grpc=True>"


def test_async_remote_client_repr_shows_host_and_grpc_preference():
    client = AsyncQdrantClient("localhost", port=6333, prefer_grpc=True)
    assert repr(client) == "<AsyncQdrantClient mode=remote host='localhost:6333' prefer_grpc=True>"


def test_remote_repr_shows_scheme():
    remote = QdrantRemote(url="https://api.qdrant.example:443")
    assert (
        repr(remote)
        == "<QdrantRemote scheme=https host='api.qdrant.example:443' prefer_grpc=False>"
    )


def test_async_remote_repr_shows_scheme():
    remote = AsyncQdrantRemote(url="https://api.qdrant.example:443")
    assert (
        repr(remote)
        == "<AsyncQdrantRemote scheme=https host='api.qdrant.example:443' prefer_grpc=False>"
    )


def test_local_repr():
    assert repr(QdrantLocal(":memory:")) == "<QdrantLocal location=':memory:'>"


def test_async_local_repr():
    assert repr(AsyncQdrantLocal(":memory:")) == "<AsyncQdrantLocal location=':memory:'>"


@pytest.mark.parametrize(
    "client",
    [
        QdrantClient("localhost", port=6333, api_key="super-secret-key"),
        QdrantClient(url="https://api.qdrant.example:443", api_key="super-secret-key"),
    ],
)
def test_api_key_never_appears_in_repr(client):
    assert "super-secret-key" not in repr(client)
    assert "super-secret-key" not in repr(client._client)
