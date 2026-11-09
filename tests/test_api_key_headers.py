import pytest

from qdrant_client import AsyncQdrantClient, QdrantClient

CLIENTS = [QdrantClient, AsyncQdrantClient]


def make_client(client_type, **kwargs):
    # check_compatibility=False keeps the client from talking to a server, so the
    # metadata it would send can be inspected offline.
    return client_type(
        host="localhost", prefer_grpc=True, check_compatibility=False, **kwargs
    )


@pytest.mark.parametrize("client_type", CLIENTS)
def test_api_key_replaces_header_in_grpc_metadata(client_type) -> None:
    client = make_client(
        client_type,
        https=True,
        api_key="from-param",
        headers={"api-key": "from-headers", "x-custom": "kept"},
    )

    assert client._client._grpc_headers == [
        ("x-custom", "kept"),
        ("api-key", "from-param"),
    ]
    assert client._client._rest_headers["api-key"] == "from-param"


@pytest.mark.parametrize("client_type", CLIENTS)
def test_api_key_replaces_header_with_any_casing(client_type) -> None:
    client = make_client(
        client_type,
        https=True,
        api_key="from-param",
        headers={"Api-Key": "from-headers"},
    )

    assert client._client._grpc_headers == [("api-key", "from-param")]


@pytest.mark.parametrize("client_type", CLIENTS)
def test_api_key_is_sent_once_without_headers(client_type) -> None:
    client = make_client(client_type, https=True, api_key="from-param")

    assert client._client._grpc_headers == [("api-key", "from-param")]


@pytest.mark.parametrize("client_type", CLIENTS)
def test_header_is_kept_when_no_api_key_is_given(client_type) -> None:
    client = make_client(client_type, https=True, headers={"api-key": "from-headers"})

    assert client._client._grpc_headers == [("api-key", "from-headers")]
