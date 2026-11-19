from unittest.mock import MagicMock, patch

import pytest

from qdrant_client.async_qdrant_remote import AsyncQdrantRemote
from qdrant_client.connection import format_host, get_async_channel, get_channel
from qdrant_client.qdrant_remote import QdrantRemote


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("localhost", "localhost"),
        ("127.0.0.1", "127.0.0.1"),
        ("fd00::1", "[fd00::1]"),
        ("::1", "[::1]"),
        ("[fd00::1]", "[fd00::1]"),
        ("fe80::1%eth0", "[fe80::1%eth0]"),
    ],
)
def test_format_host(raw: str, expected: str) -> None:
    assert format_host(raw) == expected


@pytest.mark.parametrize("client_cls", [QdrantRemote, AsyncQdrantRemote])
def test_ipv6_rest_uri_from_url_and_host(client_cls) -> None:
    client_from_url = client_cls(url="http://[fd00::1]:6333", check_compatibility=False)
    assert client_from_url.rest_uri == "http://[fd00::1]:6333"

    client_from_host = client_cls(host="fd00::1", port=6333, check_compatibility=False)
    assert client_from_host.rest_uri == "http://[fd00::1]:6333"

    client_pre_bracketed = client_cls(host="[fd00::1]", port=6333, check_compatibility=False)
    assert client_pre_bracketed.rest_uri == "http://[fd00::1]:6333"

    client_no_port = client_cls(host="fd00::1", port=None, check_compatibility=False)
    assert client_no_port.rest_uri == "http://[fd00::1]"


def test_ipv6_grpc_channel_targets() -> None:
    with (
        patch("qdrant_client.connection.grpc.insecure_channel", return_value=MagicMock()) as mock_insecure,
        patch("qdrant_client.connection.grpc.secure_channel", return_value=MagicMock()) as mock_secure,
        patch("qdrant_client.connection.grpc.intercept_channel", side_effect=lambda ch, *_: ch),
    ):
        get_channel(host="fd00::1", port=6334, ssl=False)
        assert mock_insecure.call_args.args[0] == "[fd00::1]:6334"

        get_channel(host="[fd00::1]", port=6334, ssl=True)
        assert mock_secure.call_args.args[0] == "[fd00::1]:6334"

    with (
        patch("qdrant_client.connection.grpc.aio.insecure_channel", return_value=MagicMock()) as mock_aio_insecure,
        patch("qdrant_client.connection.grpc.aio.secure_channel", return_value=MagicMock()) as mock_aio_secure,
    ):
        get_async_channel(host="fd00::1", port=6334, ssl=False)
        assert mock_aio_insecure.call_args.args[0] == "[fd00::1]:6334"

        get_async_channel(host="[fd00::1]", port=6334, ssl=True)
        assert mock_aio_secure.call_args.args[0] == "[fd00::1]:6334"
