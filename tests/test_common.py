import unittest.mock

import pytest

from qdrant_client.async_qdrant_remote import AsyncQdrantRemote
from qdrant_client.common.version_check import (
    get_server_version,
    is_compatible,
    parse_version,
)
from qdrant_client.qdrant_remote import QdrantRemote


@pytest.mark.parametrize(
    "client_version, server_version, expected_result",
    [
        ("1.9.3.dev0", "2.8.1.dev12-something", False),
        ("1.9", "2.8", False),
        ("1", "2", False),
        ("1.9.0", "2.9.0", False),
        ("1.1.0", "1.2.9", True),
        ("1.2.7", "1.1.8.dev0", True),
        ("1.2.1", "1.2.29", True),
        ("1.2.0", "1.2.0", True),
        ("1.2.0", "1.4.0", False),
        ("1.4.0", "1.2.0", False),
        ("1.9.0", "3.7.0", False),
        ("3.0.0", "1.0.0", False),
        (None, "1.0.0", False),
        ("1.0.0", None, False),
        (None, None, False),
    ],
    ids=[
        "Diff between major versions = 1, negative",
        "Diff between major versions = 1, only major and minor, negative",
        "Diff between major versions = 1, only major, negative",
        "Diff between major versions = 1, minor versions are the same, negative",
        "Diff between major versions == 0, diff between minor versions == 1 (server > client), positive",
        "Diff between major versions == 0, diff between minor versions == 1 (client > server), positive",
        "Diff between major versions == 0, diff between minor versions == 0, positive",
        "Same versions, positive",
        "Diff between major versions == 0, diff between minor versions > 1 (server > client), negative",
        "Diff between major versions == 0, diff between minor versions > 1 (client > server), negative",
        "Diff between major versions > 1 (server > client), negative",
        "Diff between major versions > 1 (client > server), negative",
        "Client version is None, negative",
        "Server version is None, negative",
        "Both versions are None, negative",
    ],
)
def test_check_compatibility(client_version, server_version, expected_result):
    assert (
        is_compatible(client_version=client_version, server_version=server_version)
        is expected_result
    )


@pytest.mark.parametrize(
    "input_version",
    ["1", "1.", ".1", ".1.", "1.None.1", "None.0.1", None, ""],
    ids=[
        "Only major part",
        "Only major part with dot",
        "Only minor part",
        "Only minor part with dot",
        "Minor part is not a number",
        "Major part is not a number",
        "Version is None",
        "Version is empty",
    ],
)
def test_parse_versions_value_error(input_version):
    with pytest.raises(ValueError):
        parse_version(input_version)


def test_get_server_version_verify(monkeypatch):
    captured_kwargs = {}

    def mock_get(url, **kwargs):
        captured_kwargs.update(kwargs)
        mock_response = unittest.mock.MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"version": "1.12.0"}
        return mock_response

    monkeypatch.setattr("httpx.get", mock_get)

    # 1. verify not provided -> kwargs should not contain verify
    captured_kwargs.clear()
    ver = get_server_version("http://localhost:6333", {}, None, 5)
    assert ver == "1.12.0"
    assert "verify" not in captured_kwargs

    # 2. verify=False
    captured_kwargs.clear()
    ver = get_server_version("https://localhost:6333", {}, None, 5, verify=False)
    assert ver == "1.12.0"
    assert captured_kwargs.get("verify") is False

    # 3. verify="/custom/ca.crt"
    captured_kwargs.clear()
    ver = get_server_version("https://localhost:6333", {}, None, 5, verify="/custom/ca.crt")
    assert ver == "1.12.0"
    assert captured_kwargs.get("verify") == "/custom/ca.crt"


def test_check_compatibility_passes_verify(monkeypatch):
    called_with = {}

    def mock_get_server_version(rest_uri, rest_headers, auth, timeout, verify=None):
        called_with["verify"] = verify
        return "1.20.0"

    monkeypatch.setattr("qdrant_client.qdrant_remote.get_server_version", mock_get_server_version)
    monkeypatch.setattr(
        "qdrant_client.async_qdrant_remote.get_server_version", mock_get_server_version
    )

    QdrantRemote._check_compatibility(
        "https://localhost:6333", {}, None, 5, verify="/path/to/custom_ca.crt"
    )
    assert called_with.get("verify") == "/path/to/custom_ca.crt"

    called_with.clear()
    AsyncQdrantRemote._check_compatibility(
        "https://localhost:6333", {}, None, 5, verify="/path/to/custom_ca.crt"
    )
    assert called_with.get("verify") == "/path/to/custom_ca.crt"
