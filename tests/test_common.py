import pytest

from qdrant_client.async_qdrant_remote import AsyncQdrantRemote
from qdrant_client.common.version_check import get_server_version, is_compatible, parse_version
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


class _FakeResponse:
    status_code = 200

    @staticmethod
    def json():
        return {"version": "1.2.0"}


@pytest.mark.parametrize(
    "verify",
    ["custom_ca.pem", False, True],
    ids=["Custom CA path", "Verification disabled", "Verification enabled"],
)
def test_get_server_version_forwards_verify(monkeypatch, verify):
    captured = {}

    def fake_get(url, **kwargs):
        captured.update(kwargs)
        return _FakeResponse()

    monkeypatch.setattr("httpx.get", fake_get)

    version = get_server_version("https://localhost:6333/", {}, None, 5, verify=verify)

    assert version == "1.2.0"
    assert captured.get("verify") is verify


def test_get_server_version_default_verify(monkeypatch):
    captured = {}

    def fake_get(url, **kwargs):
        captured.update(kwargs)
        return _FakeResponse()

    monkeypatch.setattr("httpx.get", fake_get)

    get_server_version("https://localhost:6333/", {}, None, 5)

    assert captured.get("verify") is True


def test_sync_check_compatibility_forwards_verify(monkeypatch):
    captured = {}

    def fake_get_server_version(rest_uri, rest_headers, auth_provider, timeout, verify=True):
        captured["verify"] = verify
        return "1.2.0"

    monkeypatch.setattr("qdrant_client.qdrant_remote.get_server_version", fake_get_server_version)

    QdrantRemote._check_compatibility(
        "https://localhost:6333/", {}, None, 5, verify="custom_ca.pem"
    )

    assert captured["verify"] == "custom_ca.pem"


def test_async_check_compatibility_forwards_verify(monkeypatch):
    captured = {}

    def fake_get_server_version(rest_uri, rest_headers, auth_provider, timeout, verify=True):
        captured["verify"] = verify
        return "1.2.0"

    monkeypatch.setattr(
        "qdrant_client.async_qdrant_remote.get_server_version", fake_get_server_version
    )

    AsyncQdrantRemote._check_compatibility(
        "https://localhost:6333/", {}, None, 5, verify="custom_ca.pem"
    )

    assert captured["verify"] == "custom_ca.pem"


def test_qdrant_remote_init_passes_verify_to_version_check(monkeypatch):
    captured = {}

    def fake_get_server_version(rest_uri, rest_headers, auth_provider, timeout, verify=True):
        captured["verify"] = verify
        import importlib

        return importlib.metadata.version("qdrant-client")

    class ImmediateThread:
        def __init__(self, target=None, args=(), kwargs=None, daemon=None):
            target(*args, **(kwargs or {}))

        def start(self):
            pass

    monkeypatch.setattr("qdrant_client.qdrant_remote.get_server_version", fake_get_server_version)
    monkeypatch.setattr("qdrant_client.qdrant_remote.Thread", ImmediateThread)

    QdrantRemote(url="http://localhost:6333", verify=False)

    assert captured["verify"] is False

    captured.clear()
    QdrantRemote(url="http://localhost:6333")

    assert captured["verify"] is True
