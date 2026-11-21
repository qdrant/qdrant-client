import pytest

from qdrant_client.common.version_check import is_compatible, parse_version


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


@pytest.mark.parametrize("verify_arg", [False, "/path/to/custom_ca.crt", True])
def test_get_server_version_forwards_verify(monkeypatch, verify_arg):
    import httpx
    from qdrant_client.common.version_check import get_server_version

    captured: dict[str, object] = {}

    def fake_get(url, *, headers=None, auth=None, timeout=None, verify=True):
        captured["verify"] = verify
        return httpx.Response(200, json={"version": "1.14.0"})

    monkeypatch.setattr(httpx, "get", fake_get)
    version = get_server_version("https://localhost:6333", {}, None, 5, verify=verify_arg)
    assert version == "1.14.0"
    assert captured["verify"] == verify_arg


def test_get_server_version_default_verify_true(monkeypatch):
    import httpx
    from qdrant_client.common.version_check import get_server_version

    captured: dict[str, object] = {}

    def fake_get(url, *, headers=None, auth=None, timeout=None, verify=True):
        captured["verify"] = verify
        return httpx.Response(200, json={"version": "1.14.0"})

    monkeypatch.setattr(httpx, "get", fake_get)
    version = get_server_version("https://localhost:6333", {}, None, 5)
    assert version == "1.14.0"
    assert captured["verify"] is True


def test_remote_clients_pass_verify_to_check_compatibility(monkeypatch):
    import ssl
    from qdrant_client.async_qdrant_remote import AsyncQdrantRemote
    from qdrant_client.qdrant_remote import QdrantRemote

    captured: list[tuple[str, object]] = []

    def fake_check(rest_uri, rest_headers, auth, timeout, verify=True):
        captured.append((rest_uri, verify))

    class ImmediateThread:
        def __init__(self, target, args=(), daemon=True):
            self._target = target
            self._args = args

        def start(self):
            self._target(*self._args)

    monkeypatch.setattr("importlib.metadata.version", lambda _pkg: "1.20.0")
    monkeypatch.setattr(QdrantRemote, "_check_compatibility", staticmethod(fake_check))
    monkeypatch.setattr(AsyncQdrantRemote, "_check_compatibility", staticmethod(fake_check))
    monkeypatch.setattr("qdrant_client.qdrant_remote.Thread", ImmediateThread)
    monkeypatch.setattr("qdrant_client.async_qdrant_remote.Thread", ImmediateThread)

    custom_ctx = ssl.create_default_context()
    QdrantRemote(url="https://localhost:6333", verify=custom_ctx, check_compatibility=True)
    AsyncQdrantRemote(url="https://localhost:6333", verify=False, check_compatibility=True)
    QdrantRemote(url="https://localhost:6333", check_compatibility=True)

    assert [v for _, v in captured] == [custom_ctx, False, True]
