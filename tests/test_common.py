import pytest

from qdrant_client.common.version_check import is_versions_compatible, parse_version


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
def test_check_versions(client_version, server_version, expected_result):
    assert (
        is_versions_compatible(client_version=client_version, server_version=server_version)
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
