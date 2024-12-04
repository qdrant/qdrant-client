import pytest

from qdrant_client.common.version_check import compare_versions, parse_version


@pytest.mark.parametrize(
    "test_data",
    [
        ("1.9.3.dev0", "2.0.1", False, "Diff between major versions = 1, minor versions differ"),
        (
            "1.9",
            "2.0",
            False,
            "Diff between major versions = 1, minor versions differ, only major and patch",
        ),
        ("1", "2", False, "Diff between major versions = 1, minor versions differ, only major"),
        ("1.9.0", "2.9.0", False, "Diff between major versions = 1, minor versions are the same"),
        (
            "1.1.0",
            "1.2.9",
            True,
            "Diff between major versions == 0, diff between minor versions == 1 (server > client)",
        ),
        (
            "1.2.7",
            "1.1.8.dev0",
            True,
            "Diff between major versions == 0, diff between minor versions == 1 (client > server)",
        ),
        (
            "1.2.1",
            "1.2.29",
            True,
            "Diff between major versions == 0, diff between minor versions == 0",
        ),
        ("1.2.0", "1.2.0", True, "Same versions"),
        (
            "1.2.0",
            "1.4.0",
            False,
            "Diff between major versions == 0, diff between minor versions > 1 (server > client)",
        ),
        (
            "1.4.0",
            "1.2.0",
            False,
            "Diff between major versions == 0, diff between minor versions > 1 (client > server)",
        ),
        ("1.9.0", "3.7.0", False, "Diff between major versions > 1 (server > client)"),
        ("3.0.0", "1.0.0", False, "Diff between major versions > 1 (client > server)"),
        (None, "1.0.0", False, "Client version is None"),
        ("1.0.0", None, False, "Server version is None"),
        (None, None, False, "Both versions are None"),
    ],
)
def test_check_versions(test_data):
    assert (
        compare_versions(client_version=test_data[0], server_version=test_data[1]) is test_data[2]
    )


@pytest.mark.parametrize(
    "test_data",
    [
        ("1", "Only major version"),
        ("1.", "Only major version"),
        (".1", "Only minor version"),
        (".1.", "Only minor version"),
        ("1.None.1", "Minor version is not a number"),
        ("None.0.1", "Major version is not a number"),
        (None, "Version is None"),
        ("", "Version is empty"),
    ],
)
def test_parse_versions_value_error(test_data):
    with pytest.raises(ValueError):
        parse_version(test_data[0])
