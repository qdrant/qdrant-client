import pytest
from packaging.version import Version

from qdrant_client.common.version_check import check_version


@pytest.mark.parametrize(
    "test_data",
    [
        ("1.9.3.dev0", "2.0.1", False, "Diff between major versions = 1, minor versions differ"),
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
    ],
)
def test_check_versions(test_data):
    assert (
        check_version(client_version=Version(test_data[0]), server_version=Version(test_data[1]))
        is test_data[2]
    )
