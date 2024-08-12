import os
from typing import Optional, Tuple


def read_version() -> Tuple[Optional[int], Optional[int], Optional[int], bool]:
    """Read Qdrant's version from env and parse it into a tuple

    Returns:
        Tuple[Optional[int], Optional[int], Optional[int], bool] - A tuple of (major, minor, patch, dev), where `dev` is a boolean indicating
        if it's a development version. If the version is not set or is "dev", returns (None, None, None, True)
    """
    version_str = os.getenv("QDRANT_VERSION")

    if version_str is None:
        return None, None, None, False
    if version_str == "dev":
        return None, None, None, True

    semver = version_str.replace("v", "").split(".")
    major, minor, patch = int(semver[0]), int(semver[1]), int(semver[2])
    return major, minor, patch, False
