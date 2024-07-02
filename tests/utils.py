import os
from typing import Tuple, Optional


def read_version() -> Tuple[Optional[int], Optional[int], Optional[int], bool]:
    """Read Qdrant's version from env and parse

    Returns:
        Tuple[int, int, int, bool] - major, minor, patch, dev
    """
    version = os.getenv("QDRANT_VERSION")
    if version == "dev":
        return None, None, None, True

    if version is not None:
        semver = version.replace("v", "").split(".")

        return int(semver[0]), int(semver[1]), int(semver[2]), False
    else:
        return None, None, None, False
