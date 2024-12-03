import asyncio
import importlib.metadata
import inspect
import logging
from typing import Union, TYPE_CHECKING
from collections import namedtuple


Version = namedtuple("Version", ["major", "minor", "rest"])


if TYPE_CHECKING:
    from qdrant_client.qdrant_remote import QdrantRemote
    from qdrant_client.async_qdrant_remote import AsyncQdrantRemote


def is_server_version_compatible(client: Union["QdrantRemote", "AsyncQdrantRemote"]) -> bool:
    client_version = importlib.metadata.version("qdrant-client")

    get_info = client.info()
    if inspect.iscoroutine(get_info):
        loop = asyncio.get_event_loop()
        info_version = loop.run_until_complete(get_info).version
    elif hasattr(get_info, "version"):
        info_version = get_info.version
    else:
        raise ValueError("Unable to retrieve server version")
    server_version = info_version

    return check_version(client_version, server_version)


def parse_version(version: str) -> Version:
    try:
        major, minor, *rest = version.split(".")
        return Version(int(major), int(minor), rest)
    except ValueError as er:
        raise ValueError(
            f"Unable to parse version, expected format: x.y.z, found: {version}"
        ) from er


def check_version(client_version: str, server_version: str) -> bool:
    if client_version == server_version:
        return True

    try:
        server_version = parse_version(server_version)
        client_version = parse_version(client_version)
    except ValueError as er:
        logging.warning(f"Unable to parse version: {er}")
        return False
    major_dif = abs(server_version.major - client_version.major)
    if major_dif >= 1:
        return False
    elif major_dif == 0:
        return abs(server_version.minor - client_version.minor) <= 1
    return False
