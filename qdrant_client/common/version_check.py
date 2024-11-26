import asyncio
import importlib.metadata
import inspect
from typing import Union, TYPE_CHECKING

from packaging import version
from packaging.version import Version

if TYPE_CHECKING:
    from qdrant_client.qdrant_remote import QdrantRemote
    from qdrant_client.async_qdrant_remote import AsyncQdrantRemote


def is_server_version_compatible(client: Union["QdrantRemote", "AsyncQdrantRemote"]) -> bool:
    client_version = version.parse(importlib.metadata.version("qdrant-client"))

    get_info = client.info()
    if inspect.iscoroutine(get_info):
        loop = asyncio.get_event_loop()
        info_version = loop.run_until_complete(get_info).version
    elif hasattr(get_info, "version"):
        info_version = get_info.version
    else:
        raise ValueError("Unable to retrieve server version")
    server_version = version.parse(info_version)

    return check_version(client_version, server_version)


def check_version(client_version: Version, server_version: Version) -> bool:
    if client_version == server_version:
        return True
    major_dif = abs(server_version.major - client_version.major)
    if major_dif >= 1:
        return False
    elif major_dif == 0:
        return abs(server_version.minor - client_version.minor) <= 1
    return False
