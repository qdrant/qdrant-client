import logging
import warnings
from typing import Union
from collections import namedtuple

from qdrant_client.http import SyncApis, ApiClient

Version = namedtuple("Version", ["major", "minor", "rest"])


def get_server_version(rest_uri: str) -> Union[str, None]:
    try:
        openapi_client: SyncApis[ApiClient] = SyncApis(host=rest_uri)
        version_info = openapi_client.service_api.root()

        try:
            openapi_client.close()
        except Exception:
            logging.warning(
                "Unable to close http connection. Connection was interrupted on the server side"
            )

        return version_info.version
    except Exception as er:
        warnings.warn(f"Unable to get server version: {er}, default to None")
        return None


def parse_version(version: str) -> Version:
    if not version:
        raise ValueError("Version is None")
    try:
        major, minor, *rest = version.split(".")
        return Version(int(major), int(minor), rest)
    except ValueError as er:
        raise ValueError(
            f"Unable to parse version, expected format: x.y.z, found: {version}"
        ) from er


def is_versions_compatible(
    client_version: Union[str, None], server_version: Union[str, None]
) -> bool:
    if not client_version:
        warnings.warn(f"Unable to compare with client version {client_version}")
        return False

    if not server_version:
        warnings.warn(f"Unable to compare with server version {server_version}")
        return False

    if client_version == server_version:
        return True

    try:
        parsed_server_version = parse_version(server_version)
        parsed_client_version = parse_version(client_version)
    except ValueError as er:
        warnings.warn(f"Unable to compare versions: {er}")
        return False

    major_dif = abs(parsed_server_version.major - parsed_client_version.major)
    if major_dif >= 1:
        return False
    return abs(parsed_server_version.minor - parsed_client_version.minor) <= 1
