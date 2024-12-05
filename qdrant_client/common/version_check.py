import importlib.metadata
import logging
from typing import Union, Any
from collections import namedtuple

from qdrant_client.http import SyncApis, ApiClient
from qdrant_client.http.models import models

Version = namedtuple("Version", ["major", "minor", "rest"])


def is_server_version_compatible(rest_uri: str, **kwargs: Any) -> bool:
    def get_server_info() -> Any:
        openapi_client: SyncApis[ApiClient] = SyncApis(
            host=rest_uri,
            **kwargs,
        )
        return openapi_client.client.request(
            type_=models.VersionInfo,
            method="GET",
            url="/",
            headers=None,
        )

    def get_server_version() -> Union[str, None]:
        try:
            version_info = get_server_info()
        except Exception as er:
            logging.warning(f"Unable to get server version: {er}, default to None")
            return None

        if not version_info:
            return None
        return version_info.version

    client_version = importlib.metadata.version("qdrant-client")
    server_version = get_server_version()
    return compare_versions(client_version, server_version)


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


def compare_versions(client_version: Union[str, None], server_version: Union[str, None]) -> bool:
    if not client_version or not server_version:
        logging.warning(f"Unable to compare: {client_version} vs {server_version}")
        return False

    if client_version == server_version:
        return True

    try:
        parsed_server_version = parse_version(server_version)
        parsed_client_version = parse_version(client_version)
    except ValueError as er:
        logging.warning(f"Unable to parse version: {er}")
        return False
    major_dif = abs(parsed_server_version.major - parsed_client_version.major)
    if major_dif >= 1:
        return False
    elif major_dif == 0:
        return abs(parsed_server_version.minor - parsed_client_version.minor) <= 1
    return False
