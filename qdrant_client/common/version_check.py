import warnings
from typing import Union, Dict, Any
from collections import namedtuple

import httpx
from pydantic import ValidationError

from qdrant_client.http.api_client import parse_as_type
from qdrant_client.http.models import models

Version = namedtuple("Version", ["major", "minor", "rest"])


def get_server_version(rest_uri: str, rest_headers: Dict[str, Any]) -> Union[str, None]:
    try:
        response = httpx.get(rest_uri + "/", headers=rest_headers)
    except Exception as er:
        warnings.warn(f"Unable to get server version: {er}, default to None")
        return None

    if response.status_code in [200, 201, 202]:
        try:
            version_info = parse_as_type(response.json(), models.VersionInfo)
            return version_info.version
        except ValidationError as e:
            warnings.warn(f"Unable to parse response from server: {response}, default to None")
    else:
        warnings.warn(f"Unexpected response from server: {response}, default to None")
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
