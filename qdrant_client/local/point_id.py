import re
from uuid import UUID

from qdrant_client.conversions import common_types as types
from qdrant_client.http import models


_UUID_HYPHENATED = r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"
_UUID_PATTERN = re.compile(
    rf"(?:[0-9a-fA-F]{{32}}|{_UUID_HYPHENATED}|\{{{_UUID_HYPHENATED}\}}|urn:uuid:{_UUID_HYPHENATED})"
)


def normalize_point_id(
    point_id: types.PointId, *, validate: bool = False
) -> models.ExtendedPointId:
    """Use UUID identity rather than spelling, while retaining integer point IDs."""
    if isinstance(point_id, str):
        try:
            # Match the server's simple, hyphenated, braced and URN forms.
            # UUID() alone also accepts misplaced hyphens and malformed wrappers.
            if _UUID_PATTERN.fullmatch(point_id) is None:
                raise ValueError("Invalid UUID spelling")
            return str(UUID(point_id))
        except ValueError as exc:
            if validate:
                raise ValueError(f"Point id {point_id} is not a valid UUID") from exc
            # Lookups (including group payload IDs) historically treat these as missing.
            return point_id
    if isinstance(point_id, UUID):
        return str(point_id)
    if isinstance(point_id, int):
        return point_id
    raise TypeError(f"Incompatible point id type: {type(point_id)}")
