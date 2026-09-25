from uuid import UUID

from qdrant_client.conversions import common_types as types
from qdrant_client.http import models


def normalize_point_id(
    point_id: types.PointId, *, validate: bool = False
) -> models.ExtendedPointId:
    """Use UUID identity rather than spelling, while retaining integer point IDs."""
    if isinstance(point_id, str):
        try:
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
