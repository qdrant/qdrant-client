"""Small helpers whose semantics have to match the Rust implementation in core."""

import re
from functools import lru_cache
from uuid import UUID

from qdrant_client.conversions import common_types as types
from qdrant_client.http import models


_UUID_HYPHENATED = r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"
_UUID_PATTERN = re.compile(
    rf"(?:[0-9a-fA-F]{{32}}|{_UUID_HYPHENATED}|\{{{_UUID_HYPHENATED}\}}|urn:uuid:{_UUID_HYPHENATED})"
)
_CANONICAL_UUID = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")


def last_argmax(values: list[float]) -> int:
    """Index of the maximum value, resolving ties in favour of the *last* maximum.

    Mirrors Rust's `Iterator::max_by_key`, which core uses to pick MMR candidates.
    `np.argmax` returns the first maximum instead, which orders exact ties differently.
    """
    best_index = 0
    for index in range(1, len(values)):
        if values[index] >= values[best_index]:
            best_index = index
    return best_index


def swap_remove(items: list[int], position: int) -> int:
    """Remove and return `items[position]`, moving the last item into the freed slot.

    Mirrors `IndexSet::swap_remove`, which core uses to drop a selected MMR candidate,
    and which therefore decides the order the remaining candidates are visited in.
    """
    value = items[position]
    items[position] = items[-1]
    items.pop()
    return value


def normalize_point_id(
    point_id: types.PointId, *, validate: bool = False
) -> models.ExtendedPointId:
    """Use UUID identity rather than spelling, while retaining integer point IDs."""
    if isinstance(point_id, str):
        # fast path, stored ids are already canonical
        if _CANONICAL_UUID.fullmatch(point_id) is not None:
            return point_id
        try:
            # only the forms the server accepts, UUID() alone is more lenient
            if _UUID_PATTERN.fullmatch(point_id) is None:
                raise ValueError("Invalid UUID spelling")
            return str(UUID(point_id))
        except ValueError as exc:
            if validate:
                raise ValueError(f"Point id {point_id} is not a valid UUID") from exc
            # lookups treat invalid ids as missing
            return point_id
    if isinstance(point_id, UUID):
        return str(point_id)
    if isinstance(point_id, int):
        return point_id
    raise TypeError(f"Incompatible point id type: {type(point_id)}")


@lru_cache(maxsize=32)
def normalize_point_ids(
    point_ids: tuple[types.PointId, ...],
) -> frozenset[models.ExtendedPointId]:
    """Normalize `HasIdCondition` ids once per filter rather than once per point."""
    return frozenset(normalize_point_id(point_id) for point_id in point_ids)
