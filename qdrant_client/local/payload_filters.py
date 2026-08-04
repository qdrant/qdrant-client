from datetime import date, datetime, timezone
from typing import Any
from uuid import UUID

import numpy as np

from qdrant_client.http import models
from qdrant_client.local import datetime_utils
from qdrant_client.local.geo import boolean_point_in_polygon, geo_distance
from qdrant_client.local.payload_value_extractor import value_by_key
from qdrant_client.local.siphash import point_id_slice
from qdrant_client.conversions import common_types as types


def get_value_counts(values: list[Any]) -> list[int]:
    counts = []

    if all(value is None for value in values):
        counts.append(0)
    else:
        for value in values:
            if value is None:
                counts.append(0)
            elif isinstance(value, list):
                counts.append(len(value))
            else:
                counts.append(1)
    return counts


def check_values_count(condition: models.ValuesCount, values: list[Any] | None) -> bool:
    if values is None:
        values = []

    counts = get_value_counts(values)

    # A single value's count must satisfy every bound at once, and the condition
    # matches if any one value does. Checking each bound independently across all
    # counts would let separate values satisfy separate bounds.
    return any(
        (condition.lt is None or count < condition.lt)
        and (condition.lte is None or count <= condition.lte)
        and (condition.gt is None or count > condition.gt)
        and (condition.gte is None or count >= condition.gte)
        for count in counts
    )


def check_geo_radius(condition: models.GeoRadius, values: Any) -> bool:
    if isinstance(values, dict) and "lat" in values and "lon" in values:
        lat = values["lat"]
        lon = values["lon"]

        distance = geo_distance(
            lon1=lon,
            lat1=lat,
            lon2=condition.center.lon,
            lat2=condition.center.lat,
        )

        return distance < condition.radius

    return False


def check_geo_bounding_box(condition: models.GeoBoundingBox, values: Any) -> bool:
    if isinstance(values, dict) and "lat" in values and "lon" in values:
        lat = values["lat"]
        lon = values["lon"]

        # handle anti-meridian crossing case
        if condition.top_left.lon > condition.bottom_right.lon:
            longitude_condition = lon > condition.top_left.lon or lon < condition.bottom_right.lon
        else:
            longitude_condition = condition.top_left.lon < lon < condition.bottom_right.lon

        latitude_condition = condition.top_left.lat > lat > condition.bottom_right.lat

        return longitude_condition and latitude_condition

    return False


def check_geo_polygon(condition: models.GeoPolygon, values: Any) -> bool:
    if isinstance(values, dict) and "lat" in values and "lon" in values:
        lat = values["lat"]
        lon = values["lon"]
        exterior = [(point.lat, point.lon) for point in condition.exterior.points]
        interiors = []
        if condition.interiors is not None:
            interiors = [
                [(point.lat, point.lon) for point in interior.points]
                for interior in condition.interiors
            ]
        return boolean_point_in_polygon(point=(lat, lon), exterior=exterior, interiors=interiors)

    return False


def check_range_interface(condition: models.RangeInterface, value: Any) -> bool:
    if isinstance(condition, models.Range):
        return check_range(condition, value)
    if isinstance(condition, models.DatetimeRange):
        return check_datetime_range(condition, value)
    return False


def check_range(condition: models.Range, value: Any) -> bool:
    # bool is a subclass of int in Python, but booleans are not numeric values
    # in Qdrant and must not be matched by a range condition.
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return (
        (condition.lt is None or value < condition.lt)
        and (condition.lte is None or value <= condition.lte)
        and (condition.gt is None or value > condition.gt)
        and (condition.gte is None or value >= condition.gte)
    )


def check_datetime_range(condition: models.DatetimeRange, value: Any) -> bool:
    def make_condition_tz_aware(dt: datetime | date | None) -> datetime | None:
        if isinstance(dt, date) and not isinstance(dt, datetime):
            dt = datetime.combine(dt, datetime.min.time())

        if dt is None or dt.tzinfo is not None:
            return dt

        # Assume UTC if no timezone is provided
        return dt.replace(tzinfo=timezone.utc)

    if not isinstance(value, str):
        return False

    dt = datetime_utils.parse(value)

    if dt is None:
        return False

    lt = make_condition_tz_aware(condition.lt)
    lte = make_condition_tz_aware(condition.lte)
    gt = make_condition_tz_aware(condition.gt)
    gte = make_condition_tz_aware(condition.gte)

    return (
        (lt is None or dt < lt)
        and (lte is None or dt <= lte)
        and (gt is None or dt > gt)
        and (gte is None or dt >= gte)
    )


def values_match(value: Any, other: Any) -> bool:
    # Qdrant keeps bool, integer and float as distinct payload value types for exact
    # match, but Python cross-matches them (bool is a subclass of int, and 1 == 1.0).
    # Match condition operands are only ever bool / int / str (never float), so a float
    # payload value can never be matched by an exact-match condition on the server.
    if isinstance(value, bool) != isinstance(other, bool):
        return False
    if isinstance(value, float) != isinstance(other, float):
        return False
    return value == other


def check_match(condition: models.Match, value: Any) -> bool:
    if isinstance(condition, models.MatchValue):
        return values_match(value, condition.value)
    if isinstance(condition, models.MatchText):
        return isinstance(value, str) and condition.text in value
    if isinstance(condition, models.MatchTextAny):
        return isinstance(value, str) and any(word in value for word in condition.text_any.split())
    if isinstance(condition, models.MatchPhrase):
        # Same approximation as `MatchText` above: on a field without a text index the server
        # falls back to a substring scan, which this reproduces exactly. A phrase-enabled text
        # index makes the server tokenize and lowercase instead, and local mode builds no
        # indexes, so it cannot reproduce that.
        return isinstance(value, str) and condition.phrase in value
    if isinstance(condition, models.MatchPrefix):
        # byte-wise and case-sensitive, like exact keyword matching. Non-string values never
        # match, not even against an empty prefix.
        return isinstance(value, str) and value.startswith(condition.prefix)
    if isinstance(condition, models.MatchAny):
        return any(values_match(value, v) for v in condition.any)
    if isinstance(condition, models.MatchExcept):
        return not any(values_match(value, v) for v in condition.except_)
    raise ValueError(f"Unknown match condition: {condition}")


def nested_filter_values(payload: dict[str, Any], key: str) -> list[Any]:
    """Array elements a nested filter is applied to, one at a time.

    A nested key should point to an array of objects, and may be written with or without the
    bracket notation: `data` and `data[]` are equivalent. A value which is not an array - a plain
    object, a scalar - has no elements to apply the filter to, so it never matches.
    """
    if len(key) > 2 and key.endswith("[]"):
        key = key[: -len("[]")]

    values = value_by_key(payload, key, flat=False)
    if values is None:
        return []

    elements: list[Any] = []
    for value in values:
        if isinstance(value, list):
            elements.extend(value)
    return elements


def check_nested_filter(nested_filter: models.Filter, values: list[Any]) -> bool:
    return any(check_filter(nested_filter, v, point_id=-1, has_vector={}) for v in values)


def check_condition(
    condition: models.Condition,
    payload: dict[str, Any],
    point_id: models.ExtendedPointId,
    has_vector: dict[str, bool],
) -> bool:
    if isinstance(condition, models.IsNullCondition):
        values = value_by_key(payload, condition.is_null.key, flat=False)
        if values is None:
            return False
        if any(v is None for v in values):
            return True
    elif isinstance(condition, models.IsEmptyCondition):
        values = value_by_key(payload, condition.is_empty.key, flat=False)
        if (
            values is None
            or len(values) == 0
            or all((v is None or (isinstance(v, list) and len(v) == 0)) for v in values)
        ):
            return True
    elif isinstance(condition, models.HasIdCondition):
        ids = [str(id_) if isinstance(id_, UUID) else id_ for id_ in condition.has_id]
        if point_id in ids:
            return True
    elif isinstance(condition, models.SliceCondition):
        total, index = condition.slice.total, condition.slice.index
        if total < 1:
            raise ValueError(f"Slice total must be >= 1, got {total}")
        if not 0 <= index < total:
            raise ValueError(f"Slice index must be in 0..{total}, got {index}")
        if isinstance(point_id, int) and point_id < 0:
            # sentinel id used while evaluating nested filters, it belongs to no slice
            return False
        return point_id_slice(point_id, total) == index
    elif isinstance(condition, models.HasVectorCondition):
        if condition.has_vector in has_vector and has_vector[condition.has_vector]:
            return True
    elif isinstance(condition, models.FieldCondition):
        values = value_by_key(payload, condition.key)
        if condition.match is not None:
            if values is None:
                return False
            return any(check_match(condition.match, v) for v in values)
        if condition.range is not None:
            if values is None:
                return False
            return any(check_range_interface(condition.range, v) for v in values)
        if condition.geo_bounding_box is not None:
            if values is None:
                return False
            return any(check_geo_bounding_box(condition.geo_bounding_box, v) for v in values)
        if condition.geo_radius is not None:
            if values is None:
                return False
            return any(check_geo_radius(condition.geo_radius, v) for v in values)
        if condition.values_count is not None:
            values = value_by_key(payload, condition.key, flat=False)
            return check_values_count(condition.values_count, values)
        if condition.geo_polygon is not None:
            if values is None:
                return False
            return any(check_geo_polygon(condition.geo_polygon, v) for v in values)
    elif isinstance(condition, models.NestedCondition):
        return check_nested_filter(
            condition.nested.filter, nested_filter_values(payload, condition.nested.key)
        )
    elif isinstance(condition, models.Filter):
        return check_filter(condition, payload, point_id, has_vector)
    else:
        raise ValueError(f"Unknown condition: {condition}")
    return False


def check_must(
    conditions: list[models.Condition],
    payload: dict,
    point_id: models.ExtendedPointId,
    has_vector: dict[str, bool],
) -> bool:
    return all(
        check_condition(condition, payload, point_id, has_vector) for condition in conditions
    )


def check_must_not(
    conditions: list[models.Condition],
    payload: dict,
    point_id: models.ExtendedPointId,
    has_vector: dict[str, bool],
) -> bool:
    return all(
        not check_condition(condition, payload, point_id, has_vector) for condition in conditions
    )


def check_should(
    conditions: list[models.Condition],
    payload: dict,
    point_id: models.ExtendedPointId,
    has_vector: dict[str, bool],
) -> bool:
    return any(
        check_condition(condition, payload, point_id, has_vector) for condition in conditions
    )


def check_min_should(
    conditions: list[models.Condition],
    payload: dict,
    point_id: models.ExtendedPointId,
    vectors: dict[str, Any],
    min_count: int,
) -> bool:
    return (
        sum(check_condition(condition, payload, point_id, vectors) for condition in conditions)
        >= min_count
    )


def check_filter(
    payload_filter: models.Filter,
    payload: dict,
    point_id: models.ExtendedPointId,
    has_vector: dict[str, bool],
) -> bool:
    def ensure_condition_list(
        condition: models.Condition | list[models.Condition],
    ) -> list[models.Condition]:
        if isinstance(condition, list):
            return condition
        return [condition]

    if payload_filter.must is not None:
        if not check_must(
            ensure_condition_list(payload_filter.must), payload, point_id, has_vector
        ):
            return False
    if payload_filter.must_not is not None:
        if not check_must_not(
            ensure_condition_list(payload_filter.must_not), payload, point_id, has_vector
        ):
            return False
    if payload_filter.should is not None:
        if not check_should(
            ensure_condition_list(payload_filter.should), payload, point_id, has_vector
        ):
            return False
    if payload_filter.min_should is not None:
        if not check_min_should(
            payload_filter.min_should.conditions,
            payload,
            point_id,
            has_vector,
            payload_filter.min_should.min_count,
        ):
            return False
    return True


def calculate_payload_mask(
    payloads: list[dict],
    payload_filter: models.Filter | None,
    ids_inv: list[models.ExtendedPointId],
    deleted_per_vector: dict[str, np.ndarray],
) -> types.NumpyArray:
    if payload_filter is None:
        return np.ones(len(payloads), dtype=bool)

    mask: types.NumpyArray = np.zeros(len(payloads), dtype=bool)
    for i, payload in enumerate(payloads):
        has_vector = {}
        for vector_name, deleted in deleted_per_vector.items():
            if not deleted[i]:
                has_vector[vector_name] = True

        if check_filter(payload_filter, payload, ids_inv[i], has_vector):
            mask[i] = True
    return mask
