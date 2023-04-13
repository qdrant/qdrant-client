from typing import Any, List, Optional

import numpy as np

from qdrant_client.http import models
from qdrant_client.local.geo import geo_distance
from qdrant_client.local.payload_value_extractor import value_by_key


def check_values_count(condition: models.ValuesCount, values: Optional[List[Any]]) -> bool:
    count = len(values) if values is not None else 0
    if condition.lt is not None and count >= condition.lt:
        return False
    if condition.lte is not None and count > condition.lte:
        return False
    if condition.gt is not None and count <= condition.gt:
        return False
    if condition.gte is not None and count < condition.gte:
        return False
    return True


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

        return (
            condition.top_left.lat >= lat >= condition.bottom_right.lat
            and condition.top_left.lon <= lon <= condition.bottom_right.lon
        )

    return False


def check_range(condition: models.Range, value: Any) -> bool:
    if not isinstance(value, (int, float)):
        return False
    if condition.lt is not None and value >= condition.lt:
        return False
    if condition.lte is not None and value > condition.lte:
        return False
    if condition.gt is not None and value <= condition.gt:
        return False
    if condition.gte is not None and value < condition.gte:
        return False
    return True


def check_match(condition: models.Match, value: Any) -> bool:
    if isinstance(condition, models.MatchValue):
        return value == condition.value
    if isinstance(condition, models.MatchText):
        return value is not None and condition.text in value
    if isinstance(condition, models.MatchAny):
        return value in condition.any
    raise ValueError(f"Unknown match condition: {condition}")


def check_condition(
    condition: models.Condition, payload: dict, point_id: models.ExtendedPointId
) -> bool:
    if isinstance(condition, models.IsNullCondition):
        values = value_by_key(payload, condition.is_null.key)
        if values is None:
            return False
        if any(v is None for v in values):
            return True
    elif isinstance(condition, models.IsEmptyCondition):
        values = value_by_key(payload, condition.is_empty.key)
        if values is None or len(values) == 0 or all(v is None for v in values):
            return True
    elif isinstance(condition, models.HasIdCondition):
        if point_id in condition.has_id:
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
            return any(check_range(condition.range, v) for v in values)
        if condition.geo_bounding_box is not None:
            if values is None:
                return False
            return any(check_geo_bounding_box(condition.geo_bounding_box, v) for v in values)
        if condition.geo_radius is not None:
            if values is None:
                return False
            return any(check_geo_radius(condition.geo_radius, v) for v in values)
        if condition.values_count is not None:
            return check_values_count(condition.values_count, values)
    elif isinstance(condition, models.Filter):
        return check_filter(condition, payload, point_id)
    else:
        raise ValueError(f"Unknown condition: {condition}")
    return False


def check_must(
    conditions: List[models.Condition], payload: dict, point_id: models.ExtendedPointId
) -> bool:
    return all(check_condition(condition, payload, point_id) for condition in conditions)


def check_must_not(
    conditions: List[models.Condition], payload: dict, point_id: models.ExtendedPointId
) -> bool:
    return all(not check_condition(condition, payload, point_id) for condition in conditions)


def check_should(
    conditions: List[models.Condition], payload: dict, point_id: models.ExtendedPointId
) -> bool:
    return any(check_condition(condition, payload, point_id) for condition in conditions)


def check_filter(
    payload_fileter: models.Filter, payload: dict, point_id: models.ExtendedPointId
) -> bool:
    if payload_fileter.must is not None:
        if not check_must(payload_fileter.must, payload, point_id):
            return False
    if payload_fileter.must_not is not None:
        if not check_must_not(payload_fileter.must_not, payload, point_id):
            return False
    if payload_fileter.should is not None:
        if not check_should(payload_fileter.should, payload, point_id):
            return False
    return True


def calculate_payload_mask(
    payloads: List[dict],
    payload_fileter: Optional[models.Filter],
    ids_inv: List[models.ExtendedPointId],
) -> np.ndarray:
    if payload_fileter is None:
        return np.ones(len(payloads), dtype=bool)

    mask = np.zeros(len(payloads), dtype=bool)
    for i, payload in enumerate(payloads):
        if check_filter(payload_fileter, payload, ids_inv[i]):
            mask[i] = True
    return mask
