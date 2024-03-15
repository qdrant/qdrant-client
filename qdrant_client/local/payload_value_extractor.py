from typing import Any, List, Optional

from qdrant_client.local.json_path_parser import (
    JsonPathItem,
    JsonPathItemType,
    parse_json_path,
)


def value_by_key(payload: dict, key: str, flat: bool = True) -> Optional[List[Any]]:
    """
    Get value from payload by key.
    Args:
        payload: arbitrary json-like object
        flat: If True, extend list of values. If False, append. By default, we use True and flatten the arrays,
            we need it for filters, however for `count` method we need to keep the arrays as is.
        key:
            Key or path to value in payload.
            Examples:
                - "name"
                - "address.city"
                - "location[].name"
                - "location[0].name"

    Returns:
        List of values or None if key not found.
    """
    keys = parse_json_path(key)
    result = []

    def _get_value(data: Any, k_list: List[JsonPathItem]) -> None:
        if not k_list:
            return

        current_key = k_list.pop(0)
        if len(k_list) == 0:
            if isinstance(data, dict) and current_key.item_type == JsonPathItemType.KEY:
                if current_key.key in data:
                    value = data[current_key.key]
                    if isinstance(value, list) and flat:
                        result.extend(value)
                    else:
                        result.append(value)

            elif isinstance(data, list):
                if current_key.item_type == JsonPathItemType.WILDCARD_INDEX:
                    result.extend(data)

                elif current_key.item_type == JsonPathItemType.INDEX:
                    assert current_key.index is not None

                    if current_key.index < len(data):
                        result.append(data[current_key.index])

        elif current_key.item_type == JsonPathItemType.KEY:
            if current_key.key in data:
                _get_value(data[current_key.key], k_list.copy())

        elif current_key.item_type == JsonPathItemType.INDEX:
            assert current_key.index is not None

            if current_key.index < len(data):
                _get_value(data[current_key.index], k_list.copy())

        elif current_key.item_type == JsonPathItemType.WILDCARD_INDEX:
            for item in data:
                _get_value(item, k_list.copy())

    _get_value(payload, keys)
    return result if result else None


def test_value_by_key() -> None:
    payload = {
        "name": "John",
        "counts": [1, 2, 3],
        "address": {
            "city": "New York",
        },
        "location": [
            {"name": "home", "counts": [1, 2, 3]},
            {"name": "work", "counts": [4, 5, 6]},
        ],
        "nested": [{"empty": []}, {"empty": []}, {"empty": None}],
        "the_null": None,
        "the": {"nested.key": "cuckoo"},
        "double-nest-array": [[1, 2], [3, 4], [5, 6]],
    }
    # region flat=True
    assert value_by_key(payload, "name") == ["John"]
    assert value_by_key(payload, "address.city") == ["New York"]
    assert value_by_key(payload, "location[].name") == ["home", "work"]
    assert value_by_key(payload, "location[0].name") == ["home"]
    assert value_by_key(payload, "location[1].name") == ["work"]
    assert value_by_key(payload, "location[2].name") is None
    assert value_by_key(payload, "location[].name[0]") is None
    assert value_by_key(payload, "location[0]") == [{"name": "home", "counts": [1, 2, 3]}]
    assert value_by_key(payload, "not_exits") is None
    assert value_by_key(payload, "address") == [{"city": "New York"}]
    assert value_by_key(payload, "address.city[0]") is None
    assert value_by_key(payload, "counts") == [1, 2, 3]
    assert value_by_key(payload, "location[].counts") == [1, 2, 3, 4, 5, 6]
    assert value_by_key(payload, "nested[].empty") == [None]
    assert value_by_key(payload, "the_null") == [None]
    assert value_by_key(payload, 'the."nested.key"') == ["cuckoo"]
    assert value_by_key(payload, "double-nest-array[][]") == [1, 2, 3, 4, 5, 6]
    assert value_by_key(payload, "double-nest-array[0][]") == [1, 2]
    assert value_by_key(payload, "double-nest-array[0][0]") == [1]
    assert value_by_key(payload, "double-nest-array[0][0]") == [1]
    assert value_by_key(payload, "double-nest-array[][1]") == [2, 4, 6]
    # endregion

    # region flat=False
    assert value_by_key(payload, "name", flat=False) == ["John"]
    assert value_by_key(payload, "address.city", flat=False) == ["New York"]
    assert value_by_key(payload, "location[].name", flat=False) == ["home", "work"]
    assert value_by_key(payload, "location[0].name", flat=False) == ["home"]
    assert value_by_key(payload, "location[1].name", flat=False) == ["work"]
    assert value_by_key(payload, "location[2].name", flat=False) is None
    assert value_by_key(payload, "location[].name[0]", flat=False) is None
    assert value_by_key(payload, "location[0]", flat=False) == [
        {"name": "home", "counts": [1, 2, 3]}
    ]
    assert value_by_key(payload, "not_exits", flat=False) is None
    assert value_by_key(payload, "address", flat=False) == [{"city": "New York"}]
    assert value_by_key(payload, "address.city[0]", flat=False) is None
    assert value_by_key(payload, "counts", flat=False) == [[1, 2, 3]]
    assert value_by_key(payload, "location[].counts", flat=False) == [
        [1, 2, 3],
        [4, 5, 6],
    ]
    assert value_by_key(payload, "nested[].empty", flat=False) == [[], [], None]
    assert value_by_key(payload, "the_null", flat=False) == [None]
    # endregion
