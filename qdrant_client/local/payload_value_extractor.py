from typing import Any, List, Optional


def value_by_key(payload: dict, key: str, flat: bool = True) -> Optional[List[Any]]:
    """
    Get value from payload by key.
    Args:
        payload: arbitrary json-like object
        flat: If True, extend list of values. If False, append
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
    keys = key.split(".")
    result = []

    def _get_value(data: Any, k_list: List[str]) -> None:
        if not k_list:
            return

        k = k_list.pop(0)
        if len(k_list) == 0:
            if k not in data:
                return

            value = data[k]
            if isinstance(value, list) and flat:
                result.extend(value)
            else:
                result.append(value)
            return

        if not isinstance(data, dict):
            return

        if k.endswith("]"):
            k, index = k.split("[")
            data = data.get(k)

            if not isinstance(data, list):
                return

            index = index.strip("]")
            if index == "":
                for item in data:
                    _get_value(item, k_list.copy())
            else:
                i = int(index)
                if i < len(data):
                    _get_value(data[i], k_list.copy())
        else:
            if k in data:
                _get_value(data[k], k_list.copy())

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
    }
    # region flat=True
    assert value_by_key(payload, "name") == ["John"]
    assert value_by_key(payload, "address.city") == ["New York"]
    assert value_by_key(payload, "location[].name") == ["home", "work"]
    assert value_by_key(payload, "location[0].name") == ["home"]
    assert value_by_key(payload, "location[1].name") == ["work"]
    assert value_by_key(payload, "location[2].name") is None
    assert value_by_key(payload, "location[].name[0]") is None
    assert value_by_key(payload, "location[0]") is None
    assert value_by_key(payload, "not_exits") is None
    assert value_by_key(payload, "address") == [{"city": "New York"}]
    assert value_by_key(payload, "address.city[0]") is None
    assert value_by_key(payload, "counts") == [1, 2, 3]
    assert value_by_key(payload, "location[].counts") == [1, 2, 3, 4, 5, 6]
    assert value_by_key(payload, "nested[].empty") == [None]
    assert value_by_key(payload, "the_null") == [None]
    # endregion

    # region flat=False
    assert value_by_key(payload, "name", flat=False) == ["John"]
    assert value_by_key(payload, "address.city", flat=False) == ["New York"]
    assert value_by_key(payload, "location[].name", flat=False) == ["home", "work"]
    assert value_by_key(payload, "location[0].name", flat=False) == ["home"]
    assert value_by_key(payload, "location[1].name", flat=False) == ["work"]
    assert value_by_key(payload, "location[2].name", flat=False) is None
    assert value_by_key(payload, "location[].name[0]", flat=False) is None
    assert value_by_key(payload, "location[0]", flat=False) is None
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
