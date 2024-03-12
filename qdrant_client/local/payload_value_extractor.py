from typing import Any, Dict, List, Optional


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


def set_value_by_key(payload: Dict[str, Any], new_value: Dict[str, Any], key: str) -> None:
    keys = key.split(".")

    def _set_value(data: Any, k_list: List[str], new_value: Dict[str, Any]) -> None:
        if not k_list:
            return

        k = k_list.pop(0)

        if k.endswith("]"):
            k, index = k.split("[")
            if k not in data:
                data[k] = []
                return

            data = data.get(k)

            if not isinstance(data, list):
                return

            index = index.strip("]")
            if index == "":
                for item in data:
                    if isinstance(item, dict) and len(k_list) == 0:
                        item.update(new_value)
                    else:
                        _set_value(item, k_list.copy(), new_value)
            else:
                i = int(index)
                if i < len(data):
                    if isinstance(data[i], dict) and len(k_list) == 0:
                        data[i].update(new_value)
                    else:
                        _set_value(data[i], k_list.copy(), new_value)
        else:
            if len(k_list) == 0:
                if k in data and isinstance(data[k], dict):
                    data[k].update(new_value)
                else:
                    data[k] = new_value
                return
            if k in data:
                _set_value(data[k], k_list.copy(), new_value)

    _set_value(payload, keys, new_value)


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


def test_set_value_by_key() -> None:
    # Test case 1: Simple update at the root level
    payload: Dict[str, Any] = {"a": 1, "b": 2}
    new_value: Dict[str, Any] = {"c": 3}
    set_value_by_key(payload, new_value, "c")
    assert payload == {"a": 1, "b": 2, "c": {"c": 3}}

    # Test case 2: Nested update
    payload = {"a": {"b": {"c": 1}}}
    new_value = {"d": 2}
    set_value_by_key(payload, new_value, "a.b.d")
    assert payload == {"a": {"b": {"c": 1, "d": {"d": 2}}}}

    # Test case 3: Nested update with existing key
    payload = {"a": {"b": {"c": 1}}}
    new_value = {"c": 2}
    set_value_by_key(payload, new_value, "a.b")
    assert payload == {"a": {"b": {"c": 2}}}

    # Test case 4: Nested update with existing key and array index
    payload = {"a": [{"b": 1}, {"b": 2}]}
    new_value = {"c": 3}
    set_value_by_key(payload, new_value, "a[1]")
    assert payload == {"a": [{"b": 1}, {"b": 2, "c": 3}]}

    # Test case 5: Nested update with non-existing key and array index
    payload = {"a": []}
    new_value = {"b": {"c": 1}}
    set_value_by_key(payload, new_value, "a[0]")
    assert payload == {"a": []}

    # Test case 6: Deeply nested update
    payload = {"a": {"b": {"c": {"d": {"e": 1}}}}}
    new_value = {"f": 2}
    set_value_by_key(payload, new_value, "a.b.c.d")
    assert payload == {"a": {"b": {"c": {"d": {"e": 1, "f": 2}}}}}

    # Test case 7: Update with a nested new_value
    payload = {"a": {"b": {"c": 1}}}
    new_value = {"d": {"e": 2}}
    set_value_by_key(payload, new_value, "a.b.c")
    assert payload == {"a": {"b": {"c": {"d": {"e": 2}}}}}

    # Test case 8: Update with an empty payload
    payload = {}
    new_value = {"a": 1}
    set_value_by_key(payload, new_value, "a")
    assert payload == {"a": {"a": 1}}

    # Test case 9: Update with an array index that is out of range
    payload = {"a": [{"b": 1}]}
    new_value = {"c": 2}
    set_value_by_key(payload, new_value, "a[1]")
    assert payload == {"a": [{"b": 1}]}

    # Test case 10: Update with nested array index
    payload = {"a": {"b": [{"c": 1}, {"c": 2}]}}
    new_value = {"d": 3}
    set_value_by_key(payload, new_value, "a.b[0].c")
    assert payload == {"a": {"b": [{"c": {"d": 3}}, {"c": 2}]}}

    # Test case 11: Update with a complex nested structure
    payload = {"a": {"b": {"c": [{"d": 1}]}}}
    new_value = {"e": {"f": 2}}
    set_value_by_key(payload, new_value, "a.b.c[0].d")
    assert payload == {"a": {"b": {"c": [{"d": {"e": {"f": 2}}}]}}}

    # Test case 12: Update with an array index using negative indexing
    payload = {"a": [{"b": 1}, {"b": 2}]}
    new_value = {"c": 3}
    set_value_by_key(payload, new_value, "a[-1]")
    assert payload == {"a": [{"b": 1}, {"b": 2, "c": 3}]}
