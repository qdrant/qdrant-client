from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import pytest
from pydantic import BaseModel


class JsonPathItemType(str, Enum):
    KEY = "key"
    INDEX = "index"
    WILDCARD_INDEX = "wildcard_index"


class JsonPathItem(BaseModel):
    item_type: JsonPathItemType
    index: Optional[
        int
    ] = None  # split into index and key instead of using Union, because pydantic coerces
    # int to str even in case of Union[int, str]. Tested with pydantic==1.10.14
    key: Optional[str] = None


def parse_json_path(key: str) -> List[JsonPathItem]:
    """Parse and validate json path

    Args:
        key: json path

    Returns:
        List[JsonPathItem]: json path split into separate keys

    Raises:
        ValueError: if json path is invalid or empty

    Examples:

        # >>> parse_json_path("a[0][1].b")
        # [
        # JsonPathItem(item_type=<JsonPathItemType.KEY: 'key'>, value='a'),
        # JsonPathItem(item_type=<JsonPathItemType.INDEX: 'index'>, value=0),
        # JsonPathItem(item_type=<JsonPathItemType.INDEX: 'index'>, value=1),
        # JsonPathItem(item_type=<JsonPathItemType.KEY: 'key'>, value='b')
        # ]
    """
    keys = []
    json_path = key
    while json_path:
        json_path_item, rest = match_quote(json_path)
        if json_path_item is None:
            json_path_item, rest = match_key(json_path)

        if json_path_item is None:
            raise ValueError("Invalid path")

        keys.append(json_path_item)
        brackets_chunks, rest = match_brackets(rest)
        keys.extend(brackets_chunks)
        json_path = trunk_sep(rest)
        if not json_path:
            return keys
        continue

    raise ValueError("Invalid path")


def trunk_sep(path: str) -> str:
    if not path:
        return path

    if len(path) == 1:
        raise ValueError("Invalid path")

    if path.startswith("."):
        return path[1:]

    elif path.startswith("["):
        return path
    else:
        raise ValueError("Invalid path")


def match_quote(path: str) -> Tuple[Optional[JsonPathItem], str]:
    if '"' not in path or not path.startswith('"'):
        return None, path

    if path.count('"') % 2 != 0:
        raise ValueError("Invalid path")

    left_quote_pos = 0
    right_quote_pos = path.find('"', 1)

    if left_quote_pos == (right_quote_pos + 1):
        raise ValueError("Invalid path")

    return (
        JsonPathItem(
            item_type=JsonPathItemType.KEY, key=path[left_quote_pos + 1 : right_quote_pos]
        ),
        path[right_quote_pos + 1 :],
    )


def match_key(path: str) -> Tuple[Optional[JsonPathItem], str]:
    key = []
    for char in path:
        if not char.isalnum() and char not in ["_", "-"]:
            break
        key.append(char)
    if not key:
        return None, path

    return JsonPathItem(item_type=JsonPathItemType.KEY, key="".join(key)), path[len(key) :]


def match_brackets(rest: str) -> Tuple[List[JsonPathItem], str]:
    keys = []

    while rest:
        json_path_item, rest = _match_brackets(rest)

        if json_path_item is None:
            break

        keys.append(json_path_item)

    return keys, rest


def _match_brackets(path: str) -> Tuple[Optional[JsonPathItem], str]:
    if "[" not in path or not path.startswith("["):
        return None, path

    left_bracket_pos = 0
    right_bracket_pos = path.find("]", left_bracket_pos + 1)

    if right_bracket_pos == (left_bracket_pos + 1):
        return (
            JsonPathItem(item_type=JsonPathItemType.WILDCARD_INDEX),
            path[right_bracket_pos + 1 :],
        )

    try:
        index = int(path[left_bracket_pos + 1 : right_bracket_pos])
        return (
            JsonPathItem(item_type=JsonPathItemType.INDEX, index=index),
            path[right_bracket_pos + 1 :],
        )
    except ValueError as e:
        raise ValueError("Invalid path") from e


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
            if isinstance(data, dict):
                if current_key.item_type == JsonPathItemType.KEY:
                    if current_key.key in data:
                        value = data[current_key.key]
                        if isinstance(value, list) and flat:
                            result.extend(value)
                        else:
                            result.append(value)
                    return
            elif isinstance(data, list):
                if current_key.item_type == JsonPathItemType.WILDCARD_INDEX:
                    for item in data:
                        if isinstance(data, dict):
                            continue
                        result.append(item)

                elif current_key.item_type == JsonPathItemType.INDEX:
                    assert current_key.index is not None

                    if current_key.index < len(data):
                        if isinstance(data[current_key.index], dict):
                            return
                        result.append(data[current_key.index])
                return

        if current_key.item_type == JsonPathItemType.KEY:
            if current_key.key in data:
                _get_value(data[current_key.key], k_list.copy())
            return

        if current_key.item_type == JsonPathItemType.INDEX:
            assert current_key.index is not None

            if current_key.index < len(data):
                _get_value(data[current_key.index], k_list.copy())
            return

        if current_key.item_type == JsonPathItemType.WILDCARD_INDEX:
            for item in data:
                _get_value(item, k_list.copy())
            return

    _get_value(payload, keys)
    return result if result else None


def set_value_by_key(payload: dict, key: str, value: Any) -> None:
    """
    Set value in payload by key.
    Args:
        payload: arbitrary json-like object
        key:
            Key or path to value in payload.
            Examples:
                - "name"
                - "address.city"
                - "location[].name"
                - "location[0].name"
        value: value to set
    """
    keys = parse_json_path(key)

    def _set_payload(
        data: Any,
        k_list: List[JsonPathItem],
        value: Dict[str, Any],
        prev_data: Any,
        prev_key: Optional[JsonPathItem],
    ) -> None:
        if not k_list:
            return

        current_key = k_list.pop(0)

        if current_key.item_type == JsonPathItemType.KEY:
            if isinstance(data, dict):
                if current_key.key not in data:
                    data[current_key.key] = {}

                if len(k_list) == 0:
                    if isinstance(data[current_key.key], dict):
                        data[current_key.key].update(value)
                    else:
                        data[current_key.key] = value
                    return

                _set_payload(data[current_key.key], k_list.copy(), value, data, current_key)

            else:
                assert prev_key is not None

                if len(k_list) == 0:
                    if prev_key.item_type == JsonPathItemType.KEY:
                        prev_data[prev_key.key] = {current_key.key: value}
                    else:  # if prev key was WILDCARD, we need to pass INDEX instead with an index set
                        prev_data[prev_key.index] = {current_key.key: value}
                    return
                else:
                    if prev_key.item_type == JsonPathItemType.KEY:
                        prev_data[prev_key.key] = {current_key.key: {}}
                        _set_payload(
                            prev_data[prev_key.key][current_key.key],
                            k_list.copy(),
                            value,
                            prev_data[prev_key.key],
                            current_key,
                        )
                    else:
                        prev_data[prev_key.index] = {current_key.key: {}}
                        _set_payload(
                            prev_data[prev_key.index][current_key.key],
                            k_list.copy(),
                            value,
                            prev_data[prev_key.index],
                            current_key,
                        )

        if current_key.item_type == JsonPathItemType.WILDCARD_INDEX:
            if isinstance(data, list):
                if len(k_list) == 0:
                    for i, item in enumerate(data):
                        if isinstance(item, dict):
                            data[i].update(value)
                        else:
                            data[i] = value
                    return
                else:
                    for i, item in enumerate(data):
                        _set_payload(
                            item,
                            k_list.copy(),
                            value,
                            data,
                            JsonPathItem(item_type=JsonPathItemType.INDEX, index=i),
                        )

            else:
                assert prev_key is not None

                if len(k_list) == 0:
                    if prev_key.item_type == JsonPathItemType.KEY:
                        prev_data[prev_key.key] = []
                    else:
                        prev_data[prev_key.index] = []
                    return
                else:
                    if prev_key.item_type == JsonPathItemType.KEY:
                        prev_data[prev_key.key] = []
                        return
                    else:
                        prev_data[prev_key.index] = []
                        return

        if current_key.item_type == JsonPathItemType.INDEX:
            assert current_key.index is not None

            if isinstance(data, list):
                if current_key.index < len(data):
                    if len(k_list) == 0:
                        if isinstance(data[current_key.index], dict):
                            data[current_key.index].update(value)
                        else:
                            data[current_key.index] = value
                        return
                    _set_payload(data[current_key.index], k_list.copy(), value, data, current_key)
                else:
                    return
            else:
                assert prev_key is not None

                if len(k_list) == 0:
                    if prev_key.item_type == JsonPathItemType.KEY:
                        prev_data[prev_key.key] = []
                    else:
                        prev_data[prev_key.index] = []
                    return
                else:
                    if prev_key.item_type == JsonPathItemType.KEY:
                        prev_data[prev_key.key] = []
                        return
                    else:
                        prev_data[prev_key.index] = []
                        return

    _set_payload(payload, keys, value, None, None)


def test_parse_json_path() -> None:
    jp_key = "a"
    keys = parse_json_path(jp_key)
    assert keys == [JsonPathItem(item_type=JsonPathItemType.KEY, key="a")]

    jp_key = "a.b"
    keys = parse_json_path(jp_key)
    assert keys == [
        JsonPathItem(item_type=JsonPathItemType.KEY, key="a"),
        JsonPathItem(item_type=JsonPathItemType.KEY, key="b"),
    ]

    jp_key = "a[0]"
    keys = parse_json_path(jp_key)
    assert keys == [
        JsonPathItem(item_type=JsonPathItemType.KEY, key="a"),
        JsonPathItem(item_type=JsonPathItemType.INDEX, index=0),
    ]

    jp_key = "a[0].b"
    keys = parse_json_path(jp_key)
    assert keys == [
        JsonPathItem(item_type=JsonPathItemType.KEY, key="a"),
        JsonPathItem(item_type=JsonPathItemType.INDEX, index=0),
        JsonPathItem(item_type=JsonPathItemType.KEY, key="b"),
    ]

    jp_key = "a[0].b[1]"
    keys = parse_json_path(jp_key)
    assert keys == [
        JsonPathItem(item_type=JsonPathItemType.KEY, key="a"),
        JsonPathItem(item_type=JsonPathItemType.INDEX, index=0),
        JsonPathItem(item_type=JsonPathItemType.KEY, key="b"),
        JsonPathItem(item_type=JsonPathItemType.INDEX, index=1),
    ]

    jp_key = "a[][]"
    keys = parse_json_path(jp_key)
    assert keys == [
        JsonPathItem(item_type=JsonPathItemType.KEY, key="a"),
        JsonPathItem(item_type=JsonPathItemType.WILDCARD_INDEX, index=None),
        JsonPathItem(item_type=JsonPathItemType.WILDCARD_INDEX, index=None),
    ]

    jp_key = "a[0][1]"
    keys = parse_json_path(jp_key)
    assert keys == [
        JsonPathItem(item_type=JsonPathItemType.KEY, key="a"),
        JsonPathItem(item_type=JsonPathItemType.INDEX, index=0),
        JsonPathItem(item_type=JsonPathItemType.INDEX, index=1),
    ]

    jp_key = "a[0][1].b"
    keys = parse_json_path(jp_key)
    assert keys == [
        JsonPathItem(item_type=JsonPathItemType.KEY, key="a"),
        JsonPathItem(item_type=JsonPathItemType.INDEX, index=0),
        JsonPathItem(item_type=JsonPathItemType.INDEX, index=1),
        JsonPathItem(item_type=JsonPathItemType.KEY, key="b"),
    ]

    jp_key = 'a."k.c"'
    keys = parse_json_path(jp_key)
    assert keys == [
        JsonPathItem(item_type=JsonPathItemType.KEY, key="a"),
        JsonPathItem(item_type=JsonPathItemType.KEY, key="k.c"),
    ]

    jp_key = 'a."c[][]".b'
    keys = parse_json_path(jp_key)
    assert keys == [
        JsonPathItem(item_type=JsonPathItemType.KEY, key="a"),
        JsonPathItem(item_type=JsonPathItemType.KEY, key="c[][]"),
        JsonPathItem(item_type=JsonPathItemType.KEY, key="b"),
    ]

    jp_key = 'a."c..q".b'
    keys = parse_json_path(jp_key)
    assert keys == [
        JsonPathItem(item_type=JsonPathItemType.KEY, key="a"),
        JsonPathItem(item_type=JsonPathItemType.KEY, key="c..q"),
        JsonPathItem(item_type=JsonPathItemType.KEY, key="b"),
    ]

    with pytest.raises(ValueError):
        jp_key = 'a."k.c'
        parse_json_path(jp_key)

    with pytest.raises(ValueError):
        jp_key = 'a."k.c".'
        parse_json_path(jp_key)

    with pytest.raises(ValueError):
        jp_key = 'a."k.c".[]'
        parse_json_path(jp_key)

    with pytest.raises(ValueError):
        jp_key = "a.'k.c'"
        parse_json_path(jp_key)

    with pytest.raises(ValueError):
        jp_key = "a["
        parse_json_path(jp_key)

    with pytest.raises(ValueError):
        jp_key = "a]"
        parse_json_path(jp_key)

    with pytest.raises(ValueError):
        jp_key = "a[]]"
        parse_json_path(jp_key)

    with pytest.raises(ValueError):
        jp_key = "a[][]."
        parse_json_path(jp_key)

    with pytest.raises(ValueError):
        jp_key = "a[][]b"
        parse_json_path(jp_key)

    with pytest.raises(ValueError):
        jp_key = ".a"
        parse_json_path(jp_key)

    with pytest.raises(ValueError):
        jp_key = "a[x]"
        parse_json_path(jp_key)

    with pytest.raises(ValueError):
        jp_key = 'a[]""'
        parse_json_path(jp_key)

    with pytest.raises(ValueError):
        jp_key = '""b'
        parse_json_path(jp_key)

    with pytest.raises(ValueError):
        jp_key = "[]"
        parse_json_path(jp_key)

    with pytest.raises(ValueError):
        jp_key = "a[.]"
        parse_json_path(jp_key)

    with pytest.raises(ValueError):
        jp_key = 'a["1"]'
        parse_json_path(jp_key)

    with pytest.raises(ValueError):
        jp_key = ""
        parse_json_path(jp_key)

    with pytest.raises(ValueError):
        jp_key = "a..c"
        parse_json_path(jp_key)

    with pytest.raises(ValueError):
        jp_key = "a.c[]b[]"
        parse_json_path(jp_key)

    with pytest.raises(ValueError):
        jp_key = "a.c[].[]"
        parse_json_path(jp_key)


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
    assert value_by_key(payload, "location[0]") is None
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


def test_set_value_by_key() -> None:
    # region valid keys

    payload: Dict[str, Any] = {}
    new_value: Dict[str, Any] = {}
    key = "a"
    set_value_by_key(payload, key, new_value)
    assert payload == {"a": {}}, payload

    payload = {"a": {"a": 2}}
    new_value = {}
    key = "a"
    set_value_by_key(payload, key, new_value)
    assert payload == {"a": {"a": 2}}, payload

    payload = {"a": {"a": 2}}
    new_value = {"b": 3}
    key = "a"
    set_value_by_key(payload, key, new_value)
    assert payload == {"a": {"a": 2, "b": 3}}, payload

    payload = {"a": {"a": 2}}
    new_value = {"a": 3}
    key = "a"
    set_value_by_key(payload, key, new_value)
    assert payload == {"a": {"a": 3}}, payload

    payload = {"a": {"a": 2}}
    new_value = {"a": 3}
    key = "a.a"
    set_value_by_key(payload, key, new_value)
    assert payload == {"a": {"a": {"a": 3}}}, payload

    payload = {"a": {"a": {"a": 1}}}
    new_value = {"b": 2}
    key = "a.a"
    set_value_by_key(payload, key, new_value)
    assert payload == {"a": {"a": {"a": 1, "b": 2}}}, payload

    payload = {"a": {"a": {"a": 1}}}
    new_value = {"a": 2}
    key = "a.a"
    set_value_by_key(payload, key, new_value)
    assert payload == {"a": {"a": {"a": 2}}}, payload

    payload = {"a": []}
    new_value = {"b": 2}
    key = "a[0]"
    set_value_by_key(payload, key, new_value)
    assert payload == {"a": []}, payload

    payload = {"a": [{}]}
    new_value = {"b": 2}
    key = "a[0]"
    set_value_by_key(payload, key, new_value)
    assert payload == {"a": [{"b": 2}]}, payload

    payload = {"a": [{"a": 1}]}
    new_value = {"b": 2}
    key = "a[0]"
    set_value_by_key(payload, key, new_value)
    assert payload == {"a": [{"a": 1, "b": 2}]}, payload

    payload = {"a": [[]]}
    new_value = {"b": 2}
    key = "a[0]"
    set_value_by_key(payload, key, new_value)
    assert payload == {"a": [{"b": 2}]}, payload

    payload = {"a": [[]]}
    new_value = {"b": 2}
    key = "a[1]"
    set_value_by_key(payload, key, new_value)
    assert payload == {"a": [[]]}, payload

    payload = {"a": [{"a": []}]}
    new_value = {"b": 2}
    key = "a[0].a"
    set_value_by_key(payload, key, new_value)
    assert payload == {"a": [{"a": {"b": 2}}]}, payload

    payload = {"a": [{"a": []}]}
    new_value = {"b": 2}
    key = "a[].a"
    set_value_by_key(payload, key, new_value)
    assert payload == {"a": [{"a": {"b": 2}}]}, payload

    payload = {"a": [{"a": []}, {"a": []}]}
    new_value = {"b": 2}
    key = "a[].a"
    set_value_by_key(payload, key, new_value)
    assert payload == {"a": [{"a": {"b": 2}}, {"a": {"b": 2}}]}, payload

    payload = {"a": 1, "b": 2}
    new_value = {"c": 3}
    key = "c"
    set_value_by_key(payload, key, new_value)
    assert payload == {"a": 1, "b": 2, "c": {"c": 3}}, payload

    payload = {"a": {"b": {"c": 1}}}
    new_value = {"d": 2}
    key = "a.b.d"
    set_value_by_key(payload, key, new_value)
    assert payload == {"a": {"b": {"c": 1, "d": {"d": 2}}}}, payload

    payload = {"a": {"b": {"c": 1}}}
    new_value = {"c": 2}
    key = "a.b"
    set_value_by_key(payload, key, new_value)
    assert payload == {"a": {"b": {"c": 2}}}, payload

    payload = {"a": [{"b": 1}, {"b": 2}]}
    new_value = {"c": 3}
    key = "a[1]"
    set_value_by_key(payload, key, new_value)
    assert payload == {"a": [{"b": 1}, {"b": 2, "c": 3}]}, payload

    payload = {"a": []}
    new_value = {"b": {"c": 1}}
    key = "a[0]"
    set_value_by_key(payload, key, new_value)
    assert payload == {"a": []}, payload

    payload = {"a": {"b": {"c": {"d": {"e": 1}}}}}
    new_value = {"f": 2}
    key = "a.b.c.d"
    set_value_by_key(payload, key, new_value)
    assert payload == {"a": {"b": {"c": {"d": {"e": 1, "f": 2}}}}}, payload

    payload = {"a": {"b": {"c": 1}}}
    new_value = {"d": {"e": 2}}
    key = "a.b.c"
    set_value_by_key(payload, key, new_value)
    assert payload == {"a": {"b": {"c": {"d": {"e": 2}}}}}, payload

    payload = {"a": [{"b": 1}]}
    new_value = {"c": 2}
    key = "a[1]"
    set_value_by_key(payload, key, new_value)
    assert payload == {"a": [{"b": 1}]}, payload

    payload = {"a": {"b": [{"c": 1}, {"c": 2}]}}
    new_value = {"d": 3}
    key = "a.b[0].c"
    set_value_by_key(payload, key, new_value)
    assert payload == {"a": {"b": [{"c": {"d": 3}}, {"c": 2}]}}, payload

    payload = {"a": {"b": {"c": [{"d": 1}]}}}
    new_value = {"e": {"f": 2}}
    key = "a.b.c[0].d"
    set_value_by_key(payload, key, new_value)
    assert payload == {"a": {"b": {"c": [{"d": {"e": {"f": 2}}}]}}}, payload

    payload = {"a": [[{"b": 1}], [{"b": 2}]]}
    new_value = {"c": 3}
    key = "a[0][0]"
    set_value_by_key(payload, key, new_value)
    assert payload == {"a": [[{"b": 1, "c": 3}], [{"b": 2}]]}, payload

    payload = {"a": [[{"b": 1}], [{"b": 2}]]}
    new_value = {"c": 3}
    key = "a[1][0]"
    set_value_by_key(payload, key, new_value)
    assert payload == {"a": [[{"b": 1}], [{"b": 2, "c": 3}]]}, payload

    payload = {"a": [[{"b": 1}], [{"b": 2}]]}
    new_value = {"c": 3}
    key = "a[1][1]"
    set_value_by_key(payload, key, new_value)
    assert payload == {"a": [[{"b": 1}], [{"b": 2}]]}, payload

    payload = {"a": [[{"b": 1}], [{"b": 2}]]}
    new_value = {"c": 3}
    key = "a[][0]"
    set_value_by_key(payload, key, new_value)
    assert payload == {"a": [[{"b": 1, "c": 3}], [{"b": 2, "c": 3}]]}, payload

    payload = {"a": [[{"b": 1}], [{"b": 2}]]}
    new_value = {"c": 3}
    key = "a[][]"
    set_value_by_key(payload, key, new_value)
    assert payload == {"a": [[{"b": 1, "c": 3}], [{"b": 2, "c": 3}]]}, payload

    payload = {"a": []}
    new_value = {"c": 3}
    key = 'a."b.c"'
    set_value_by_key(payload, key, new_value)
    assert payload == {"a": {"b.c": {"c": 3}}}, payload

    payload = {"a": {"c": [1]}}
    new_value = {"a": 1}
    key = "a.c[0]"
    set_value_by_key(payload, key, new_value)
    assert payload == {"a": {"c": [{"a": 1}]}}, payload

    payload = {"a": {"c": [1]}}
    new_value = {"a": 1}
    key = "a.c[0].d"
    set_value_by_key(payload, key, new_value)
    assert payload == {"a": {"c": [{"d": {"a": 1}}]}}, payload

    # endregion

    # region exceptions

    try:
        payload = {"a": []}
        new_value = {"c": 3}
        key = "a.'b.c'"
        set_value_by_key(payload, key, new_value)
        assert False, f"Should've raised an exception due to the key with incorrect quotes: {key}"
    except Exception:
        assert True

    try:
        payload = {"a": [{"b": 1}, {"b": 2}]}
        new_value = {"c": 3}
        key = "a[-1]"
        set_value_by_key(payload, key, new_value)
        assert False, "Negative indexation is not supported"
    except Exception:
        assert True

    try:
        payload = {"a": [{"b": 1}, {"b": 2}]}
        new_value = {"c": 3}
        key = "a["
        set_value_by_key(payload, key, new_value)
        assert False, f"Should've raised an exception due to the incorrect key: {key}"
    except Exception:
        assert True

    try:
        payload = {"a": [{"b": 1}, {"b": 2}]}
        new_value = {"c": 3}
        key = "a]"
        set_value_by_key(payload, key, new_value)
        assert False, f"Should've raise an exception due to the incorrect key: {key}"
    except Exception:
        assert True

    # endregion

    # region wrong keys
    payload = {"a": []}
    new_value = {}
    key = "a.b[0]"
    set_value_by_key(payload, key, new_value)
    assert payload == {"a": {"b": []}}, payload

    payload = {"a": []}
    new_value = {}
    key = "a.b"
    set_value_by_key(payload, key, new_value)
    assert payload == {"a": {"b": {}}}, payload

    payload = {"a": []}
    new_value = {"c": 2}
    key = "a.b"
    set_value_by_key(payload, key, new_value)
    assert payload == {"a": {"b": {"c": 2}}}, payload

    payload = {"a": [[{"a": 1}]]}
    new_value = {"a": 2}
    key = "a.b[0][0]"
    set_value_by_key(payload, key, new_value)
    assert payload == {"a": {"b": []}}, payload

    payload = {"a": {"c": 2}}
    new_value = {"a": 1}
    key = "a[]"
    set_value_by_key(payload, key, new_value)
    assert payload == {"a": []}, payload

    payload = {"a": {"c": 2}}
    new_value = {"a": 1}
    key = "a[].b"
    set_value_by_key(payload, key, new_value)
    assert payload == {"a": []}, payload

    payload = {"a": {"c": [1]}}
    new_value = {"a": 1}
    key = "a.c[][][0]"
    set_value_by_key(payload, key, new_value)
    assert payload == {"a": {"c": [[]]}}, payload

    payload = {"a": {"c": [{"d": 1}]}}
    new_value = {"a": 1}
    key = "a.c[][]"
    set_value_by_key(payload, key, new_value)
    assert payload == {"a": {"c": [[]]}}, payload
    # endregion
