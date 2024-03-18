from typing import Any, Dict, List, Optional, Type, Union

from qdrant_client.local.json_path_parser import (
    JsonPathItem,
    JsonPathItemType,
    parse_json_path,
)


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
    Setter.add_setter(JsonPathItemType.KEY, KeySetter)
    Setter.add_setter(JsonPathItemType.INDEX, IndexSetter)
    Setter.add_setter(JsonPathItemType.WILDCARD_INDEX, WildcardIndexSetter)
    Setter.set(payload, keys, value, None, None)


class Setter:
    TYPE: Any
    SETTERS: Dict[JsonPathItemType, Type["Setter"]] = {}

    @classmethod
    def add_setter(cls, item_type: JsonPathItemType, setter: Type["Setter"]) -> None:
        cls.SETTERS[item_type] = setter

    @classmethod
    def set(
        cls,
        data: Any,
        k_list: List[JsonPathItem],
        value: Dict[str, Any],
        prev_data: Any,
        prev_key: Optional[JsonPathItem],
    ) -> None:
        if not k_list:
            return

        current_key = k_list.pop(0)
        cls.SETTERS[current_key.item_type]._set(
            data,
            current_key,
            k_list,
            value,
            prev_data,
            prev_key,
        )

    @classmethod
    def _set(
        cls,
        data: Any,
        current_key: JsonPathItem,
        k_list: List[JsonPathItem],
        value: Dict[str, Any],
        prev_data: Any,
        prev_key: Optional[JsonPathItem],
    ) -> None:
        if isinstance(data, cls.TYPE):
            cls._set_compatible_types(
                data=data, current_key=current_key, k_list=k_list, value=value
            )
        else:
            cls._set_incompatible_types(
                current_key=current_key,
                k_list=k_list,
                value=value,
                prev_data=prev_data,
                prev_key=prev_key,
            )

    @classmethod
    def _set_compatible_types(
        cls,
        data: Any,
        current_key: JsonPathItem,
        k_list: List[JsonPathItem],
        value: Dict[str, Any],
    ) -> None:
        raise NotImplementedError()

    @classmethod
    def _set_incompatible_types(
        cls,
        current_key: JsonPathItem,
        k_list: List[JsonPathItem],
        value: Dict[str, Any],
        prev_data: Any,
        prev_key: Optional[JsonPathItem],
    ) -> None:
        raise NotImplementedError()


class KeySetter(Setter):
    TYPE = Dict

    @classmethod
    def _set_compatible_types(
        cls,
        data: Any,
        current_key: JsonPathItem,
        k_list: List[JsonPathItem],
        value: Dict[str, Any],
    ) -> None:
        if current_key.key not in data:
            data[current_key.key] = {}

        if len(k_list) == 0:
            if isinstance(data[current_key.key], dict):
                data[current_key.key].update(value)
            else:
                data[current_key.key] = value
        else:
            cls.set(data[current_key.key], k_list.copy(), value, data, current_key)

    @classmethod
    def _set_incompatible_types(
        cls,
        current_key: JsonPathItem,
        k_list: List[JsonPathItem],
        value: Dict[str, Any],
        prev_data: Any,
        prev_key: Optional[JsonPathItem],
    ) -> None:
        assert prev_key is not None

        if len(k_list) == 0:
            if prev_key.item_type == JsonPathItemType.KEY:
                prev_data[prev_key.key] = {current_key.key: value}
            else:  # if prev key was WILDCARD, we need to pass INDEX instead with an index set
                prev_data[prev_key.index] = {current_key.key: value}
        else:
            if prev_key.item_type == JsonPathItemType.KEY:
                prev_data[prev_key.key] = {current_key.key: {}}
                cls.set(
                    prev_data[prev_key.key][current_key.key],
                    k_list.copy(),
                    value,
                    prev_data[prev_key.key],
                    current_key,
                )
            else:
                prev_data[prev_key.index] = {current_key.key: {}}
                cls.set(
                    prev_data[prev_key.index][current_key.key],
                    k_list.copy(),
                    value,
                    prev_data[prev_key.index],
                    current_key,
                )


class _ListSetter(Setter):
    TYPE = List

    @classmethod
    def _set_incompatible_types(
        cls,
        current_key: JsonPathItem,
        k_list: List[JsonPathItem],
        value: Dict[str, Any],
        prev_data: Any,
        prev_key: Optional[JsonPathItem],
    ) -> None:
        assert prev_key is not None

        if prev_key.item_type == JsonPathItemType.KEY:
            prev_data[prev_key.key] = []
            return
        else:
            prev_data[prev_key.index] = []
            return

    @classmethod
    def _set_compatible_types(
        cls,
        data: Any,
        current_key: JsonPathItem,
        k_list: List[JsonPathItem],
        value: Dict[str, Any],
    ) -> None:
        raise NotImplementedError()


class IndexSetter(_ListSetter):
    @classmethod
    def _set_compatible_types(
        cls,
        data: Any,
        current_key: JsonPathItem,
        k_list: List[JsonPathItem],
        value: Dict[str, Any],
    ) -> None:
        assert current_key.index is not None

        if current_key.index < len(data):
            if len(k_list) == 0:
                if isinstance(data[current_key.index], dict):
                    data[current_key.index].update(value)
                else:
                    data[current_key.index] = value
                return

            cls.set(data[current_key.index], k_list.copy(), value, data, current_key)


class WildcardIndexSetter(_ListSetter):
    @classmethod
    def _set_compatible_types(
        cls,
        data: Any,
        current_key: JsonPathItem,
        k_list: List[JsonPathItem],
        value: Dict[str, Any],
    ) -> None:
        if len(k_list) == 0:
            for i, item in enumerate(data):
                if isinstance(item, dict):
                    data[i].update(value)
                else:
                    data[i] = value
        else:
            for i, item in enumerate(data):
                cls.set(
                    item,
                    k_list.copy(),
                    value,
                    data,
                    JsonPathItem(item_type=JsonPathItemType.INDEX, index=i),
                )


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
