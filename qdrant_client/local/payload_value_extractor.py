import uuid
from functools import lru_cache
from typing import Any

from qdrant_client.local.json_path_parser import (
    JsonPathItem,
    JsonPathItemType,
    parse_json_path,
)


def value_by_key(payload: dict[str, Any], key: str, flat: bool = True) -> list[Any] | None:
    """
    Get value from payload by key.
    Args:
        payload: arbitrary json-like object
        flat: If True, unwrap a selected array one level, like the server does. If False, keep
            arrays as is.
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

    def _get_value(data: Any, k_list: list[JsonPathItem]) -> None:
        if not k_list:
            # the whole path is consumed, `data` is one of the values it selects
            if isinstance(data, list) and flat:
                result.extend(data)
            else:
                result.append(data)
            return

        current_key, rest = k_list[0], k_list[1:]

        if current_key.item_type == JsonPathItemType.KEY:
            if isinstance(data, dict) and current_key.key in data:
                _get_value(data[current_key.key], rest)

        elif current_key.item_type == JsonPathItemType.INDEX:
            assert current_key.index is not None

            if isinstance(data, list) and current_key.index < len(data):
                _get_value(data[current_key.index], rest)

        elif current_key.item_type == JsonPathItemType.WILDCARD_INDEX:
            if isinstance(data, list):
                for item in data:
                    _get_value(item, rest)

    _get_value(payload, keys)
    return result if result else None


def include_payload(payload: dict[str, Any], keys: list[str]) -> dict[str, Any]:
    """Keep only the parts of payload under the given keys, like `PayloadSelectorInclude`."""
    return _include(payload, _selector_tree(tuple(keys)))


def exclude_payload(payload: dict[str, Any], keys: list[str]) -> dict[str, Any]:
    """Drop the parts of payload under the given keys, like `PayloadSelectorExclude`."""
    return _exclude(payload, _selector_tree(tuple(keys)))


# cached so a request builds the tree once, not once per returned point; the size isn't tuned
@lru_cache(maxsize=128)
def _selector_tree(keys: tuple[str, ...]) -> dict:
    """Merge payload selector keys into a prefix tree.

    `a.b`, `c[]`, `d[0]` -> {"a": {"b": True}, "c": {None: True}, "d": {0: True}}, where True
    selects everything under it. Cached, don't modify.
    """
    tree: dict = {}
    for key in keys:
        *parents, last = [
            item.key if item.item_type == JsonPathItemType.KEY else item.index
            for item in parse_json_path(key)
        ]
        node = tree
        for part in parents:
            node = node.setdefault(part, {})
            if node is True:  # a shorter path already selects everything below
                break
        else:
            node[last] = True
    return tree


def _include(value: Any, tree: dict) -> Any:
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            if key not in tree:
                continue
            subtree = tree[key]
            result[key] = item if subtree is True else _include(item, subtree)
        return result
    if isinstance(value, list):
        # only `[]` applies to list items, `a[0]` matches nothing, same as in core
        if None not in tree:
            return []
        subtree = tree[None]
        if subtree is True:
            return value
        return [_include(item, subtree) for item in value]
    return value


def _exclude(value: Any, tree: dict) -> Any:
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            if key not in tree:
                result[key] = item
            elif tree[key] is not True:
                result[key] = _exclude(item, tree[key])
        return result
    if isinstance(value, list):
        if None not in tree:
            return value
        subtree = tree[None]
        if subtree is True:
            return []
        return [_exclude(item, subtree) for item in value]
    return value


def parse_uuid(value: Any) -> uuid.UUID | None:
    """
    Parse UUID from value.
    Args:
        value: arbitrary value
    """
    try:
        return uuid.UUID(str(value))
    except ValueError:
        return None
