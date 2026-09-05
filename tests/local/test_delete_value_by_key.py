import pytest

from qdrant_client.local.json_path_parser import parse_json_path
from qdrant_client.local.payload_value_setter import delete_value_by_key


def _delete(payload: dict, key: str) -> dict:
    delete_value_by_key(payload, parse_json_path(key))
    return payload


@pytest.mark.parametrize(
    ("payload", "key", "expected"),
    [
        # top-level key
        ({"a": 1, "b": 2}, "a", {"b": 2}),
        # nested dict path removes only the leaf, siblings preserved
        ({"a": {"b": 1, "c": 2}, "top": 9}, "a.b", {"a": {"c": 2}, "top": 9}),
        # deeper path
        ({"a": {"b": {"c": 1, "d": 2}}}, "a.b.c", {"a": {"b": {"d": 2}}}),
        # array index
        ({"loc": [{"x": 1}, {"x": 2}]}, "loc[0].x", {"loc": [{}, {"x": 2}]}),
        # array wildcard removes the field from every element
        (
            {"loc": [{"x": 1, "y": 2}, {"x": 3, "y": 4}]},
            "loc[].x",
            {"loc": [{"y": 2}, {"y": 4}]},
        ),
        # non-existent path is a no-op, nothing else touched
        ({"a": {"c": 2}}, "a.b", {"a": {"c": 2}}),
        ({"a": {"c": 2}}, "nope.nested", {"a": {"c": 2}}),
        # path through a non-dict is a no-op
        ({"a": 5}, "a.b", {"a": 5}),
    ],
)
def test_delete_value_by_key(payload, key, expected):
    assert _delete(payload, key) == expected
