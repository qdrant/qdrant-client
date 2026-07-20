from qdrant_client import _json_backend


def test_backend_is_known() -> None:
    assert _json_backend.BACKEND in ("msgspec", "orjson", "json")


def test_loads_accepts_str_and_bytes() -> None:
    assert _json_backend.loads('{"a": 1}') == {"a": 1}
    assert _json_backend.loads(b'{"a": 1}') == {"a": 1}
