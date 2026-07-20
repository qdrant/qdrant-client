from typing import Any

try:
    import orjson

    BACKEND = "orjson"

    def loads(data: bytes | str) -> Any:
        if isinstance(data, str):
            data = data.encode("utf-8")
        return orjson.loads(data)

except ImportError:
    import json

    BACKEND = "json"

    def loads(data: bytes | str) -> Any:
        return json.loads(data)
