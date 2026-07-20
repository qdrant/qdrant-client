from typing import Any


try:
    import msgspec

    BACKEND = "msgspec"
    _decoder = msgspec.json.Decoder()

    def loads(data: bytes | str) -> Any:
        return _decoder.decode(data)

except ImportError:
    try:
        import orjson

        BACKEND = "orjson"

        def loads(data: bytes | str) -> Any:
            return orjson.loads(data)

    except ImportError:
        import json

        BACKEND = "json"

        def loads(data: bytes | str) -> Any:
            return json.loads(data)
