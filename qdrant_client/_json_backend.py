from typing import Any


try:
    import msgspec

    BACKEND = "msgspec"
    DecodeError: type[Exception] = msgspec.DecodeError
    _decoder = msgspec.json.Decoder()

    def loads(data: bytes | str) -> Any:
        return _decoder.decode(data)

except ImportError:
    DecodeError = ValueError

    try:
        import orjson

        BACKEND = "orjson"

        def loads(data: bytes | str) -> Any:
            return orjson.loads(data)

    except ImportError:
        try:
            from pydantic_core import from_json  # pydantic >= 2.5

            BACKEND = "pydantic_core"

            def loads(data: bytes | str) -> Any:
                return from_json(data)

        except ImportError:
            import json

            BACKEND = "json"

            def loads(data: bytes | str) -> Any:
                return json.loads(data)
