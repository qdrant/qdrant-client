"""Parsing of REST response bodies into ``qdrant_client.http.models``.

Server responses are trusted, so with pydantic v2 the two most expensive parts of validating them are
skipped: dense and multi-dense vectors and payloads keep the lists and dicts the JSON decoder built, instead
of being checked and copied element by element. Everything else is validated as before, and the returned
objects are identical to the ones ``parse_as_type(json.loads(body), type_)`` builds.
"""

import json
from functools import lru_cache
from typing import Any, Callable, TypeVar

from pydantic import create_model

from qdrant_client._pydantic_compat import PYDANTIC_V2

if PYDANTIC_V2:
    from pydantic import TypeAdapter
    from pydantic_core import SchemaValidator

T = TypeVar("T")

try:
    from orjson import loads as _loads  # optional, used when installed
except ImportError:  # pragma: no cover
    try:
        from pydantic_core import from_json as _loads  # pydantic >= 2.5
    except ImportError:
        _loads = json.loads

# union members that never accept a list; definition-refs in the generated models always point to models
_NOT_LISTS = {"model", "definition-ref", "dict", "int", "float", "str", "bool", "uuid", "none"}
_ANY_LIST = {"type": "is-instance", "cls": list}
_ANY_DICT = {"type": "is-instance", "cls": dict}


def parse_response(content: bytes, type_: type[T]) -> T:
    """Parse a successful REST response body into ``type_``."""
    if PYDANTIC_V2:
        try:
            return _validator(type_)(_loads(content))  # type: ignore[arg-type]
        except ValueError:
            # a validation or JSON error: the old path below raises the same error as before
            pass
    return parse_as_type(json.loads(content), type_)


def parse_as_type(obj: Any, type_: type[T]) -> T:
    """The regular path: validate ``obj`` as the only field of a model wrapping ``type_``."""
    model_type = _get_parsing_type(type_, source=parse_as_type.__name__)  # type: ignore[arg-type]
    return model_type(obj=obj).obj


@lru_cache(maxsize=None)
def _get_parsing_type(type_: Any, source: str) -> Any:
    type_name = getattr(type_, "__name__", str(type_))
    return create_model(f"ParsingModel[{type_name}] (for {source})", obj=(type_, ...))


@lru_cache(maxsize=None)
def _validator(type_: Any) -> Callable[[Any], Any]:
    adapter = TypeAdapter(type_)
    on_path: set[type] = set()
    try:
        schema = _rewrite(adapter.core_schema, (), on_path)
        if on_path:
            return SchemaValidator(_use_shadows(schema, on_path)).validate_python
    except Exception:  # pragma: no cover  # a core schema layout this module does not know
        pass
    return adapter.validate_python


def _is_float_list(schema: Any) -> bool:
    # list[StrictFloat] or list[list[StrictFloat]]
    depth = 0
    while isinstance(schema, dict) and schema.get("type") == "list":
        schema, depth = schema.get("items_schema"), depth + 1
    return (
        depth > 0
        and isinstance(schema, dict)
        and schema.get("type") == "float"
        and schema.get("strict") is True
    )


def _is_vector_union(schema: dict[str, Any]) -> bool:
    # VectorStructOutput and VectorOutput: float lists plus members that never accept a list
    choices = schema["choices"] if schema.get("type") == "union" else []
    return any(_is_float_list(c) for c in choices) and all(
        _is_float_list(c) or (isinstance(c, dict) and c.get("type") in _NOT_LISTS) for c in choices
    )


def _is_payload(schema: dict[str, Any]) -> bool:
    # Payload, Dict[str, Any]: JSON object keys are always str
    return (
        schema.get("type") == "dict"
        and schema.get("keys_schema", {}).get("type") == "str"
        and schema.get("values_schema", {}).get("type") == "any"
    )


def _rewrite(schema: Any, models: tuple[type, ...], on_path: set[type]) -> Any:
    """Copy a core schema, letting vectors and payloads through as they are.

    ``on_path`` collects the models between the root and a rewritten field.
    """
    if isinstance(schema, list):
        return [_rewrite(item, models, on_path) for item in schema]
    if not isinstance(schema, dict):
        return schema
    if _is_vector_union(schema):
        on_path.update(models)
        rest = [_rewrite(c, models, on_path) for c in schema["choices"] if not _is_float_list(c)]
        return {**schema, "choices": [_ANY_LIST, *rest], "mode": "left_to_right"}
    if _is_payload(schema):
        on_path.update(models)
        return _ANY_DICT
    if schema.get("type") == "model":
        models = (*models, schema["cls"])
    return {key: _rewrite(value, models, on_path) for key, value in schema.items()}


def _use_shadows(schema: Any, on_path: set[type]) -> Any:
    """Point the models in ``on_path`` at private subclasses and switch the objects back afterwards.

    pydantic-core reuses a model's own validator wherever the model appears in a schema, which would ignore
    the rewritten fields. It builds a subclass marked incomplete from the given schema instead.
    """
    if isinstance(schema, list):
        return [_use_shadows(item, on_path) for item in schema]
    if not isinstance(schema, dict):
        return schema
    new = {key: _use_shadows(value, on_path) for key, value in schema.items()}
    if new.get("type") != "model" or new["cls"] not in on_path:
        return new
    cls = new["cls"]
    new["cls"] = _shadow(cls)
    wrapper = {
        "type": "function-after",
        "function": {"type": "no-info", "function": _switch_to(cls)},
        "schema": new,
    }
    if "ref" in new:
        # models used in several places are shared through definition-refs, which must reach the wrapper
        wrapper["ref"] = new.pop("ref")
    return wrapper


@lru_cache(maxsize=None)
def _shadow(cls: type) -> type:
    # type.__new__ skips pydantic's class construction: the subclass inherits everything and adds no slots,
    # so its objects can be switched to ``cls``
    namespace = {
        "__module__": __name__,
        "__qualname__": cls.__qualname__,
        "__pydantic_complete__": False,
    }
    return type.__new__(type(cls), cls.__name__, (cls,), namespace)


def _switch_to(cls: type) -> Callable[[Any], Any]:
    def switch(obj: Any) -> Any:
        object.__setattr__(obj, "__class__", cls)
        return obj

    return switch
