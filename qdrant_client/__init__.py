from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from qdrant_client.async_qdrant_client import AsyncQdrantClient
    from qdrant_client.qdrant_client import QdrantClient

_LAZY_EXPORTS: dict[str, str] = {
    "QdrantClient": "qdrant_client.qdrant_client",
    "AsyncQdrantClient": "qdrant_client.async_qdrant_client",
}


def __getattr__(name: str) -> object:
    module_path = _LAZY_EXPORTS.get(name)
    if module_path is not None:
        from importlib import import_module

        module = import_module(module_path)
        result = getattr(module, name)
        globals()[name] = result
        return result
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)


__all__ = ["AsyncQdrantClient", "QdrantClient"]
