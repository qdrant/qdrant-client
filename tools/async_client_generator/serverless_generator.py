import ast
import inspect

from qdrant_client.async_qdrant_remote import AsyncQdrantRemote
from qdrant_client.serverless.grpc.serverless_collections_pb2_grpc import CollectionsServiceStub
from tools.async_client_generator.base_generator import BaseGenerator
from tools.async_client_generator.transformers import (
    CallTransformer,
    ClassDefTransformer,
    ConstantTransformer,
    FunctionDefTransformer,
    ImportTransformer,
    NameTransformer,
)
from tools.async_client_generator.transformers.remote import RemoteImportFromTransformer


class ServerlessFunctionDefTransformer(FunctionDefTransformer):
    """FunctionDefTransformer with method removal.

    RemoteFunctionDefTransformer is not reusable here: it overrides `close` with an
    AsyncQdrantRemote-specific body, while the serverless close just delegates.
    """

    def __init__(
        self,
        keep_sync: list[str] | None = None,
        exclude_methods: list[str] | None = None,
    ):
        super().__init__(keep_sync=keep_sync)
        self.exclude_methods = exclude_methods if exclude_methods is not None else []

    def visit_FunctionDef(self, sync_node: ast.FunctionDef) -> ast.AST | None:
        if sync_node.name in self.exclude_methods:
            return None
        return super().visit_FunctionDef(sync_node)


# Same helpers as RemoteGenerator; that module cannot be imported standalone since it
# depends on the generated async_client_base, which only exists mid-run of the script.
def get_async_methods(class_obj: type) -> list[str]:
    return [
        name
        for name, method in inspect.getmembers(class_obj)
        if inspect.iscoroutinefunction(method)
    ]


def get_grpc_methods(grpc_stub_class: type) -> list[str]:
    parsed = ast.parse(inspect.getsource(grpc_stub_class))
    return [
        target.attr
        for node in ast.walk(parsed)
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Attribute)
        and isinstance(target.value, ast.Name)
        and target.value.id == "self"
    ]


class ServerlessGenerator(BaseGenerator):
    def __init__(self) -> None:
        super().__init__()

        class_replace_map = {
            "QdrantServerless": "AsyncQdrantServerless",
            "QdrantRemote": "AsyncQdrantRemote",
        }
        import_replace_map = {
            "qdrant_client.qdrant_remote": "qdrant_client.async_qdrant_remote",
            "QdrantRemote": "AsyncQdrantRemote",
        }
        # delegated point/collection methods are coroutines on AsyncQdrantRemote;
        # stub RPCs are awaitable on an aio channel
        async_methods = get_async_methods(AsyncQdrantRemote) + get_grpc_methods(
            CollectionsServiceStub
        )

        self.transformers.append(
            RemoteImportFromTransformer(import_replace_map=import_replace_map)
        )
        self.transformers.append(ClassDefTransformer(class_replace_map=class_replace_map))
        self.transformers.append(
            CallTransformer(class_replace_map=class_replace_map, async_methods=async_methods)
        )
        self.transformers.append(ImportTransformer(import_replace_map=import_replace_map))
        self.transformers.append(
            ServerlessFunctionDefTransformer(
                keep_sync=["__init__", "_collections", "_collections_timeout"],
                # a sync context manager makes no sense on the async client;
                # the regular async client has none either
                exclude_methods=["__enter__", "__exit__"],
            )
        )
        self.transformers.append(
            NameTransformer(
                class_replace_map=class_replace_map, import_replace_map=import_replace_map
            )
        )
        self.transformers.append(
            ConstantTransformer(
                constant_replace_map={
                    "QdrantServerless": "AsyncQdrantServerless",
                    ">>> client.create_collection(": ">>> await client.create_collection(",
                }
            )
        )


if __name__ == "__main__":
    from tools.async_client_generator.config import CLIENT_DIR, CODE_DIR

    with open(CLIENT_DIR / "serverless" / "client.py", "r") as source_file:
        code = source_file.read()

    generator = ServerlessGenerator()
    modified_code = generator.generate(code)

    with open(CODE_DIR / "async_client.py", "w") as target_file:
        target_file.write(modified_code)
