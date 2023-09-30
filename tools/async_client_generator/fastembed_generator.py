import ast
import inspect
from typing import Optional

from qdrant_client.async_client_base import AsyncQdrantBase


# Define a custom AST transformer to add 'await' before method calls
class AsyncAwaitTransformer(ast.NodeTransformer):
    def __init__(
        self,
        keep_sync: Optional[list[str]] = None,
        class_replace_map: Optional[dict] = None,
        import_replace_map: Optional[dict] = None,
    ):
        self._async_methods = None
        self.keep_sync = keep_sync if keep_sync is not None else []
        self.class_replace_map = class_replace_map if class_replace_map is not None else {}
        self.import_replace_map = import_replace_map if import_replace_map is not None else {}

    def _keep_sync(self, name: str) -> bool:
        return name in self.keep_sync

    @property
    def async_methods(self):
        if self._async_methods is None:
            self._async_methods = self.get_async_methods(AsyncQdrantBase)
        return self._async_methods

    @staticmethod
    def get_async_methods(class_obj):
        async_methods = []
        for name, method in inspect.getmembers(class_obj):
            if inspect.iscoroutinefunction(method):
                async_methods.append(name)
        return async_methods

    def visit_Call(self, node):
        if isinstance(node.func, ast.Attribute):
            if isinstance(node.func.value, ast.Name) and node.func.value.id == "self":
                if node.func.attr in self.async_methods:
                    return ast.Await(value=node)
        return self.generic_visit(node)

    def visit_FunctionDef(self, sync_node: ast.FunctionDef):
        if self._keep_sync(sync_node.name):
            return self.generic_visit(sync_node)
        async_node = ast.AsyncFunctionDef(
            name=sync_node.name,
            args=sync_node.args,
            body=sync_node.body,
            decorator_list=sync_node.decorator_list,
            returns=sync_node.returns,
            type_comment=sync_node.type_comment,
        )
        async_node.lineno = sync_node.lineno
        async_node.col_offset = sync_node.col_offset
        async_node.end_lineno = sync_node.end_lineno
        async_node.end_col_offset = sync_node.end_col_offset
        return self.generic_visit(async_node)

    def visit_Import(self, node: ast.Import):
        for old_value, new_value in self.import_replace_map.items():
            for alias in node.names:
                alias.name = alias.name.replace(old_value, new_value)
        return self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom):
        # update module name
        for old_value, new_value in self.import_replace_map.items():
            node.module = node.module.replace(old_value, new_value)

        # update imported item name
        for alias in node.names:
            if hasattr(alias, "name"):
                for old_value, new_value in self.import_replace_map.items():
                    alias.name = alias.name.replace(old_value, new_value)

        return self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef):
        # update class name
        for old_value, new_value in self.class_replace_map.items():
            node.name = node.name.replace(old_value, new_value)

        # update parent classes names
        for base in node.bases:
            for old_value, new_value in self.class_replace_map.items():
                base.id = base.id.replace(old_value, new_value)
        return self.generic_visit(node)


class FastembedAsyncAwaitTransformer(AsyncAwaitTransformer):
    def _keep_sync(self, name: str) -> bool:
        return name in self.keep_sync or name.startswith("_")


if __name__ == "__main__":
    from .config import CODE_DIR

    with open(CODE_DIR / "qdrant_fastembed.py", "r") as source_file:
        code = source_file.read()

    # Parse the code into an AST
    parsed_code = ast.parse(code)

    await_transformer = FastembedAsyncAwaitTransformer(
        keep_sync=[
            "__init__",
            "set_model",
        ],
        class_replace_map={
            "QdrantBase": "AsyncQdrantBase",
            "QdrantFastembedMixin": "AsyncQdrantFastembedMixin",
        },
        import_replace_map={
            "qdrant_client.client_base": "qdrant_client.async_client_base",
            "QdrantBase": "AsyncQdrantBase",
        },
    )
    modified_code_ast = await_transformer.visit(parsed_code)
    modified_code = ast.unparse(modified_code_ast)

    with open(CODE_DIR / "async_qdrant_fastembed.py", "w") as target_file:
        target_file.write(modified_code)
