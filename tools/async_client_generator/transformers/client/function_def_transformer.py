import ast
from typing import Optional

from tools.async_client_generator.transformers import FunctionDefTransformer


class ClientFunctionDefTransformer(FunctionDefTransformer):
    def __init__(
        self,
        keep_sync: Optional[list[str]] = None,
        class_replace_map: Optional[dict[str, str]] = None,
        exclude_methods: Optional[list[str]] = None,
        async_methods: Optional[list[str]] = None,
        rename_methods: Optional[dict[str, str]] = None,
    ):
        super().__init__(keep_sync, rename_methods, class_replace_map)
        self.exclude_methods = exclude_methods if exclude_methods is not None else []
        self.async_methods = async_methods if async_methods is not None else []

    def _keep_sync(self, name: str) -> bool:
        return name in self.keep_sync or (
            name not in self.async_methods
            and self.rename_methods.get(name, name) not in self.async_methods
        )

    def visit_FunctionDef(self, sync_node: ast.FunctionDef) -> Optional[ast.AST]:
        if sync_node.name in self.exclude_methods:
            return None

        return super().visit_FunctionDef(sync_node)
