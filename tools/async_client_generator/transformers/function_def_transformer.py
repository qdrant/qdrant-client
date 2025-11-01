import ast
import sys
from typing import Optional


class FunctionDefTransformer(ast.NodeTransformer):
    def __init__(
        self,
        keep_sync: Optional[list[str]] = None,
        rename_methods: Optional[dict[str, str]] = None,
        class_replace_map: Optional[dict[str, str]] = None,
    ):
        self.keep_sync = keep_sync if keep_sync is not None else []
        self.rename_methods = rename_methods if rename_methods is not None else {}
        self.class_replace_map = class_replace_map if class_replace_map is not None else {}

    def _keep_sync(self, name: str) -> bool:
        return name in self.keep_sync

    def visit_FunctionDef(self, sync_node: ast.FunctionDef) -> ast.AST:
        if self._keep_sync(sync_node.name):
            return self.generic_visit(sync_node)

        returns = sync_node.returns
        if isinstance(returns, ast.Constant) and returns.value in self.class_replace_map:
            returns.value = self.class_replace_map[returns.value]
            pass

        params: list = [
            self.rename_methods.get(sync_node.name, sync_node.name),
            sync_node.args,
            sync_node.body,
            sync_node.decorator_list,
            sync_node.returns,
            sync_node.type_comment,
        ]

        if sys.version_info >= (3, 12):
            type_params = sync_node.type_params

        async_node = ast.AsyncFunctionDef(*params)

        async_node.lineno = sync_node.lineno
        async_node.col_offset = sync_node.col_offset
        async_node.end_lineno = sync_node.end_lineno
        async_node.end_col_offset = sync_node.end_col_offset
        return self.generic_visit(async_node)
