import ast
import sys
from typing import List, Optional


class FunctionDefTransformer(ast.NodeTransformer):
    def __init__(self, keep_sync: Optional[List[str]] = None):
        self.keep_sync = keep_sync if keep_sync is not None else []

    def _keep_sync(self, name: str) -> bool:
        return name in self.keep_sync

    def visit_FunctionDef(self, sync_node: ast.FunctionDef) -> ast.AST:
        if self._keep_sync(sync_node.name):
            return self.generic_visit(sync_node)

        async_kwargs = {
            "name": sync_node.name,
            "args": sync_node.args,
            "body": sync_node.body,
            "decorator_list": sync_node.decorator_list,
            "returns": sync_node.returns,
            "type_comment": sync_node.type_comment,
        }

        if sys.version_info >= (3, 12):
            async_kwargs["type_params"] = []

        async_node = ast.AsyncFunctionDef(**async_kwargs)

        async_node.lineno = sync_node.lineno
        async_node.col_offset = sync_node.col_offset
        async_node.end_lineno = sync_node.end_lineno
        async_node.end_col_offset = sync_node.end_col_offset
        return self.generic_visit(async_node)
