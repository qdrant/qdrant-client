import ast
from typing import Dict, Optional


class RemoteImportFromTransformer(ast.NodeTransformer):
    def __init__(self, import_replace_map: Optional[Dict[str, str]] = None):
        self.import_replace_map = import_replace_map if import_replace_map is not None else {}

    def visit_ImportFrom(self, node: ast.ImportFrom) -> ast.AST:
        # update module name
        for old_value, new_value in self.import_replace_map.items():
            if node.module is not None:
                node.module = node.module.replace(old_value, new_value)

        # update imported item name

        for i, alias in enumerate(node.names):
            if hasattr(alias, "name"):
                for old_value, new_value in self.import_replace_map.items():
                    alias.name = alias.name.replace(old_value, new_value)
                if alias.name == "get_async_channel":
                    alias.asname = "get_channel"
        node.names = [alias for alias in node.names if alias.name != "get_channel"]

        return self.generic_visit(node)
