import ast
from typing import Dict, Optional


class ImportTransformer(ast.NodeTransformer):
    def __init__(self, import_replace_map: Optional[Dict[str, str]] = None):
        self.import_replace_map = import_replace_map if import_replace_map is not None else {}

    def visit_Import(self, node: ast.Import) -> ast.AST:
        for old_value, new_value in self.import_replace_map.items():
            for alias in node.names:
                alias.name = alias.name.replace(old_value, new_value)
        return self.generic_visit(node)
