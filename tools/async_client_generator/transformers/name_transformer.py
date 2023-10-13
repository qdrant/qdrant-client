import ast
from typing import Dict, Optional


class NameTransformer(ast.NodeTransformer):
    """Required to change return statement type hints"""

    def __init__(
        self,
        class_replace_map: Optional[Dict[str, str]] = None,
        import_replace_map: Optional[Dict[str, str]] = None,
        rename_methods: Optional[Dict[str, str]] = None,
    ):
        self.class_replace_map = class_replace_map if class_replace_map is not None else {}
        self.import_replace_map = import_replace_map if import_replace_map is not None else {}
        self.rename_methods = rename_methods if rename_methods is not None else {}

    def visit_Name(self, node: ast.Name) -> ast.AST:
        if node.id in self.class_replace_map:
            node.id = self.class_replace_map[node.id]
        elif node.id in self.import_replace_map:
            node.id = self.import_replace_map[node.id]
        elif node.id in self.rename_methods:
            node.id = self.rename_methods[node.id]
        return self.generic_visit(node)
