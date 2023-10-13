import ast
from typing import Dict, Optional


class AnnAssignTransformer(ast.NodeTransformer):
    def __init__(self, class_replace_map: Optional[Dict[str, str]] = None):
        self.class_replace_map = class_replace_map if class_replace_map is not None else {}

    def visit_AnnAssign(self, node: ast.AnnAssign) -> ast.AST:
        for old_value, new_value in self.class_replace_map.items():
            if isinstance(node.annotation, ast.Name):
                node.annotation.id = node.annotation.id.replace(old_value, new_value)
        return self.generic_visit(node)
