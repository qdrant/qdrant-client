import ast
from typing import Dict, Optional


class ConstantTransformer(ast.NodeTransformer):
    def __init__(self, constant_replace_map: Optional[Dict[str, str]]):
        self.constant_replace_map = (
            constant_replace_map if constant_replace_map is not None else {}
        )

    def visit_Constant(self, node: ast.Constant) -> ast.AST:
        for old_value, new_value in self.constant_replace_map.items():
            if isinstance(node.value, str):
                node.value = node.value.replace(old_value, new_value)
        return self.generic_visit(node)
