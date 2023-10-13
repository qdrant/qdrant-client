import ast
from typing import Dict, List, Optional, Union


class CallTransformer(ast.NodeTransformer):
    def __init__(
        self,
        class_replace_map: Optional[Dict[str, str]] = None,
        async_methods: Optional[List[str]] = None,
    ):
        self.class_replace_map = class_replace_map if class_replace_map is not None else {}
        self.async_methods = async_methods if async_methods is not None else []

    def visit_Call(self, node: ast.Call) -> Union[ast.AST, ast.Await]:
        if isinstance(node.func, ast.Name):
            if node.func.id in self.class_replace_map:
                node.func.id = self.class_replace_map[node.func.id]

        if isinstance(node.func, ast.Attribute):
            if node.func.attr in self.async_methods:
                return ast.Await(value=node)

        return self.generic_visit(node)
