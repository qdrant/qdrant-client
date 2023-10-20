import ast
from typing import Union

from tools.async_client_generator.transformers.call_transformer import CallTransformer


class LocalCallTransformer(CallTransformer):
    def visit_Call(self, node: ast.Call) -> Union[ast.AST, ast.Await]:
        if isinstance(node.func, ast.Name):
            if node.func.id in self.class_replace_map:
                node.func.id = self.class_replace_map[node.func.id]

        if isinstance(node.func, ast.Attribute):
            if node.func.attr in self.async_methods:
                if getattr(node.func.value, "id", None) == "self":
                    return ast.Await(value=node)

        return self.generic_visit(node)
