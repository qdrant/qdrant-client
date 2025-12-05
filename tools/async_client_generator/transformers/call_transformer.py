import ast


class CallTransformer(ast.NodeTransformer):
    def __init__(
        self,
        class_replace_map: dict[str, str] | None = None,
        async_methods: list[str] | None = None,
    ):
        self.class_replace_map = class_replace_map if class_replace_map is not None else {}
        self.async_methods = async_methods if async_methods is not None else []

    def visit_Call(self, node: ast.Call) -> ast.AST | ast.Await:
        if isinstance(node.func, ast.Name):
            if node.func.id in self.class_replace_map:
                node.func.id = self.class_replace_map[node.func.id]

        if isinstance(node.func, ast.Attribute):
            if node.func.attr in self.async_methods:
                return ast.Await(value=node)

        return self.generic_visit(node)
