import ast


class FastembedCallTransformer(ast.NodeTransformer):
    def __init__(self, async_methods: list[str] | None = None):
        self.async_methods = async_methods if async_methods is not None else []

    def visit_Call(self, node: ast.Call) -> ast.AST | ast.Await:
        if isinstance(node.func, ast.Attribute):
            if isinstance(node.func.value, ast.Name):
                if node.func.attr in self.async_methods:
                    return ast.Await(value=node)
        return self.generic_visit(node)
