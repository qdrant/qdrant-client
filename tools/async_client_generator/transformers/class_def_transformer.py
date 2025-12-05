import ast


class ClassDefTransformer(ast.NodeTransformer):
    def __init__(self, class_replace_map: dict[str, str] | None):
        self.class_replace_map = class_replace_map if class_replace_map is not None else {}

    def visit_ClassDef(self, node: ast.ClassDef) -> ast.AST:
        # update class name
        for old_value, new_value in self.class_replace_map.items():
            node.name = node.name.replace(old_value, new_value)

        # update parent classes names
        for base in node.bases:
            for old_value, new_value in self.class_replace_map.items():
                if hasattr(base, "id"):
                    base.id = base.id.replace(old_value, new_value)
        return self.generic_visit(node)
