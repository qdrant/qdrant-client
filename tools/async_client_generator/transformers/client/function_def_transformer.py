import ast
from typing import Dict, List, Optional

from ...transformers import FunctionDefTransformer


class ClientFunctionDefTransformer(FunctionDefTransformer):
    def __init__(
        self,
        keep_sync: Optional[List[str]] = None,
        class_replace_map: Optional[Dict[str, str]] = None,
        exclude_methods: Optional[List[str]] = None,
        rename_methods: Optional[Dict[str, str]] = None,
        async_methods: Optional[List[str]] = None,
    ):
        super().__init__(keep_sync)
        self.class_replace_map = class_replace_map if class_replace_map is not None else {}
        self.exclude_methods = exclude_methods if exclude_methods is not None else []
        self.rename_methods = rename_methods if rename_methods is not None else {}
        self.async_methods = async_methods if async_methods is not None else []

    def _keep_sync(self, name: str) -> bool:
        return name in self.keep_sync or name not in self.async_methods

    def visit_FunctionDef(self, sync_node: ast.FunctionDef) -> Optional[ast.AST]:
        if sync_node.name in self.exclude_methods:
            return None

        if sync_node.name == "__init__":
            return self.generate_init(sync_node)

        return super().visit_FunctionDef(sync_node)

    def generate_init(self, sync_node: ast.FunctionDef) -> ast.AST:
        def traverse(node):
            assignment_nodes = []

            if isinstance(node, ast.Assign):
                assignment_nodes.append(node)
            for field_name, field_value in ast.iter_fields(node):
                if isinstance(field_value, ast.AST):
                    assignment_nodes.extend(traverse(field_value))
                elif isinstance(field_value, list):
                    for item in field_value:
                        if isinstance(item, ast.AST):
                            assignment_nodes.extend(traverse(item))
            return assignment_nodes

        def unwrap_orelse_assignment(assign_node: ast.Assign) -> Optional[ast.Assign]:
            for target in assign_node.targets:
                if isinstance(target, ast.Attribute) and target.attr == "_client":
                    if isinstance(assign_node.value, ast.Call):
                        if assign_node.value.func.id in self.class_replace_map:
                            assign_node.value.func.id = self.class_replace_map[
                                assign_node.value.func.id
                            ]
                            return assign_node

        args, defaults = [sync_node.args.args[0]], []
        for arg, default in zip(sync_node.args.args[1:], sync_node.args.defaults):
            if arg.arg not in ("location", "path"):
                args.append(arg)
                defaults.append(default)
        sync_node.args.args = args
        sync_node.args.defaults = defaults

        for i, child_node in enumerate(sync_node.body):
            if isinstance(child_node, ast.If):
                orelse_assignment_nodes = traverse(child_node)
                assignments = list(
                    filter(
                        lambda x: x,
                        [
                            unwrap_orelse_assignment(assign_node)
                            for assign_node in orelse_assignment_nodes
                        ],
                    )
                )
                if len(assignments) == 1:
                    sync_node.body[i] = assignments[0]
                    break
        return self.generic_visit(sync_node)
