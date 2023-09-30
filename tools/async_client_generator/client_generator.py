# type: ignore

import ast
import inspect
from typing import Optional

from qdrant_client.async_client_base import AsyncQdrantBase


class AsyncAwaitTransformer(ast.NodeTransformer):
    def __init__(
        self,
        keep_sync: Optional[list[str]] = None,
        class_replace_map: Optional[dict] = None,
        import_replace_map: Optional[dict] = None,
        exclude_methods: Optional[list[str]] = None,
        rename_methods: Optional[dict[str, str]] = None,
    ):
        self._async_methods = None
        self.keep_sync = keep_sync if keep_sync is not None else []
        self.class_replace_map = class_replace_map if class_replace_map is not None else {}
        self.import_replace_map = import_replace_map if import_replace_map is not None else {}
        self.exclude_methods = exclude_methods if exclude_methods is not None else []
        self.rename_methods = rename_methods if rename_methods is not None else {}

    def _keep_sync(self, name: str) -> bool:
        return name in self.keep_sync

    @property
    def async_methods(self):
        if self._async_methods is None:
            self._async_methods = self.get_async_methods(AsyncQdrantBase)
        return self._async_methods

    @staticmethod
    def get_async_methods(class_obj):
        async_methods = []
        for name, method in inspect.getmembers(class_obj):
            if inspect.iscoroutinefunction(method):
                async_methods.append(name)
        return async_methods

    def visit_Name(self, node: ast.Name):
        if node.id in self.class_replace_map:
            node.id = self.class_replace_map[node.id]
        elif node.id in self.import_replace_map:
            node.id = self.import_replace_map[node.id]
        elif node.id in self.rename_methods:
            node.id = self.rename_methods[node.id]
        return self.generic_visit(node)

    def visit_Call(self, node):
        if isinstance(node.func, ast.Name):
            if node.func.id in self.class_replace_map:
                node.func.id = self.class_replace_map[node.func.id]

        if isinstance(node.func, ast.Attribute):
            if node.func.attr in self.async_methods:
                return ast.Await(value=node)

        return self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign):
        for old_value, new_value in self.class_replace_map.items():
            if isinstance(node.annotation, ast.Name):
                node.annotation.id = node.annotation.id.replace(old_value, new_value)
        return self.generic_visit(node)

    def visit_FunctionDef(self, sync_node: ast.FunctionDef):
        if sync_node.name in self.exclude_methods:
            return None

        if sync_node.name == "__init__":
            return self.generate_init(sync_node)

        if self._keep_sync(sync_node.name):
            return self.generic_visit(sync_node)

        async_node = ast.AsyncFunctionDef(
            name=sync_node.name,
            args=sync_node.args,
            body=sync_node.body,
            decorator_list=sync_node.decorator_list,
            returns=sync_node.returns,
            type_comment=sync_node.type_comment,
        )
        async_node.lineno = sync_node.lineno
        async_node.col_offset = sync_node.col_offset
        async_node.end_lineno = sync_node.end_lineno
        async_node.end_col_offset = sync_node.end_col_offset
        return self.generic_visit(async_node)

    def visit_Import(self, node: ast.Import):
        for old_value, new_value in self.import_replace_map.items():
            for alias in node.names:
                alias.name = alias.name.replace(old_value, new_value)
        return self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom):
        # update module name
        for old_value, new_value in self.import_replace_map.items():
            node.module = node.module.replace(old_value, new_value)

        # update imported item name
        for alias in node.names:
            if hasattr(alias, "name"):
                for old_value, new_value in self.import_replace_map.items():
                    alias.name = alias.name.replace(old_value, new_value)

        return self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef):
        # update class name
        for old_value, new_value in self.class_replace_map.items():
            node.name = node.name.replace(old_value, new_value)

        # update parent classes names
        for base in node.bases:
            for old_value, new_value in self.class_replace_map.items():
                base.id = base.id.replace(old_value, new_value)
        return self.generic_visit(node)

    def generate_init(self, sync_node: ast.FunctionDef):
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

        def unwrap_orelse_assignment(assign_node: ast.Assign):
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


class ClientAsyncAwaitTransformer(AsyncAwaitTransformer):
    def _keep_sync(self, name: str) -> bool:
        return name in self.keep_sync or name not in self.async_methods


if __name__ == "__main__":
    from .config import CODE_DIR

    with open(CODE_DIR / "qdrant_client.py", "r") as source_file:
        code = source_file.read()

    parsed_code = ast.parse(code)

    await_transformer = ClientAsyncAwaitTransformer(
        class_replace_map={
            "QdrantBase": "AsyncQdrantBase",
            "QdrantFastembedMixin": "AsyncQdrantFastembedMixin",
            "QdrantClient": "AsyncQdrantClient",
            "QdrantRemote": "AsyncQdrantRemote",
        },
        import_replace_map={
            "qdrant_client.client_base": "qdrant_client.async_client_base",
            "QdrantBase": "AsyncQdrantBase",
            "QdrantFastembedMixin": "AsyncQdrantFastembedMixin",
            "qdrant_client.qdrant_fastembed": "qdrant_client.async_qdrant_fastembed",
            "qdrant_client.qdrant_remote": "qdrant_client.async_qdrant_remote",
            "QdrantRemote": "AsyncQdrantRemote",
            "ApiClient": "AsyncApiClient",
            "SyncApis": "AsyncApis",
        },
        exclude_methods=[
            "__del__",
            "migrate",
            "async_grpc_collections",
            "async_grpc_points",
        ],
    )
    modified_code_ast = await_transformer.visit(parsed_code)
    modified_code = ast.unparse(modified_code_ast)

    with open("async_qdrant_client.py", "w") as target_file:
        target_file.write(modified_code)
