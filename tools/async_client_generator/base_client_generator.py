# type: ignore
import ast
from typing import Optional


class AsyncAwaitTransformer(ast.NodeTransformer):
    def __init__(
        self,
        keep_sync: Optional[list[str]] = None,
        class_replace_map: Optional[dict] = None,
        import_replace_map: Optional[dict] = None,
        constant_replace_map: Optional[dict] = None,
    ):
        self._async_methods = None
        self.keep_sync = keep_sync if keep_sync is not None else []
        self.class_replace_map = class_replace_map if class_replace_map is not None else {}
        self.import_replace_map = import_replace_map if import_replace_map is not None else {}
        self.constant_replace_map = (
            constant_replace_map if constant_replace_map is not None else {}
        )

    def _keep_sync(self, name: str) -> bool:
        return name in self.keep_sync

    def visit_FunctionDef(self, sync_node: ast.FunctionDef):
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

    def visit_ClassDef(self, node: ast.ClassDef):
        # update class name
        for old_value, new_value in self.class_replace_map.items():
            node.name = node.name.replace(old_value, new_value)
        return self.generic_visit(node)

    def visit_Constant(self, node: ast.arg):
        for old_value, new_value in self.constant_replace_map.items():
            if isinstance(node.value, str):
                node.value = node.value.replace(old_value, new_value)
        return self.generic_visit(node)


if __name__ == "__main__":
    from .config import CODE_DIR

    with open(CODE_DIR / "client_base.py", "r") as source_file:
        code = source_file.read()

    # Parse the code into an AST
    parsed_code = ast.parse(code)

    await_transformer = AsyncAwaitTransformer(
        keep_sync=["__init__", "upload_records", "upload_collection", "migrate"],
        class_replace_map={"QdrantBase": "AsyncQdrantBase"},
        constant_replace_map={"QdrantBase": "AsyncQdrantBase"},
    )
    modified_code_ast = await_transformer.visit(parsed_code)
    modified_code = ast.unparse(modified_code_ast)

    with open("async_client_base.py", "w") as target_file:
        target_file.write(modified_code)
