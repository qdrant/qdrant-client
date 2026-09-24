import ast

from tools.async_client_generator.transformers import FunctionDefTransformer


class ClientFunctionDefTransformer(FunctionDefTransformer):
    def __init__(
        self,
        keep_sync: list[str] | None = None,
        class_replace_map: dict[str, str] | None = None,
        exclude_methods: list[str] | None = None,
        async_methods: list[str] | None = None,
        rename_methods: dict[str, str] | None = None,
    ):
        super().__init__(keep_sync)
        self.class_replace_map = class_replace_map if class_replace_map is not None else {}
        self.exclude_methods = exclude_methods if exclude_methods is not None else []
        self.async_methods = async_methods if async_methods is not None else []
        self.rename_methods = rename_methods if rename_methods is not None else {}

    def _keep_sync(self, name: str) -> bool:
        # Renamed methods (e.g. __enter__ -> __aenter__) have no counterpart on
        # AsyncQdrantBase, so the async_methods lookup would keep them sync.
        if name in self.rename_methods.values():
            return False
        return name in self.keep_sync or name not in self.async_methods

    def visit_FunctionDef(self, sync_node: ast.FunctionDef) -> ast.AST | None:
        if sync_node.name in self.exclude_methods:
            return None

        if sync_node.name in self.rename_methods:
            sync_node.name = self.rename_methods[sync_node.name]
            # A string return annotation ("QdrantClient") is a Constant, so the
            # name-based transformers leave it alone.
            if isinstance(sync_node.returns, ast.Constant) and isinstance(
                sync_node.returns.value, str
            ):
                sync_node.returns.value = self.class_replace_map.get(
                    sync_node.returns.value, sync_node.returns.value
                )

        return super().visit_FunctionDef(sync_node)
