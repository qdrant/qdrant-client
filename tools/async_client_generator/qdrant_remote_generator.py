import ast
import inspect
from typing import Optional

from experiments.new_gen.code.async_client_base import AsyncQdrantBase
from qdrant_client.grpc import CollectionsStub, PointsStub, SnapshotsStub
from qdrant_client.http import AsyncApiClient
from qdrant_client.http.api.cluster_api import AsyncClusterApi
from qdrant_client.http.api.collections_api import AsyncCollectionsApi
from qdrant_client.http.api.points_api import AsyncPointsApi
from qdrant_client.http.api.service_api import AsyncServiceApi
from qdrant_client.http.api.snapshots_api import AsyncSnapshotsApi


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

    @staticmethod
    def _get_grpc_methods(grpc_stub_class):
        init_source = inspect.getsource(grpc_stub_class)

        # Parse the source code using ast
        parsed = ast.parse(init_source)

        # Extract attribute names
        field_names = []
        for node in ast.walk(parsed):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if (
                        isinstance(target, ast.Attribute)
                        and isinstance(target.value, ast.Name)
                        and target.value.id == "self"
                    ):
                        field_name = target.attr
                        field_names.append(field_name)
        return field_names

    @property
    def async_methods(self):
        if self._async_methods is None:
            self._async_methods = []
            self._async_methods.extend(self.get_async_methods(AsyncQdrantBase))
            self._async_methods.extend(self.get_async_methods(AsyncClusterApi))
            self._async_methods.extend(self.get_async_methods(AsyncCollectionsApi))
            self._async_methods.extend(self.get_async_methods(AsyncPointsApi))
            self._async_methods.extend(self.get_async_methods(AsyncServiceApi))
            self._async_methods.extend(self.get_async_methods(AsyncSnapshotsApi))
            self._async_methods.extend(self.get_async_methods(AsyncApiClient))

            self._async_methods.extend(self._get_grpc_methods(PointsStub))
            self._async_methods.extend(self._get_grpc_methods(SnapshotsStub))
            self._async_methods.extend(self._get_grpc_methods(CollectionsStub))

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
            method_name = node.func.attr
            if method_name in self.async_methods:
                return ast.Await(value=node)

        return self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign):
        for old_value, new_value in self.class_replace_map.items():
            if isinstance(node.annotation, ast.Name):
                node.annotation.id = node.annotation.id.replace(old_value, new_value)
        return self.generic_visit(node)

    @staticmethod
    def override_init(sync_node):
        kick_assignments = []
        for child_node in sync_node.body:
            if isinstance(child_node, ast.Assign):
                for target in child_node.targets:
                    if isinstance(target, ast.Attribute) and hasattr(target, "attr"):
                        if "aio" in target.attr:
                            kick_assignments.append(child_node)
            if isinstance(child_node, ast.AnnAssign) and hasattr(child_node.target, "attr"):
                if "aio" in child_node.target.attr:
                    kick_assignments.append(child_node)
        sync_node.body = [node for node in sync_node.body if node not in kick_assignments]
        return sync_node

    @staticmethod
    def override_close():
        code = """
async def close(self, grpc_grace: Optional[float] = None, **kwargs: Any) -> None:
    if hasattr(self, "_grpc_channel") and self._grpc_channel is not None:
        try:
            await self._grpc_channel.close(grace=grpc_grace)
        except AttributeError:
            logging.warning(
                "Unable to close grpc_channel. Connection was interrupted on the server side"
            )
        except RuntimeError:
            pass

    try:
        await self.http.aclose()
    except Exception:
        logging.warning(
            "Unable to close http connection. Connection was interrupted on the server side"
        )

    self._closed = True
            """

        parsed_code = ast.parse(code)
        return parsed_code.body[0]

    def visit_FunctionDef(self, sync_node: ast.FunctionDef):
        if sync_node.name in self.exclude_methods:
            return None

        if sync_node.name == "__init__":
            sync_node = self.override_init(sync_node)
            return self.generic_visit(sync_node)

        if sync_node.name == "close":
            return self.override_close()

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

        for i, alias in enumerate(node.names):
            if hasattr(alias, "name"):
                for old_value, new_value in self.import_replace_map.items():
                    alias.name = alias.name.replace(old_value, new_value)
                if alias.name == "get_async_channel":
                    alias.asname = "get_channel"
        node.names = [alias for alias in node.names if alias.name != "get_channel"]

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


class RemoteAsyncAwaitTransformer(AsyncAwaitTransformer):
    def _keep_sync(self, name: str) -> bool:
        return name in self.keep_sync or name not in self.async_methods


if __name__ == "__main__":
    from .config import CODE_DIR

    with open(CODE_DIR / "qdrant_remote.py", "r") as source_file:
        code = source_file.read()

    # Parse the code into an AST
    parsed_code = ast.parse(code)

    await_transformer = RemoteAsyncAwaitTransformer(
        class_replace_map={
            "QdrantBase": "AsyncQdrantBase",
            "QdrantFastembedMixin": "AsyncQdrantFastembedMixin",
            "QdrantClient": "AsyncQdrantClient",
            "QdrantRemote": "AsyncQdrantRemote",
        },
        import_replace_map={
            "qdrant_client.client_base": "qdrant_client.async_client_base",
            "QdrantBase": "AsyncQdrantBase",
            "QdrantRemote": "AsyncQdrantRemote",
            "ApiClient": "AsyncApiClient",
            "SyncApis": "AsyncApis",
        },
        exclude_methods=[
            "__del__",
            "migrate",
            "async_grpc_collections",
            "async_grpc_points",
            "async_grpc_snapshots",
            "_init_async_grpc_points_client",
            "_init_async_grpc_collections_client",
            "_init_async_grpc_snapshots_client",
            "_init_async_grpc_channel",
        ],
    )
    modified_code_ast = await_transformer.visit(parsed_code)
    modified_code = ast.unparse(modified_code_ast)

    with open(CODE_DIR / "async_qdrant_remote.py", "w") as target_file:
        target_file.write(modified_code)
