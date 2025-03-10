import ast
import inspect
from typing import Optional

from qdrant_client.grpc import CollectionsStub, PointsStub, SnapshotsStub, QdrantStub
from qdrant_client.http import AsyncApiClient
from qdrant_client.http.api.distributed_api import AsyncDistributedApi
from qdrant_client.http.api.aliases_api import AsyncAliasesApi
from qdrant_client.http.api.indexes_api import AsyncIndexesApi
from qdrant_client.http.api.search_api import AsyncSearchApi
from qdrant_client.http.api.collections_api import AsyncCollectionsApi
from qdrant_client.http.api.points_api import AsyncPointsApi
from qdrant_client.http.api.service_api import AsyncServiceApi
from qdrant_client.http.api.snapshots_api import AsyncSnapshotsApi
from tools.async_client_generator.async_client_base import AsyncQdrantBase
from tools.async_client_generator.base_generator import BaseGenerator
from tools.async_client_generator.transformers import (
    CallTransformer,
    ClassDefTransformer,
    ImportTransformer,
    NameTransformer,
)
from tools.async_client_generator.transformers.remote import (
    RemoteFunctionDefTransformer,
    RemoteImportFromTransformer,
)


class RemoteGenerator(BaseGenerator):
    def __init__(
        self,
        keep_sync: Optional[list[str]] = None,
        class_replace_map: Optional[dict] = None,
        import_replace_map: Optional[dict] = None,
        exclude_methods: Optional[list[str]] = None,
        rename_methods: Optional[dict[str, str]] = None,
    ):
        super().__init__()
        self._async_methods: Optional[list[str]] = None

        self.transformers.append(
            RemoteImportFromTransformer(import_replace_map=import_replace_map)
        )
        self.transformers.append(ClassDefTransformer(class_replace_map=class_replace_map))
        self.transformers.append(
            CallTransformer(class_replace_map=class_replace_map, async_methods=self.async_methods)
        )
        self.transformers.append(ImportTransformer(import_replace_map=import_replace_map))
        self.transformers.append(
            RemoteFunctionDefTransformer(
                keep_sync=keep_sync,
                exclude_methods=exclude_methods,
                async_methods=self.async_methods,
            )
        )
        self.transformers.append(
            NameTransformer(
                class_replace_map=class_replace_map,
                import_replace_map=import_replace_map,
                rename_methods=rename_methods,
            )
        )

    @staticmethod
    def _get_grpc_methods(grpc_stub_class: type) -> list[str]:
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
    def async_methods(self) -> list[str]:
        if self._async_methods is None:
            self._async_methods = []
            for cls_ in (
                AsyncQdrantBase,
                AsyncDistributedApi,
                AsyncCollectionsApi,
                AsyncPointsApi,
                AsyncServiceApi,
                AsyncSnapshotsApi,
                AsyncIndexesApi,
                AsyncAliasesApi,
                AsyncSearchApi,
                AsyncApiClient,
            ):
                self._async_methods.extend(self.get_async_methods(cls_))

            for cls_ in (PointsStub, SnapshotsStub, CollectionsStub, QdrantStub):
                self._async_methods.extend(self._get_grpc_methods(cls_))

        return self._async_methods

    @staticmethod
    def get_async_methods(class_obj: type) -> list[str]:
        async_methods = []
        for name, method in inspect.getmembers(class_obj):
            if inspect.iscoroutinefunction(method):
                async_methods.append(name)
        return async_methods


if __name__ == "__main__":
    from tools.async_client_generator.config import CLIENT_DIR, CODE_DIR

    with open(CLIENT_DIR / "qdrant_remote.py", "r") as source_file:
        code = source_file.read()

    generator = RemoteGenerator(
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
        ],
    )

    modified_code = generator.generate(code)

    with open(CODE_DIR / "async_qdrant_remote.py", "w") as target_file:
        target_file.write(modified_code)
