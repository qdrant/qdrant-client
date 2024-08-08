import inspect
from typing import Dict, List, Optional

from tools.async_client_generator.async_client_base import AsyncQdrantBase
from tools.async_client_generator.base_generator import BaseGenerator
from tools.async_client_generator.transformers import (
    CallTransformer,
    ClassDefTransformer,
    ImportFromTransformer,
    ImportTransformer,
    NameTransformer,
)
from tools.async_client_generator.transformers.client import (
    ClientFunctionDefTransformer,
)


class ClientGenerator(BaseGenerator):
    def __init__(
        self,
        keep_sync: Optional[List[str]] = None,
        class_replace_map: Optional[Dict[str, str]] = None,
        import_replace_map: Optional[Dict[str, str]] = None,
        exclude_methods: Optional[List[str]] = None,
    ):
        super().__init__()
        self._async_methods: Optional[List[str]] = None

        self.transformers.append(ImportTransformer(import_replace_map=import_replace_map))
        self.transformers.append(ImportFromTransformer(import_replace_map=import_replace_map))
        self.transformers.append(
            ClientFunctionDefTransformer(
                keep_sync=keep_sync,
                class_replace_map=class_replace_map,
                exclude_methods=exclude_methods,
                async_methods=self.async_methods,
            )
        )
        self.transformers.append(ClassDefTransformer(class_replace_map=class_replace_map))

        # call_transformer should be after function_def_transformer
        self.transformers.append(
            CallTransformer(class_replace_map=class_replace_map, async_methods=self.async_methods)
        )
        # name_transformer should be after function_def, class_def and ann_assign transformers
        self.transformers.append(
            NameTransformer(
                class_replace_map=class_replace_map,
                import_replace_map=import_replace_map,
            )
        )

    @property
    def async_methods(self) -> List[str]:
        if self._async_methods is None:
            self._async_methods = self.get_async_methods(AsyncQdrantBase)
        return self._async_methods

    @staticmethod
    def get_async_methods(class_obj: type) -> List[str]:
        async_methods = []
        for name, method in inspect.getmembers(class_obj):
            if inspect.iscoroutinefunction(method):
                async_methods.append(name)
        return async_methods


if __name__ == "__main__":
    from tools.async_client_generator.config import CLIENT_DIR, CODE_DIR

    with open(CLIENT_DIR / "qdrant_client.py", "r") as source_file:
        code = source_file.read()

    generator = ClientGenerator(
        class_replace_map={
            "QdrantBase": "AsyncQdrantBase",
            "QdrantFastembedMixin": "AsyncQdrantFastembedMixin",
            "QdrantClient": "AsyncQdrantClient",
            "QdrantRemote": "AsyncQdrantRemote",
            "QdrantLocal": "AsyncQdrantLocal",
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
            "qdrant_client.local.qdrant_local": "qdrant_client.local.async_qdrant_local",
            "QdrantLocal": "AsyncQdrantLocal",
        },
        exclude_methods=[
            "__del__",
            "migrate",
            "async_grpc_collections",
            "async_grpc_points",
            "async_grpc_root",
        ],
    )

    modified_code = generator.generate(code)

    with open(CODE_DIR / "async_qdrant_client.py", "w") as target_file:
        target_file.write(modified_code)
