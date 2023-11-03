import inspect
from typing import List, Optional

from tools.async_client_generator.async_client_base import AsyncQdrantBase
from tools.async_client_generator.base_generator import BaseGenerator
from tools.async_client_generator.transformers import (
    ClassDefTransformer,
    ImportFromTransformer,
    ImportTransformer,
    NameTransformer,
)
from tools.async_client_generator.transformers.client import (
    ClientFunctionDefTransformer,
)
from tools.async_client_generator.transformers.local import LocalCallTransformer


class LocalGenerator(BaseGenerator):
    def __init__(
        self,
        keep_sync: Optional[List[str]] = None,
        class_replace_map: Optional[dict] = None,
        import_replace_map: Optional[dict] = None,
        exclude_methods: Optional[List[str]] = None,
    ):
        super().__init__()
        self._async_methods: Optional[List[str]] = None

        self.transformers.append(ImportFromTransformer(import_replace_map=import_replace_map))
        self.transformers.append(ClassDefTransformer(class_replace_map=class_replace_map))
        self.transformers.append(
            LocalCallTransformer(
                class_replace_map=class_replace_map, async_methods=self.async_methods
            )
        )
        self.transformers.append(ImportTransformer(import_replace_map=import_replace_map))
        self.transformers.append(
            ClientFunctionDefTransformer(
                keep_sync=keep_sync,
                exclude_methods=exclude_methods,
                async_methods=self.async_methods,
            )
        )
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

    with open(CLIENT_DIR / "local" / "qdrant_local.py", "r") as source_file:
        code = source_file.read()

    generator = LocalGenerator(
        class_replace_map={
            "QdrantBase": "AsyncQdrantBase",
            "QdrantLocal": "AsyncQdrantLocal",
        },
        import_replace_map={
            "qdrant_client.client_base": "qdrant_client.async_client_base",
            "QdrantBase": "AsyncQdrantBase",
            "QdrantLocal": "AsyncQdrantLocal",
        },
        exclude_methods=[
            "migrate",
        ],
    )

    modified_code = generator.generate(code)

    with open(CODE_DIR / "async_qdrant_local.py", "w") as target_file:
        target_file.write(modified_code)
