# type: ignore

import inspect
from typing import Dict, List, Optional

from tools.async_client_generator.async_client_base import AsyncQdrantBase
from tools.async_client_generator.base_generator import BaseGenerator
from tools.async_client_generator.transformers import (
    ClassDefTransformer,
    ImportFromTransformer,
    ImportTransformer,
)
from tools.async_client_generator.transformers.fastembed import (
    FastembedCallTransformer,
    FastembedFunctionDefTransformer,
)


class FastembedGenerator(BaseGenerator):
    def __init__(
        self,
        keep_sync: Optional[List[str]] = None,
        class_replace_map: Optional[Dict[str, str]] = None,
        import_replace_map: Optional[Dict[str, str]] = None,
    ):
        super().__init__()
        self._async_methods = None
        self.transformers.append(FastembedCallTransformer(self.async_methods))
        self.transformers.append(ClassDefTransformer(class_replace_map))
        self.transformers.append(ImportTransformer(import_replace_map))
        self.transformers.append(ImportFromTransformer(import_replace_map))
        self.transformers.append(FastembedFunctionDefTransformer(keep_sync))

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

    with open(CLIENT_DIR / "qdrant_fastembed.py", "r") as source_file:
        code = source_file.read()

    generator = FastembedGenerator(
        keep_sync=[
            "__init__",
            "set_model",
        ],
        class_replace_map={
            "QdrantBase": "AsyncQdrantBase",
            "QdrantFastembedMixin": "AsyncQdrantFastembedMixin",
        },
        import_replace_map={
            "qdrant_client.client_base": "qdrant_client.async_client_base",
            "QdrantBase": "AsyncQdrantBase",
        },
    )
    modified_code = generator.generate(code)

    with open(CODE_DIR / "async_qdrant_fastembed.py", "w") as target_file:
        target_file.write(modified_code)
