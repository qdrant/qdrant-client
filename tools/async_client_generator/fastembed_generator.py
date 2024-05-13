import inspect
from typing import Dict, List, Optional

from qdrant_client.qdrant_fastembed import QdrantFastembedMixin
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
        self._async_methods: Optional[List[str]] = None
        self.transformers.append(
            FastembedCallTransformer(
                async_methods=self.async_methods(exclude_list=keep_sync or [])
            )
        )
        self.transformers.append(ClassDefTransformer(class_replace_map=class_replace_map))
        self.transformers.append(ImportTransformer(import_replace_map=import_replace_map))
        self.transformers.append(ImportFromTransformer(import_replace_map=import_replace_map))
        self.transformers.append(FastembedFunctionDefTransformer(keep_sync=keep_sync))

    def async_methods(self, exclude_list: List[str]) -> List[str]:
        if self._async_methods is None:
            self._async_methods = self.get_async_methods(AsyncQdrantBase)
            self._async_methods.extend(
                self.get_self_async_methods(
                    QdrantFastembedMixin, [*self._async_methods, *exclude_list]
                )
            )
        return self._async_methods

    @staticmethod
    def get_async_methods(class_obj: type) -> List[str]:
        async_methods = []
        for name, method in inspect.getmembers(class_obj):
            if inspect.iscoroutinefunction(method):
                async_methods.append(name)
        return async_methods

    @staticmethod
    def get_self_async_methods(class_obj: type, exclude_list: list[str]) -> List[str]:
        async_methods = []
        for name, method in inspect.getmembers(class_obj):
            if inspect.isfunction(method) and name not in exclude_list:
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
            "set_image_model",
            "set_sparse_model",
            "get_vector_field_name",
            "get_image_vector_field_name",
            "get_sparse_vector_field_name",
            "get_fastembed_vector_params",
            "get_fastembed_sparse_vector_params",
            "embedding_model_name",
            "sparse_embedding_model_name",
            "image_embedding_model_name",
            "_import_fastembed",
            "_get_text_model_params",
            "_get_image_model_params",
            "_get_or_init_text_model",
            "_get_or_init_image_model",
            "_get_or_init_sparse_model",
            "_embed_documents",
            "_embed_images",
            "_sparse_embed_documents",
            "_scored_points_to_query_responses",
            "_points_iterator",
            "_validate_collection_info",
            "vector_field_from_model",
            "_extract_embeddings",
            "_build_image_batch_requests",
            "_build_text_batch_requests",
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
