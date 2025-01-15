from typing import Optional, Sequence, Any

from qdrant_client.http import models
from qdrant_client.embed.models import NumericVector
from qdrant_client.fastembed_common import (
    TextEmbedding,
    SparseTextEmbedding,
    LateInteractionTextEmbedding,
    ImageEmbedding,
    SUPPORTED_EMBEDDING_MODELS,
    SUPPORTED_SPARSE_EMBEDDING_MODELS,
    _LATE_INTERACTION_EMBEDDING_MODELS,
    OnnxProvider,
    ImageInput,
)


class Embedder:
    def __init__(self) -> None:
        self.embedding_models: dict[str, "TextEmbedding"] = {}
        self.sparse_embedding_models: dict[str, "SparseTextEmbedding"] = {}
        self.late_interaction_embedding_models: dict[str, "LateInteractionTextEmbedding"] = {}
        self.image_embedding_models: dict[str, "ImageEmbedding"] = {}

    def get_or_init_model(
        self,
        model_name: str,
        cache_dir: Optional[str] = None,
        threads: Optional[int] = None,
        providers: Optional[Sequence["OnnxProvider"]] = None,
        cuda: bool = False,
        device_ids: Optional[list[int]] = None,
        **kwargs: Any,
    ) -> TextEmbedding:
        self.embedding_models[model_name] = TextEmbedding(
            model_name=model_name,
            cache_dir=cache_dir,
            threads=threads,
            providers=providers,
            cuda=cuda,
            device_ids=device_ids,
            **kwargs,
        )
        return self.embedding_models[model_name]

    def get_or_init_sparse_model(
        self,
        model_name: str,
        cache_dir: Optional[str] = None,
        threads: Optional[int] = None,
        providers: Optional[Sequence["OnnxProvider"]] = None,
        cuda: bool = False,
        device_ids: Optional[list[int]] = None,
        **kwargs: Any,
    ) -> SparseTextEmbedding:
        self.sparse_embedding_models[model_name] = SparseTextEmbedding(
            model_name=model_name,
            cache_dir=cache_dir,
            threads=threads,
            providers=providers,
            cuda=cuda,
            device_ids=device_ids,
            **kwargs,
        )
        return self.sparse_embedding_models[model_name]

    def get_or_init_late_interaction_model(
        self,
        model_name: str,
        cache_dir: Optional[str] = None,
        threads: Optional[int] = None,
        providers: Optional[Sequence["OnnxProvider"]] = None,
        cuda: bool = False,
        device_ids: Optional[list[int]] = None,
        **kwargs: Any,
    ) -> LateInteractionTextEmbedding:
        self.late_interaction_embedding_models[model_name] = LateInteractionTextEmbedding(
            model_name=model_name,
            cache_dir=cache_dir,
            threads=threads,
            providers=providers,
            cuda=cuda,
            device_ids=device_ids,
            **kwargs,
        )
        return self.late_interaction_embedding_models[model_name]

    def get_or_init_image_model(
        self,
        model_name: str,
        cache_dir: Optional[str] = None,
        threads: Optional[int] = None,
        providers: Optional[Sequence["OnnxProvider"]] = None,
        cuda: bool = False,
        device_ids: Optional[list[int]] = None,
        **kwargs: Any,
    ) -> ImageEmbedding:
        self.image_embedding_models[model_name] = ImageEmbedding(
            model_name=model_name,
            cache_dir=cache_dir,
            threads=threads,
            providers=providers,
            cuda=cuda,
            device_ids=device_ids,
            **kwargs,
        )
        return self.image_embedding_models[model_name]

    def embed(
        self,
        model_name: str,
        texts: Optional[list[str]] = None,
        images: Optional[list[ImageInput]] = None,
        is_query: bool = False,
        batch_size: int = 32,
        **options: Any,
    ) -> NumericVector:
        if (texts is None) is (images is None):
            raise ValueError("Either documents or images should be provided")

        if model_name in SUPPORTED_EMBEDDING_MODELS:
            embedding_model_inst = self.get_or_init_model(model_name=model_name, **options or {})
            if not is_query:
                embeddings = [
                    embedding.tolist()
                    for embedding in embedding_model_inst.embed(
                        documents=texts, batch_size=batch_size
                    )
                ]
            else:
                embeddings = [
                    embedding.tolist()
                    for embedding in embedding_model_inst.query_embed(query=texts)
                ]
        elif model_name in SUPPORTED_SPARSE_EMBEDDING_MODELS.keys():
            embedding_model_inst = self.get_or_init_sparse_model(
                model_name=model_name, **options or {}
            )
            if not is_query:
                embeddings = [
                    models.SparseVector(
                        indices=sparse_embedding.indices.tolist(),
                        values=sparse_embedding.values.tolist(),
                    )
                    for sparse_embedding in embedding_model_inst.embed(
                        documents=texts, batch_size=batch_size
                    )
                ]
            else:
                embeddings = [
                    models.SparseVector(
                        indices=sparse_embedding.indices.tolist(),
                        values=sparse_embedding.values.tolist(),
                    )
                    for sparse_embedding in embedding_model_inst.query_embed(query=texts)
                ]

        elif model_name in _LATE_INTERACTION_EMBEDDING_MODELS:
            embedding_model_inst = self.get_or_init_late_interaction_model(
                model_name=model_name, **options or {}
            )
            if not is_query:
                embeddings = [
                    embedding.tolist()
                    for embedding in embedding_model_inst.embed(
                        documents=texts, batch_size=batch_size
                    )
                ]
            else:
                embeddings = [
                    embedding.tolist()
                    for embedding in embedding_model_inst.query_embed(query=texts)
                ]
        else:
            embedding_model_inst = self.get_or_init_image_model(
                model_name=model_name, **options or {}
            )
            embeddings = [
                embedding.tolist()
                for embedding in embedding_model_inst.embed(images=images, batch_size=batch_size)
            ]

        return embeddings
