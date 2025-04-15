from collections import defaultdict
from typing import Optional, Sequence, Any, TypeVar, Generic
from pydantic import BaseModel

from qdrant_client.http import models
from qdrant_client.embed.models import NumericVector
from qdrant_client.fastembed_common import (
    OnnxProvider,
    ImageInput,
    TextEmbedding,
    SparseTextEmbedding,
    LateInteractionTextEmbedding,
    LateInteractionMultimodalEmbedding,
    ImageEmbedding,
    SUPPORTED_EMBEDDING_MODELS,
    SUPPORTED_SPARSE_EMBEDDING_MODELS,
    _LATE_INTERACTION_EMBEDDING_MODELS,
    _IMAGE_EMBEDDING_MODELS,
    _LATE_INTERACTION_MULTIMODAL_EMBEDDING_MODELS,
)


T = TypeVar("T")


class ModelInstance(BaseModel, Generic[T], arbitrary_types_allowed=True):  # type: ignore[call-arg]
    model: T
    options: dict[str, Any]
    deprecated: bool = False


class Embedder:
    def __init__(self, threads: Optional[int] = None, **kwargs: Any) -> None:
        self.embedding_models: dict[str, list[ModelInstance[TextEmbedding]]] = defaultdict(list)
        self.sparse_embedding_models: dict[str, list[ModelInstance[SparseTextEmbedding]]] = (
            defaultdict(list)
        )
        self.late_interaction_embedding_models: dict[
            str, list[ModelInstance[LateInteractionTextEmbedding]]
        ] = defaultdict(list)
        self.image_embedding_models: dict[str, list[ModelInstance[ImageEmbedding]]] = defaultdict(
            list
        )
        self.late_interaction_multimodal_embedding_models: dict[
            str, list[ModelInstance[LateInteractionMultimodalEmbedding]]
        ] = defaultdict(list)
        self._threads = threads

    def get_or_init_model(
        self,
        model_name: str,
        cache_dir: Optional[str] = None,
        threads: Optional[int] = None,
        providers: Optional[Sequence["OnnxProvider"]] = None,
        cuda: bool = False,
        device_ids: Optional[list[int]] = None,
        deprecated: bool = False,
        **kwargs: Any,
    ) -> TextEmbedding:
        if model_name not in SUPPORTED_EMBEDDING_MODELS:
            raise ValueError(
                f"Unsupported embedding model: {model_name}. Supported models: {SUPPORTED_EMBEDDING_MODELS}"
            )
        options = {
            "cache_dir": cache_dir,
            "threads": threads or self._threads,
            "providers": providers,
            "cuda": cuda,
            "device_ids": device_ids,
            **kwargs,
        }
        for instance in self.embedding_models[model_name]:
            if (deprecated and instance.deprecated) or (
                not deprecated and instance.options == options
            ):
                return instance.model

        model = TextEmbedding(model_name=model_name, **options)
        model_instance: ModelInstance[TextEmbedding] = ModelInstance(
            model=model, options=options, deprecated=deprecated
        )
        self.embedding_models[model_name].append(model_instance)
        return model

    def get_or_init_sparse_model(
        self,
        model_name: str,
        cache_dir: Optional[str] = None,
        threads: Optional[int] = None,
        providers: Optional[Sequence["OnnxProvider"]] = None,
        cuda: bool = False,
        device_ids: Optional[list[int]] = None,
        deprecated: bool = False,
        **kwargs: Any,
    ) -> SparseTextEmbedding:
        if model_name not in SUPPORTED_SPARSE_EMBEDDING_MODELS:
            raise ValueError(
                f"Unsupported embedding model: {model_name}. Supported models: {SUPPORTED_SPARSE_EMBEDDING_MODELS}"
            )

        options = {
            "cache_dir": cache_dir,
            "threads": threads or self._threads,
            "providers": providers,
            "cuda": cuda,
            "device_ids": device_ids,
            **kwargs,
        }

        for instance in self.sparse_embedding_models[model_name]:
            if (deprecated and instance.deprecated) or (
                not deprecated and instance.options == options
            ):
                return instance.model

        model = SparseTextEmbedding(model_name=model_name, **options)
        model_instance: ModelInstance[SparseTextEmbedding] = ModelInstance(
            model=model, options=options, deprecated=deprecated
        )
        self.sparse_embedding_models[model_name].append(model_instance)
        return model

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
        if model_name not in _LATE_INTERACTION_EMBEDDING_MODELS:
            raise ValueError(
                f"Unsupported embedding model: {model_name}. Supported models: {_LATE_INTERACTION_EMBEDDING_MODELS}"
            )
        options = {
            "cache_dir": cache_dir,
            "threads": threads or self._threads,
            "providers": providers,
            "cuda": cuda,
            "device_ids": device_ids,
            **kwargs,
        }

        for instance in self.late_interaction_embedding_models[model_name]:
            if instance.options == options:
                return instance.model

        model = LateInteractionTextEmbedding(model_name=model_name, **options)
        model_instance: ModelInstance[LateInteractionTextEmbedding] = ModelInstance(
            model=model, options=options
        )
        self.late_interaction_embedding_models[model_name].append(model_instance)
        return model

    def get_or_init_late_interaction_multimodal_model(
        self,
        model_name: str,
        cache_dir: Optional[str] = None,
        threads: Optional[int] = None,
        providers: Optional[Sequence["OnnxProvider"]] = None,
        cuda: bool = False,
        device_ids: Optional[list[int]] = None,
        **kwargs: Any,
    ) -> LateInteractionMultimodalEmbedding:
        if model_name not in _LATE_INTERACTION_MULTIMODAL_EMBEDDING_MODELS:
            raise ValueError(
                f"Unsupported embedding model: {model_name}. Supported models: {_LATE_INTERACTION_MULTIMODAL_EMBEDDING_MODELS}"
            )
        options = {
            "cache_dir": cache_dir,
            "threads": threads or self._threads,
            "providers": providers,
            "cuda": cuda,
            "device_ids": device_ids,
            **kwargs,
        }

        for instance in self.late_interaction_multimodal_embedding_models[model_name]:
            if instance.options == options:
                return instance.model

        model = LateInteractionMultimodalEmbedding(model_name=model_name, **options)
        model_instance: ModelInstance[LateInteractionMultimodalEmbedding] = ModelInstance(
            model=model, options=options
        )
        self.late_interaction_multimodal_embedding_models[model_name].append(model_instance)
        return model

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
        if model_name not in _IMAGE_EMBEDDING_MODELS:
            raise ValueError(
                f"Unsupported embedding model: {model_name}. Supported models: {_IMAGE_EMBEDDING_MODELS}"
            )
        options = {
            "cache_dir": cache_dir,
            "threads": threads or self._threads,
            "providers": providers,
            "cuda": cuda,
            "device_ids": device_ids,
            **kwargs,
        }

        for instance in self.image_embedding_models[model_name]:
            if instance.options == options:
                return instance.model

        model = ImageEmbedding(model_name=model_name, **options)
        model_instance: ModelInstance[ImageEmbedding] = ModelInstance(model=model, options=options)
        self.image_embedding_models[model_name].append(model_instance)
        return model

    def embed(
        self,
        model_name: str,
        texts: Optional[list[str]] = None,
        images: Optional[list[ImageInput]] = None,
        options: Optional[dict[str, Any]] = None,
        is_query: bool = False,
        batch_size: int = 8,
    ) -> NumericVector:
        if (texts is None) is (images is None):
            raise ValueError("Either documents or images should be provided")

        embeddings: NumericVector  # define type for a static type checker
        if texts is not None:
            if model_name in SUPPORTED_EMBEDDING_MODELS:
                embeddings = self._embed_dense_text(
                    texts, model_name, options, is_query, batch_size
                )
            elif model_name in SUPPORTED_SPARSE_EMBEDDING_MODELS:
                embeddings = self._embed_sparse_text(
                    texts, model_name, options, is_query, batch_size
                )
            elif model_name in _LATE_INTERACTION_EMBEDDING_MODELS:
                embeddings = self._embed_late_interaction_text(
                    texts, model_name, options, is_query, batch_size
                )
            elif model_name in _LATE_INTERACTION_MULTIMODAL_EMBEDDING_MODELS:
                embeddings = self._embed_late_interaction_multimodal_text(
                    texts, model_name, options, batch_size
                )
            else:
                raise ValueError(f"Unsupported embedding model: {model_name}")
        else:
            assert (
                images is not None
            )  # just to satisfy mypy which can't infer it from the previous conditions
            if model_name in _IMAGE_EMBEDDING_MODELS:
                embeddings = self._embed_dense_image(images, model_name, options, batch_size)
            elif model_name in _LATE_INTERACTION_MULTIMODAL_EMBEDDING_MODELS:
                embeddings = self._embed_late_interaction_multimodal_image(
                    images, model_name, options, batch_size
                )
            else:
                raise ValueError(f"Unsupported embedding model: {model_name}")

        return embeddings

    def _embed_dense_text(
        self,
        texts: list[str],
        model_name: str,
        options: Optional[dict[str, Any]],
        is_query: bool,
        batch_size: int,
    ) -> list[list[float]]:
        embedding_model_inst = self.get_or_init_model(model_name=model_name, **options or {})

        if not is_query:
            embeddings = [embedding.tolist() for embedding in embedding_model_inst.embed(documents=texts, batch_size=batch_size)]
        else:
            embeddings = [
                embedding.tolist() for embedding in embedding_model_inst.query_embed(query=texts)
            ]
        return embeddings

    def _embed_sparse_text(
        self,
        texts: list[str],
        model_name: str,
        options: Optional[dict[str, Any]],
        is_query: bool,
        batch_size: int,
    ) -> list[models.SparseVector]:
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
        return embeddings

    def _embed_late_interaction_text(
        self,
        texts: list[str],
        model_name: str,
        options: Optional[dict[str, Any]],
        is_query: bool,
        batch_size: int,
    ) -> list[list[list[float]]]:
        embedding_model_inst = self.get_or_init_late_interaction_model(
            model_name=model_name, **options or {}
        )
        if not is_query:
            embeddings = [
                embedding.tolist()
                for embedding in embedding_model_inst.embed(documents=texts, batch_size=batch_size)
            ]
        else:
            embeddings = [
                embedding.tolist() for embedding in embedding_model_inst.query_embed(query=texts)
            ]
        return embeddings

    def _embed_late_interaction_multimodal_text(
        self,
        texts: list[str],
        model_name: str,
        options: Optional[dict[str, Any]],
        batch_size: int,
    ) -> list[list[list[float]]]:
        embedding_model_inst = self.get_or_init_late_interaction_multimodal_model(
            model_name=model_name, **options or {}
        )
        return [
            embedding.tolist()
            for embedding in embedding_model_inst.embed_text(
                documents=texts, batch_size=batch_size
            )
        ]

    def _embed_late_interaction_multimodal_image(
        self,
        images: list[ImageInput],
        model_name: str,
        options: Optional[dict[str, Any]],
        batch_size: int,
    ) -> list[list[list[float]]]:
        embedding_model_inst = self.get_or_init_late_interaction_multimodal_model(
            model_name=model_name, **options or {}
        )
        return [
            embedding.tolist()
            for embedding in embedding_model_inst.embed_image(images=images, batch_size=batch_size)
        ]

    def _embed_dense_image(
        self,
        images: list[ImageInput],
        model_name: str,
        options: Optional[dict[str, Any]],
        batch_size: int,
    ) -> list[list[float]]:
        embedding_model_inst = self.get_or_init_image_model(model_name=model_name, **options or {})
        embeddings = [
            embedding.tolist()
            for embedding in embedding_model_inst.embed(images=images, batch_size=batch_size)
        ]
        return embeddings
