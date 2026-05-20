from typing import Any, Iterable, Sequence, get_args
from copy import deepcopy

import numpy as np
from pydantic import BaseModel

from qdrant_client import grpc
from qdrant_client.client_base import QdrantBase
from qdrant_client.embed.model_embedder import ModelEmbedder
from qdrant_client.http import models
from qdrant_client.conversions import common_types as types
from qdrant_client.conversions.conversion import GrpcToRest
from qdrant_client.embed.common import INFERENCE_OBJECT_TYPES
from qdrant_client.embed.schema_parser import ModelSchemaParser
from qdrant_client.fastembed_common import FastEmbedMisc


class QdrantFastembedMixin(QdrantBase):
    DEFAULT_BATCH_SIZE = 8
    _FASTEMBED_INSTALLED: bool

    def __init__(self, parser: ModelSchemaParser, is_local_mode: bool):
        self.__class__._FASTEMBED_INSTALLED = FastEmbedMisc.is_installed()
        self._model_embedder = ModelEmbedder(parser=parser, is_local_mode=is_local_mode)
        super().__init__()

    @classmethod
    def list_text_models(cls) -> dict[str, tuple[int, models.Distance]]:
        """Lists the supported dense text models.

        Returns:
            dict[str, tuple[int, models.Distance]]: A dict of model names, their dimensions and distance metrics.
        """
        return FastEmbedMisc.list_text_models()

    @classmethod
    def list_image_models(cls) -> dict[str, tuple[int, models.Distance]]:
        """Lists the supported image dense models.

        Returns:
            dict[str, tuple[int, models.Distance]]: A dict of model names, their dimensions and distance metrics.
        """
        return FastEmbedMisc.list_image_models()

    @classmethod
    def list_late_interaction_text_models(cls) -> dict[str, tuple[int, models.Distance]]:
        """Lists the supported late interaction text models.

        Returns:
            dict[str, tuple[int, models.Distance]]: A dict of model names, their dimensions and distance metrics.
        """
        return FastEmbedMisc.list_late_interaction_text_models()

    @classmethod
    def list_late_interaction_multimodal_models(cls) -> dict[str, tuple[int, models.Distance]]:
        """Lists the supported late interaction multimodal models.

        Returns:
            dict[str, tuple[int, models.Distance]]: A dict of model names, their dimensions and distance metrics.
        """
        return FastEmbedMisc.list_late_interaction_multimodal_models()

    @classmethod
    def list_sparse_models(cls) -> dict[str, dict[str, Any]]:
        """Lists the supported sparse text models.

        Returns:
            dict[str, dict[str, Any]]: A dict of model names and their descriptions.
        """
        return FastEmbedMisc.list_sparse_models()

    @classmethod
    def _get_model_params(cls, model_name: str) -> tuple[int, models.Distance]:
        FastEmbedMisc.import_fastembed()

        for descriptions in (
            FastEmbedMisc.list_text_models(),
            FastEmbedMisc.list_image_models(),
            FastEmbedMisc.list_late_interaction_text_models(),
            FastEmbedMisc.list_late_interaction_multimodal_models(),
        ):
            if params := descriptions.get(model_name):
                return params

        if model_name in FastEmbedMisc.list_sparse_models():
            raise ValueError(
                "Sparse embeddings do not return fixed embedding size and distance type"
            )

        raise ValueError(f"Unsupported embedding model: {model_name}")

    def get_embedding_size(
        self,
        model_name: str,
    ) -> int:
        """Get the size of the embeddings produced by the specified model.

        Args:
            model_name: the name of the model to get the embedding size for.
        Returns:
            int: the size of the embeddings produced by the model.

        Raises:
            ValueError: If sparse model name is passed or model is not found in the supported models.
        """
        embeddings_size, _ = self._get_model_params(model_name=model_name)
        return embeddings_size

    @classmethod
    def _resolve_query(
        cls,
        query: types.PointId
        | list[float]
        | list[list[float]]
        | types.SparseVector
        | types.Query
        | types.NumpyArray
        | models.Document
        | models.Image
        | models.InferenceObject
        | None,
    ) -> models.Query | None:
        """Resolves query interface into a models.Query object

        Args:
            query: models.QueryInterface - query as a model or a plain structure like list[float]

        Returns:
            Optional[models.Query]: query as it was, models.Query(nearest=query) or None

        Raises:
            ValueError: if query is not of supported type
        """
        if isinstance(query, get_args(types.Query)):
            return query

        if isinstance(query, types.SparseVector):
            return models.NearestQuery(nearest=query)

        if isinstance(query, np.ndarray):
            return models.NearestQuery(nearest=query.tolist())
        if isinstance(query, list):
            return models.NearestQuery(nearest=query)

        if isinstance(query, get_args(types.PointId)):
            query = (
                GrpcToRest.convert_point_id(query) if isinstance(query, grpc.PointId) else query
            )
            return models.NearestQuery(nearest=query)

        if isinstance(query, get_args(INFERENCE_OBJECT_TYPES)):
            return models.NearestQuery(nearest=query)

        if query is None:
            return None

        raise ValueError(f"Unsupported query type: {type(query)}")

    def _resolve_query_request(self, query: models.QueryRequest) -> models.QueryRequest:
        """Resolve QueryRequest query field

        Args:
            query: models.QueryRequest - query request to resolve

        Returns:
            models.QueryRequest: A deepcopy of the query request with resolved query field
        """
        query = deepcopy(query)
        query.query = self._resolve_query(query.query)
        return query

    def _resolve_query_batch_request(
        self, requests: Sequence[models.QueryRequest]
    ) -> Sequence[models.QueryRequest]:
        """Resolve query field for each query request in a batch

        Args:
            requests: Sequence[models.QueryRequest] - query requests to resolve

        Returns:
            Sequence[models.QueryRequest]: A list of deep copied query requests with resolved query fields
        """
        return [self._resolve_query_request(query) for query in requests]

    def _embed_models(
        self,
        raw_models: BaseModel | Iterable[BaseModel],
        is_query: bool = False,
        batch_size: int | None = None,
    ) -> Iterable[BaseModel]:
        yield from self._model_embedder.embed_models(
            raw_models=raw_models,
            is_query=is_query,
            batch_size=batch_size or self.DEFAULT_BATCH_SIZE,
        )

    def _embed_models_strict(
        self,
        raw_models: Iterable[dict[str, BaseModel] | BaseModel],
        batch_size: int | None = None,
        parallel: int | None = None,
    ) -> Iterable[BaseModel]:
        yield from self._model_embedder.embed_models_strict(
            raw_models=raw_models,
            batch_size=batch_size or self.DEFAULT_BATCH_SIZE,
            parallel=parallel,
        )
