import uuid
from itertools import tee
from typing import Any, Iterable, Optional, Sequence, Union, get_args
from copy import deepcopy

import numpy as np
from pydantic import BaseModel

from qdrant_client import grpc
from qdrant_client.common.client_warnings import show_warning
from qdrant_client.client_base import QdrantBase
from qdrant_client.embed.model_embedder import ModelEmbedder
from qdrant_client.http import models
from qdrant_client.conversions import common_types as types
from qdrant_client.conversions.conversion import GrpcToRest
from qdrant_client.embed.common import INFERENCE_OBJECT_TYPES
from qdrant_client.embed.schema_parser import ModelSchemaParser
from qdrant_client.hybrid.fusion import reciprocal_rank_fusion
from qdrant_client.fastembed_common import (
    QueryResponse,
    TextEmbedding,
    LateInteractionTextEmbedding,
    ImageEmbedding,
    SparseTextEmbedding,
    SUPPORTED_EMBEDDING_MODELS,
    SUPPORTED_SPARSE_EMBEDDING_MODELS,
    IDF_EMBEDDING_MODELS,
    _LATE_INTERACTION_EMBEDDING_MODELS,
    _IMAGE_EMBEDDING_MODELS,
    OnnxProvider,
)


class QdrantFastembedMixin(QdrantBase):
    DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-en"

    _FASTEMBED_INSTALLED: bool

    def __init__(self, parser: ModelSchemaParser, **kwargs: Any):
        self._embedding_model_name: Optional[str] = None
        self._sparse_embedding_model_name: Optional[str] = None
        self._model_embedder = ModelEmbedder(parser=parser)

        try:
            from fastembed import SparseTextEmbedding, TextEmbedding

            assert len(SparseTextEmbedding.list_supported_models()) > 0
            assert len(TextEmbedding.list_supported_models()) > 0

            self.__class__._FASTEMBED_INSTALLED = True
        except ImportError:
            self.__class__._FASTEMBED_INSTALLED = False

        super().__init__(**kwargs)

    @property
    def embedding_model_name(self) -> str:
        if self._embedding_model_name is None:
            self._embedding_model_name = self.DEFAULT_EMBEDDING_MODEL
        return self._embedding_model_name

    @property
    def sparse_embedding_model_name(self) -> Optional[str]:
        return self._sparse_embedding_model_name

    def set_model(
        self,
        embedding_model_name: str,
        max_length: Optional[int] = None,
        cache_dir: Optional[str] = None,
        threads: Optional[int] = None,
        providers: Optional[Sequence["OnnxProvider"]] = None,
        cuda: bool = False,
        device_ids: Optional[list[int]] = None,
        lazy_load: bool = False,
        **kwargs: Any,
    ) -> None:
        """
        Set embedding model to use for encoding documents and queries.

        Args:
            embedding_model_name: One of the supported embedding models. See `SUPPORTED_EMBEDDING_MODELS` for details.
            max_length (int, optional): Deprecated. Defaults to None.
            cache_dir (str, optional): The path to the cache directory.
                Can be set using the `FASTEMBED_CACHE_PATH` env variable.
                Defaults to `fastembed_cache` in the system's temp directory.
            threads (int, optional): The number of threads single onnxruntime session can use. Defaults to None.
            providers: The list of onnx providers (with or without options) to use. Defaults to None.
                Example configuration:
                https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#configuration-options
            cuda (bool, optional): Whether to use cuda for inference. Mutually exclusive with `providers`
                Defaults to False.
            device_ids (Optional[list[int]], optional): The list of device ids to use for data parallel processing in
                workers. Should be used with `cuda=True`, mutually exclusive with `providers`. Defaults to None.
            lazy_load (bool, optional): Whether to load the model during class initialization or on demand.
                Should be set to True when using multiple-gpu and parallel encoding. Defaults to False.
        Raises:
            ValueError: If embedding model is not supported.
            ImportError: If fastembed is not installed.

        Returns:
            None
        """

        if max_length is not None:
            show_warning(
                message="max_length parameter is deprecated and will be removed in the future. "
                "It's not used by fastembed models.",
                category=DeprecationWarning,
                stacklevel=3,
            )

        self._get_or_init_model(
            model_name=embedding_model_name,
            cache_dir=cache_dir,
            threads=threads,
            providers=providers,
            cuda=cuda,
            device_ids=device_ids,
            lazy_load=lazy_load,
            **kwargs,
        )
        self._embedding_model_name = embedding_model_name

    def set_sparse_model(
        self,
        embedding_model_name: Optional[str],
        cache_dir: Optional[str] = None,
        threads: Optional[int] = None,
        providers: Optional[Sequence["OnnxProvider"]] = None,
        cuda: bool = False,
        device_ids: Optional[list[int]] = None,
        lazy_load: bool = False,
        **kwargs: Any,
    ) -> None:
        """
        Set sparse embedding model to use for hybrid search over documents in combination with dense embeddings.

        Args:
            embedding_model_name: One of the supported sparse embedding models. See `SUPPORTED_SPARSE_EMBEDDING_MODELS` for details.
                        If None, sparse embeddings will not be used.
            cache_dir (str, optional): The path to the cache directory.
                                       Can be set using the `FASTEMBED_CACHE_PATH` env variable.
                                       Defaults to `fastembed_cache` in the system's temp directory.
            threads (int, optional): The number of threads single onnxruntime session can use. Defaults to None.
            providers: The list of onnx providers (with or without options) to use. Defaults to None.
                Example configuration:
                https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#configuration-options
            cuda (bool, optional): Whether to use cuda for inference. Mutually exclusive with `providers`
                Defaults to False.
            device_ids (Optional[list[int]], optional): The list of device ids to use for data parallel processing in
                workers. Should be used with `cuda=True`, mutually exclusive with `providers`. Defaults to None.
            lazy_load (bool, optional): Whether to load the model during class initialization or on demand.
                Should be set to True when using multiple-gpu and parallel encoding. Defaults to False.
        Raises:
            ValueError: If embedding model is not supported.
            ImportError: If fastembed is not installed.

        Returns:
            None
        """
        if embedding_model_name is not None:
            self._get_or_init_sparse_model(
                model_name=embedding_model_name,
                cache_dir=cache_dir,
                threads=threads,
                providers=providers,
                cuda=cuda,
                device_ids=device_ids,
                lazy_load=lazy_load,
                **kwargs,
            )
        self._sparse_embedding_model_name = embedding_model_name

    @classmethod
    def _import_fastembed(cls) -> None:
        if cls._FASTEMBED_INSTALLED:
            return

        # If it's not, ask the user to install it
        raise ImportError(
            "fastembed is not installed."
            " Please install it to enable fast vector indexing with `pip install fastembed`."
        )

    @classmethod
    def _get_model_params(cls, model_name: str) -> tuple[int, models.Distance]:
        cls._import_fastembed()

        if model_name not in SUPPORTED_EMBEDDING_MODELS:
            raise ValueError(
                f"Unsupported embedding model: {model_name}. Supported models: {SUPPORTED_EMBEDDING_MODELS}"
            )

        return SUPPORTED_EMBEDDING_MODELS[model_name]

    def _get_or_init_model(
        self,
        model_name: str,
        cache_dir: Optional[str] = None,
        threads: Optional[int] = None,
        providers: Optional[Sequence["OnnxProvider"]] = None,
        **kwargs: Any,
    ) -> "TextEmbedding":
        if model_name in self._model_embedder.embedder.embedding_models:
            return self._model_embedder.embedder.embedding_models[model_name]

        self._import_fastembed()

        if model_name not in SUPPORTED_EMBEDDING_MODELS:
            raise ValueError(
                f"Unsupported embedding model: {model_name}. Supported models: {SUPPORTED_EMBEDDING_MODELS}"
            )

        self._model_embedder.embedder.get_or_init_model(
            model_name=model_name,
            cache_dir=cache_dir,
            threads=threads,
            providers=providers,
            **kwargs,
        )

        return self._model_embedder.embedder.embedding_models[model_name]

    def _get_or_init_sparse_model(
        self,
        model_name: str,
        cache_dir: Optional[str] = None,
        threads: Optional[int] = None,
        providers: Optional[Sequence["OnnxProvider"]] = None,
        **kwargs: Any,
    ) -> "SparseTextEmbedding":
        if model_name in self._model_embedder.embedder.sparse_embedding_models:
            return self._model_embedder.embedder.sparse_embedding_models[model_name]

        self._import_fastembed()

        if model_name not in SUPPORTED_SPARSE_EMBEDDING_MODELS:
            raise ValueError(
                f"Unsupported embedding model: {model_name}. Supported models: {SUPPORTED_SPARSE_EMBEDDING_MODELS}"
            )

        return self._model_embedder.embedder.get_or_init_sparse_model(
            model_name=model_name,
            cache_dir=cache_dir,
            threads=threads,
            providers=providers,
            **kwargs,
        )

    def _get_or_init_late_interaction_model(
        self,
        model_name: str,
        cache_dir: Optional[str] = None,
        threads: Optional[int] = None,
        providers: Optional[Sequence["OnnxProvider"]] = None,
        **kwargs: Any,
    ) -> "LateInteractionTextEmbedding":
        if model_name in self._model_embedder.embedder.late_interaction_embedding_models:
            return self._model_embedder.embedder.late_interaction_embedding_models[model_name]

        self._import_fastembed()

        if model_name not in _LATE_INTERACTION_EMBEDDING_MODELS:
            raise ValueError(
                f"Unsupported embedding model: {model_name}. Supported models: {_LATE_INTERACTION_EMBEDDING_MODELS}"
            )

        return self._model_embedder.embedder.get_or_init_late_interaction_model(
            model_name=model_name,
            cache_dir=cache_dir,
            threads=threads,
            providers=providers,
            **kwargs,
        )

    def _get_or_init_image_model(
        self,
        model_name: str,
        cache_dir: Optional[str] = None,
        threads: Optional[int] = None,
        providers: Optional[Sequence["OnnxProvider"]] = None,
        **kwargs: Any,
    ) -> "ImageEmbedding":
        if model_name in self._model_embedder.embedder.image_embedding_models:
            return self._model_embedder.embedder.image_embedding_models[model_name]

        self._import_fastembed()

        if model_name not in _IMAGE_EMBEDDING_MODELS:
            raise ValueError(
                f"Unsupported embedding model: {model_name}. Supported models: {_IMAGE_EMBEDDING_MODELS}"
            )

        return self._model_embedder.embedder.get_or_init_image_model(
            model_name=model_name,
            cache_dir=cache_dir,
            threads=threads,
            providers=providers,
            **kwargs,
        )

    def _embed_documents(
        self,
        documents: Iterable[str],
        embedding_model_name: str = DEFAULT_EMBEDDING_MODEL,
        batch_size: int = 32,
        embed_type: str = "default",
        parallel: Optional[int] = None,
    ) -> Iterable[tuple[str, list[float]]]:
        embedding_model = self._get_or_init_model(model_name=embedding_model_name)
        documents_a, documents_b = tee(documents, 2)
        if embed_type == "passage":
            vectors_iter = embedding_model.passage_embed(
                documents_a, batch_size=batch_size, parallel=parallel
            )
        elif embed_type == "query":
            vectors_iter = (
                list(embedding_model.query_embed(query=query))[0] for query in documents_a
            )
        elif embed_type == "default":
            vectors_iter = embedding_model.embed(
                documents_a, batch_size=batch_size, parallel=parallel
            )
        else:
            raise ValueError(f"Unknown embed type: {embed_type}")

        for vector, doc in zip(vectors_iter, documents_b):
            yield doc, vector.tolist()

    def _sparse_embed_documents(
        self,
        documents: Iterable[str],
        embedding_model_name: str = DEFAULT_EMBEDDING_MODEL,
        batch_size: int = 32,
        parallel: Optional[int] = None,
    ) -> Iterable[types.SparseVector]:
        sparse_embedding_model = self._get_or_init_sparse_model(model_name=embedding_model_name)

        vectors_iter = sparse_embedding_model.embed(
            documents, batch_size=batch_size, parallel=parallel
        )

        for sparse_vector in vectors_iter:
            yield types.SparseVector(
                indices=sparse_vector.indices.tolist(),
                values=sparse_vector.values.tolist(),
            )

    def get_vector_field_name(self) -> str:
        """
        Returns name of the vector field in qdrant collection, used by current fastembed model.
        Returns:
            Name of the vector field.
        """
        model_name = self.embedding_model_name.split("/")[-1].lower()
        return f"fast-{model_name}"

    def get_sparse_vector_field_name(self) -> Optional[str]:
        """
        Returns name of the vector field in qdrant collection, used by current fastembed model.
        Returns:
            Name of the vector field.
        """
        if self.sparse_embedding_model_name is not None:
            model_name = self.sparse_embedding_model_name.split("/")[-1].lower()
            return f"fast-sparse-{model_name}"
        return None

    def _scored_points_to_query_responses(
        self,
        scored_points: list[types.ScoredPoint],
    ) -> list[QueryResponse]:
        response = []
        vector_field_name = self.get_vector_field_name()
        sparse_vector_field_name = self.get_sparse_vector_field_name()

        for scored_point in scored_points:
            embedding = (
                scored_point.vector.get(vector_field_name, None)
                if isinstance(scored_point.vector, dict)
                else None
            )
            sparse_embedding = None
            if sparse_vector_field_name is not None:
                sparse_embedding = (
                    scored_point.vector.get(sparse_vector_field_name, None)
                    if isinstance(scored_point.vector, dict)
                    else None
                )

            response.append(
                QueryResponse(
                    id=scored_point.id,
                    embedding=embedding,
                    sparse_embedding=sparse_embedding,
                    metadata=scored_point.payload,
                    document=scored_point.payload.get("document", ""),
                    score=scored_point.score,
                )
            )
        return response

    def _points_iterator(
        self,
        ids: Optional[Iterable[models.ExtendedPointId]],
        metadata: Optional[Iterable[dict[str, Any]]],
        encoded_docs: Iterable[tuple[str, list[float]]],
        ids_accumulator: list,
        sparse_vectors: Optional[Iterable[types.SparseVector]] = None,
    ) -> Iterable[models.PointStruct]:
        if ids is None:
            ids = iter(lambda: uuid.uuid4().hex, None)

        if metadata is None:
            metadata = iter(lambda: {}, None)

        if sparse_vectors is None:
            sparse_vectors = iter(lambda: None, True)

        vector_name = self.get_vector_field_name()
        sparse_vector_name = self.get_sparse_vector_field_name()

        for idx, meta, (doc, vector), sparse_vector in zip(
            ids, metadata, encoded_docs, sparse_vectors
        ):
            ids_accumulator.append(idx)
            payload = {"document": doc, **meta}
            point_vector: dict[str, models.Vector] = {vector_name: vector}
            if sparse_vector_name is not None and sparse_vector is not None:
                point_vector[sparse_vector_name] = sparse_vector
            yield models.PointStruct(id=idx, payload=payload, vector=point_vector)

    def _validate_collection_info(self, collection_info: models.CollectionInfo) -> None:
        embeddings_size, distance = self._get_model_params(model_name=self.embedding_model_name)
        vector_field_name = self.get_vector_field_name()

        # Check if collection has compatible vector params
        assert isinstance(
            collection_info.config.params.vectors, dict
        ), f"Collection have incompatible vector params: {collection_info.config.params.vectors}"

        assert (
            vector_field_name in collection_info.config.params.vectors
        ), f"Collection have incompatible vector params: {collection_info.config.params.vectors}, expected {vector_field_name}"

        vector_params = collection_info.config.params.vectors[vector_field_name]

        assert (
            embeddings_size == vector_params.size
        ), f"Embedding size mismatch: {embeddings_size} != {vector_params.size}"

        assert (
            distance == vector_params.distance
        ), f"Distance mismatch: {distance} != {vector_params.distance}"

        sparse_vector_field_name = self.get_sparse_vector_field_name()
        if sparse_vector_field_name is not None:
            assert (
                sparse_vector_field_name in collection_info.config.params.sparse_vectors
            ), f"Collection have incompatible vector params: {collection_info.config.params.vectors}"
            if self.sparse_embedding_model_name in IDF_EMBEDDING_MODELS:
                modifier = collection_info.config.params.sparse_vectors[
                    sparse_vector_field_name
                ].modifier
                assert (
                    modifier == models.Modifier.IDF
                ), f"{self.sparse_embedding_model_name} requires modifier IDF, current modifier is {modifier}"

    def get_fastembed_vector_params(
        self,
        on_disk: Optional[bool] = None,
        quantization_config: Optional[models.QuantizationConfig] = None,
        hnsw_config: Optional[models.HnswConfigDiff] = None,
    ) -> dict[str, models.VectorParams]:
        """
        Generates vector configuration, compatible with fastembed models.

        Args:
            on_disk: if True, vectors will be stored on disk. If None, default value will be used.
            quantization_config: Quantization configuration. If None, quantization will be disabled.
            hnsw_config: HNSW configuration. If None, default configuration will be used.

        Returns:
            Configuration for `vectors_config` argument in `create_collection` method.
        """
        vector_field_name = self.get_vector_field_name()
        embeddings_size, distance = self._get_model_params(model_name=self.embedding_model_name)
        return {
            vector_field_name: models.VectorParams(
                size=embeddings_size,
                distance=distance,
                on_disk=on_disk,
                quantization_config=quantization_config,
                hnsw_config=hnsw_config,
            )
        }

    def get_fastembed_sparse_vector_params(
        self,
        on_disk: Optional[bool] = None,
        modifier: Optional[models.Modifier] = None,
    ) -> Optional[dict[str, models.SparseVectorParams]]:
        """
        Generates vector configuration, compatible with fastembed sparse models.

        Args:
            on_disk: if True, vectors will be stored on disk. If None, default value will be used.
            modifier: Sparse vector queries modifier. E.g. Modifier.IDF for idf-based rescoring. Default: None.
        Returns:
            Configuration for `vectors_config` argument in `create_collection` method.
        """
        vector_field_name = self.get_sparse_vector_field_name()
        if self.sparse_embedding_model_name in IDF_EMBEDDING_MODELS:
            modifier = models.Modifier.IDF if modifier is None else modifier

        if vector_field_name is None:
            return None

        return {
            vector_field_name: models.SparseVectorParams(
                index=models.SparseIndexParams(
                    on_disk=on_disk,
                ),
                modifier=modifier,
            )
        }

    def add(
        self,
        collection_name: str,
        documents: Iterable[str],
        metadata: Optional[Iterable[dict[str, Any]]] = None,
        ids: Optional[Iterable[models.ExtendedPointId]] = None,
        batch_size: int = 32,
        parallel: Optional[int] = None,
        **kwargs: Any,
    ) -> list[Union[str, int]]:
        """
        Adds text documents into qdrant collection.
        If collection does not exist, it will be created with default parameters.
        Metadata in combination with documents will be added as payload.
        Documents will be embedded using the specified embedding model.

        If you want to use your own vectors, use `upsert` method instead.

        Args:
            collection_name (str):
                Name of the collection to add documents to.
            documents (Iterable[str]):
                List of documents to embed and add to the collection.
            metadata (Iterable[dict[str, Any]], optional):
                List of metadata dicts. Defaults to None.
            ids (Iterable[models.ExtendedPointId], optional):
                List of ids to assign to documents.
                If not specified, UUIDs will be generated. Defaults to None.
            batch_size (int, optional):
                How many documents to embed and upload in single request. Defaults to 32.
            parallel (Optional[int], optional):
                How many parallel workers to use for embedding. Defaults to None.
                If number is specified, data-parallel process will be used.

        Raises:
            ImportError: If fastembed is not installed.

        Returns:
            List of IDs of added documents. If no ids provided, UUIDs will be randomly generated on client side.

        """

        # check if we have fastembed installed
        encoded_docs = self._embed_documents(
            documents=documents,
            embedding_model_name=self.embedding_model_name,
            batch_size=batch_size,
            embed_type="passage",
            parallel=parallel,
        )

        encoded_sparse_docs = None
        if self.sparse_embedding_model_name is not None:
            encoded_sparse_docs = self._sparse_embed_documents(
                documents=documents,
                embedding_model_name=self.sparse_embedding_model_name,
                batch_size=batch_size,
                parallel=parallel,
            )

        # Check if collection by same name exists, if not, create it
        try:
            collection_info = self.get_collection(collection_name=collection_name)
        except Exception:
            self.create_collection(
                collection_name=collection_name,
                vectors_config=self.get_fastembed_vector_params(),
                sparse_vectors_config=self.get_fastembed_sparse_vector_params(),
            )
            collection_info = self.get_collection(collection_name=collection_name)

        self._validate_collection_info(collection_info)

        inserted_ids: list = []

        points = self._points_iterator(
            ids=ids,
            metadata=metadata,
            encoded_docs=encoded_docs,
            ids_accumulator=inserted_ids,
            sparse_vectors=encoded_sparse_docs,
        )

        self.upload_points(
            collection_name=collection_name,
            points=points,
            wait=True,
            parallel=parallel or 1,
            batch_size=batch_size,
            **kwargs,
        )

        return inserted_ids

    def query(
        self,
        collection_name: str,
        query_text: str,
        query_filter: Optional[models.Filter] = None,
        limit: int = 10,
        **kwargs: Any,
    ) -> list[QueryResponse]:
        """
        Search for documents in a collection.
        This method automatically embeds the query text using the specified embedding model.
        If you want to use your own query vector, use `search` method instead.

        Args:
            collection_name: Collection to search in
            query_text:
                Text to search for. This text will be embedded using the specified embedding model.
                And then used as a query vector.
            query_filter:
                - Exclude vectors which doesn't fit given conditions.
                - If `None` - search among all vectors
            limit: How many results return
            **kwargs: Additional search parameters. See `qdrant_client.models.SearchRequest` for details.

        Returns:
            list[types.ScoredPoint]: List of scored points.

        """

        embedding_model_inst = self._get_or_init_model(model_name=self.embedding_model_name)
        embeddings = list(embedding_model_inst.query_embed(query=query_text))
        query_vector = embeddings[0].tolist()

        if self.sparse_embedding_model_name is None:
            return self._scored_points_to_query_responses(
                self.search(
                    collection_name=collection_name,
                    query_vector=models.NamedVector(
                        name=self.get_vector_field_name(), vector=query_vector
                    ),
                    query_filter=query_filter,
                    limit=limit,
                    with_payload=True,
                    **kwargs,
                )
            )

        sparse_embedding_model_inst = self._get_or_init_sparse_model(
            model_name=self.sparse_embedding_model_name
        )
        sparse_vector = list(sparse_embedding_model_inst.query_embed(query=query_text))[0]
        sparse_query_vector = models.SparseVector(
            indices=sparse_vector.indices.tolist(),
            values=sparse_vector.values.tolist(),
        )

        dense_request = models.SearchRequest(
            vector=models.NamedVector(
                name=self.get_vector_field_name(),
                vector=query_vector,
            ),
            filter=query_filter,
            limit=limit,
            with_payload=True,
            **kwargs,
        )
        sparse_request = models.SearchRequest(
            vector=models.NamedSparseVector(
                name=self.get_sparse_vector_field_name(),
                vector=sparse_query_vector,
            ),
            filter=query_filter,
            limit=limit,
            with_payload=True,
            **kwargs,
        )

        dense_request_response, sparse_request_response = self.search_batch(
            collection_name=collection_name, requests=[dense_request, sparse_request]
        )
        return self._scored_points_to_query_responses(
            reciprocal_rank_fusion([dense_request_response, sparse_request_response], limit=limit)
        )

    def query_batch(
        self,
        collection_name: str,
        query_texts: list[str],
        query_filter: Optional[models.Filter] = None,
        limit: int = 10,
        **kwargs: Any,
    ) -> list[list[QueryResponse]]:
        """
        Search for documents in a collection with batched query.
        This method automatically embeds the query text using the specified embedding model.

        Args:
            collection_name: Collection to search in
            query_texts:
                A list of texts to search for. Each text will be embedded using the specified embedding model.
                And then used as a query vector for a separate search requests.
            query_filter:
                - Exclude vectors which doesn't fit given conditions.
                - If `None` - search among all vectors
                This filter will be applied to all search requests.
            limit: How many results return
            **kwargs: Additional search parameters. See `qdrant_client.models.SearchRequest` for details.

        Returns:
            list[list[QueryResponse]]: List of lists of responses for each query text.

        """
        embedding_model_inst = self._get_or_init_model(model_name=self.embedding_model_name)
        query_vectors = list(embedding_model_inst.query_embed(query=query_texts))
        requests = []
        for vector in query_vectors:
            request = models.SearchRequest(
                vector=models.NamedVector(
                    name=self.get_vector_field_name(), vector=vector.tolist()
                ),
                filter=query_filter,
                limit=limit,
                with_payload=True,
                **kwargs,
            )

            requests.append(request)

        if self.sparse_embedding_model_name is None:
            responses = self.search_batch(
                collection_name=collection_name,
                requests=requests,
            )
            return [self._scored_points_to_query_responses(response) for response in responses]

        sparse_embedding_model_inst = self._get_or_init_sparse_model(
            model_name=self.sparse_embedding_model_name
        )
        sparse_query_vectors = [
            models.SparseVector(
                indices=sparse_vector.indices.tolist(),
                values=sparse_vector.values.tolist(),
            )
            for sparse_vector in sparse_embedding_model_inst.embed(documents=query_texts)
        ]
        for sparse_vector in sparse_query_vectors:
            request = models.SearchRequest(
                vector=models.NamedSparseVector(
                    name=self.get_sparse_vector_field_name(),
                    vector=sparse_vector,
                ),
                filter=query_filter,
                limit=limit,
                with_payload=True,
                **kwargs,
            )

            requests.append(request)

        responses = self.search_batch(
            collection_name=collection_name,
            requests=requests,
        )

        dense_responses = responses[: len(query_texts)]
        sparse_responses = responses[len(query_texts) :]
        responses = [
            reciprocal_rank_fusion([dense_response, sparse_response], limit=limit)
            for dense_response, sparse_response in zip(dense_responses, sparse_responses)
        ]

        return [self._scored_points_to_query_responses(response) for response in responses]

    @classmethod
    def _resolve_query(
        cls,
        query: Union[
            types.PointId,
            list[float],
            list[list[float]],
            types.SparseVector,
            types.Query,
            types.NumpyArray,
            models.Document,
            models.Image,
            models.InferenceObject,
            None,
        ],
    ) -> Optional[models.Query]:
        """Resolves query interface into a models.Query object

        Args:
            query: models.QueryInterface - query as a model or a plain structure like list[float]

        Returns:
            Optional[models.Query]: query as it was, models.Query(nearest=query) or None

        Raises:
            ValueError: if query is not of supported type
        """
        if isinstance(query, get_args(types.Query)) or isinstance(query, grpc.Query):
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
        raw_models: Union[BaseModel, Iterable[BaseModel]],
        is_query: bool = False,
        batch_size: int = 32,
    ) -> Iterable[BaseModel]:
        yield from self._model_embedder.embed_models(
            raw_models=raw_models, is_query=is_query, batch_size=batch_size
        )

    def _embed_models_strict(
        self,
        raw_models: Iterable[Union[dict[str, BaseModel], BaseModel]],
        batch_size: int = 32,
        parallel: Optional[int] = None,
    ) -> Iterable[BaseModel]:
        yield from self._model_embedder.embed_models_strict(
            raw_models=raw_models, batch_size=batch_size, parallel=parallel
        )
