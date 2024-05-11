import uuid
import warnings
from itertools import tee
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

from qdrant_client.client_base import QdrantBase
from qdrant_client.conversions import common_types as types
from qdrant_client.fastembed_common import QueryResponse
from qdrant_client.http import models
from qdrant_client.hybrid.fusion import reciprocal_rank_fusion

try:
    from fastembed import ImageEmbedding, SparseTextEmbedding, TextEmbedding
    from fastembed.common import OnnxProvider, PathInput

except ImportError:
    TextEmbedding, ImageEmbedding, SparseTextEmbedding = None, None, None
    OnnxProvider, PathInput = None, None


SUPPORTED_EMBEDDING_MODELS: Dict[str, Tuple[int, models.Distance]] = (
    {
        model["model"]: (model["dim"], models.Distance.COSINE)
        for model in TextEmbedding.list_supported_models()
    }
    if TextEmbedding
    else {}
)

SUPPORTED_IMAGE_EMBEDDING_MODELS: Dict[str, Tuple[int, models.Distance]] = (
    {
        model["model"]: (model["dim"], models.Distance.COSINE)
        for model in ImageEmbedding.list_supported_models()
    }
    if ImageEmbedding
    else {}
)

SUPPORTED_SPARSE_EMBEDDING_MODELS: Dict[str, Tuple[int, models.Distance]] = (
    {model["model"]: model for model in SparseTextEmbedding.list_supported_models()}
    if SparseTextEmbedding
    else {}
)


class QdrantFastembedMixin(QdrantBase):
    DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-en"

    embedding_models: Dict[str, "TextEmbedding"] = {}
    image_embedding_models: Dict[str, "ImageEmbedding"] = {}
    sparse_embedding_models: Dict[str, "SparseTextEmbedding"] = {}

    _FASTEMBED_INSTALLED: bool

    def __init__(self, **kwargs: Any):
        self._embedding_model_name: Optional[str] = None
        self._image_embedding_model_name: Optional[str] = None
        self._sparse_embedding_model_name: Optional[str] = None
        try:
            from fastembed import SparseTextEmbedding, TextEmbedding, ImageEmbedding

            assert len(SparseTextEmbedding.list_supported_models()) > 0
            assert len(TextEmbedding.list_supported_models()) > 0
            assert len(ImageEmbedding.list_supported_models()) > 0

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
    def image_embedding_model_name(self) -> Optional[str]:
        return self._image_embedding_model_name

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
        **kwargs: Any,
    ) -> None:
        """
        Set text embedding model to use for encoding documents and queries.
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
        Raises:
            ValueError: If embedding model is not supported.
            ImportError: If fastembed is not installed.

        Returns:
            None
        """
        if max_length is not None:
            warnings.warn(
                "max_length parameter is deprecated and will be removed in the future. "
                "It's not used by fastembed models.",
                DeprecationWarning,
                stacklevel=2,
            )

        self._get_or_init_text_model(
            model_name=embedding_model_name,
            cache_dir=cache_dir,
            threads=threads,
            providers=providers,
            **kwargs,
        )

        self._embedding_model_name = embedding_model_name

    def set_image_model(
        self,
        embedding_model_name: str,
        cache_dir: Optional[str] = None,
        threads: Optional[int] = None,
        providers: Optional[Sequence[OnnxProvider]] = None,
        **kwargs: Any,
    ) -> None:
        """
        Set image embedding model to use for encoding documents and queries.
        Args:
            embedding_model_name: One of the supported embedding models. See `SUPPORTED_IMAGE_EMBEDDING_MODELS` for details.
            cache_dir (str, optional): The path to the cache directory.
                                       Can be set using the `FASTEMBED_CACHE_PATH` env variable.
                                       Defaults to `fastembed_cache` in the system's temp directory.
            threads (int, optional): The number of threads single onnxruntime session can use. Defaults to None.
            providers: The list of onnx providers (with or without options) to use. Defaults to None.
                Example configuration:
                https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#configuration-options
        Raises:
            ValueError: If embedding model is not supported.
            ImportError: If fastembed is not installed.

        Returns:
            None
        """
        self._get_or_init_image_model(
            model_name=embedding_model_name,
            cache_dir=cache_dir,
            threads=threads,
            providers=providers,
            **kwargs,
        )

        self._image_embedding_model_name = embedding_model_name

    def set_sparse_model(
        self,
        embedding_model_name: Optional[str],
        cache_dir: Optional[str] = None,
        threads: Optional[int] = None,
        providers: Optional[Sequence["OnnxProvider"]] = None,
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
    def _get_text_model_params(cls, model_name: str) -> Tuple[int, models.Distance]:
        cls._import_fastembed()

        if model_name not in SUPPORTED_EMBEDDING_MODELS:
            raise ValueError(
                f"Unsupported embedding model: {model_name}. Supported models: {SUPPORTED_EMBEDDING_MODELS}"
            )

        return SUPPORTED_EMBEDDING_MODELS[model_name]

    @classmethod
    def _get_image_model_params(cls, model_name: str) -> Tuple[int, models.Distance]:
        cls._import_fastembed()

        if model_name not in SUPPORTED_IMAGE_EMBEDDING_MODELS:
            raise ValueError(
                f"Unsupported embedding model: {model_name}. Supported models: {SUPPORTED_IMAGE_EMBEDDING_MODELS}"
            )

        return SUPPORTED_IMAGE_EMBEDDING_MODELS[model_name]

    @classmethod
    def _get_or_init_text_model(
        cls,
        model_name: str,
        cache_dir: Optional[str] = None,
        threads: Optional[int] = None,
        providers: Optional[Sequence["OnnxProvider"]] = None,
        **kwargs: Any,
    ) -> "TextEmbedding":
        if model_name in cls.embedding_models:
            return cls.embedding_models[model_name]

        cls._import_fastembed()

        if model_name not in SUPPORTED_EMBEDDING_MODELS:
            raise ValueError(
                f"Unsupported text embedding model: {model_name}. Supported models: {SUPPORTED_EMBEDDING_MODELS}"
            )

        cls.embedding_models[model_name] = TextEmbedding(
            model_name=model_name,
            cache_dir=cache_dir,
            threads=threads,
            providers=providers,
            **kwargs,
        )
        return cls.embedding_models[model_name]

    @classmethod
    def _get_or_init_image_model(
        cls,
        model_name: str,
        cache_dir: Optional[str] = None,
        threads: Optional[int] = None,
        providers: Optional[Sequence["OnnxProvider"]] = None,
        **kwargs: Any,
    ) -> "ImageEmbedding":
        if model_name in cls.image_embedding_models:
            return cls.image_embedding_models[model_name]

        cls._import_fastembed()

        if model_name not in SUPPORTED_IMAGE_EMBEDDING_MODELS:
            raise ValueError(
                f"Unsupported image embedding model: {model_name}. Supported models: {SUPPORTED_IMAGE_EMBEDDING_MODELS}"
            )

        cls.image_embedding_models[model_name] = ImageEmbedding(
            model_name=model_name,
            cache_dir=cache_dir,
            threads=threads,
            providers=providers,
            **kwargs,
        )
        return cls.image_embedding_models[model_name]

    @classmethod
    def _get_or_init_sparse_model(
        cls,
        model_name: str,
        cache_dir: Optional[str] = None,
        threads: Optional[int] = None,
        providers: Optional[Sequence["OnnxProvider"]] = None,
        **kwargs: Any,
    ) -> "SparseTextEmbedding":
        if model_name in cls.sparse_embedding_models:
            return cls.sparse_embedding_models[model_name]

        cls._import_fastembed()

        if model_name not in SUPPORTED_SPARSE_EMBEDDING_MODELS:
            raise ValueError(
                f"Unsupported sparse embedding model: {model_name}. Supported models: {SUPPORTED_SPARSE_EMBEDDING_MODELS}"
            )

        cls.sparse_embedding_models[model_name] = SparseTextEmbedding(
            model_name=model_name,
            cache_dir=cache_dir,
            threads=threads,
            providers=providers,
            **kwargs,
        )
        return cls.sparse_embedding_models[model_name]

    def _embed_documents(
        self,
        documents: Iterable[str],
        embedding_model_name: str = DEFAULT_EMBEDDING_MODEL,
        batch_size: int = 32,
        embed_type: str = "default",
        parallel: Optional[int] = None,
    ) -> Iterable[List[float]]:
        embedding_model = self._get_or_init_text_model(model_name=embedding_model_name)
        if embed_type == "passage":
            vectors_iter = embedding_model.passage_embed(
                documents, batch_size=batch_size, parallel=parallel
            )
        elif embed_type == "query":
            vectors_iter = (
                list(embedding_model.query_embed(query=query))[0] for query in documents
            )
        elif embed_type == "default":
            vectors_iter = embedding_model.embed(
                documents, batch_size=batch_size, parallel=parallel
            )
        else:
            raise ValueError(f"Unknown embed type: {embed_type}")

        yield from vectors_iter

    def _embed_images(
        self,
        images: Iterable[PathInput],
        embedding_model_name: str,
        batch_size: int = 8,
        parallel: Optional[int] = None,
    ) -> Iterable[List[float]]:
        embedding_model = self._get_or_init_image_model(model_name=embedding_model_name)
        vectors_iter = embedding_model.embed(images, batch_size=batch_size, parallel=parallel)
        yield from vectors_iter

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

    @staticmethod
    def _get_vector_field_name(model_name: str, prefix: str = "fast"):
        return f"{prefix}-{model_name.split('/')[-1].lower()}"

    def get_vector_field_name(self) -> Optional[str]:
        """
        Returns name of the text vector field in qdrant collection, used by current fastembed model.
        Returns:
            Name of the vector field.
        """
        if self._embedding_model_name is not None:
            return self._get_vector_field_name(self._embedding_model_name)
        return None

    def get_image_vector_field_name(self) -> Optional[str]:
        """
        Returns name of the image vector field in qdrant collection, used by current fastembed model.
        Returns:
            Name of the vector field.
        """
        if self.image_embedding_model_name is not None:
            return self._get_vector_field_name(
                self.image_embedding_model_name, prefix="fast-image"
            )
        return None

    def get_sparse_vector_field_name(self) -> Optional[str]:
        """
        Returns name of the vector field in qdrant collection, used by current fastembed model.
        Returns:
            Name of the vector field.
        """
        if self.sparse_embedding_model_name is not None:
            return self._get_vector_field_name(
                self.sparse_embedding_model_name, prefix="fast-sparse"
            )
        return None

    def _scored_points_to_query_responses(
        self,
        scored_points: List[types.ScoredPoint],
    ) -> List[QueryResponse]:
        def _get_embedding(
            scored_point: types.ScoredPoint, field_name: Optional[str]
        ) -> Optional[List[float]]:
            if field_name is None:
                return None
            vector = (
                scored_point.vector.get(field_name, None)
                if isinstance(scored_point.vector, Dict)
                else None
            )
            return vector

        response = []
        # If models were not set, we can't extract embeddings.
        # Should we look for prefixes `fast-`, `fast-image`, `fast-sparse`?

        vector_field_name = self.get_vector_field_name()
        image_vector_field_name = self.get_image_vector_field_name()
        sparse_vector_field_name = self.get_sparse_vector_field_name()

        for scored_point in scored_points:
            embedding = _get_embedding(scored_point, vector_field_name)
            sparse_embedding = _get_embedding(scored_point, sparse_vector_field_name)
            image_embedding = _get_embedding(scored_point, image_vector_field_name)

            response.append(
                QueryResponse(
                    id=scored_point.id,
                    embedding=embedding,
                    image_embedding=image_embedding,
                    sparse_embedding=sparse_embedding,
                    metadata=scored_point.payload,
                    document=scored_point.payload.get("document", ""),
                    image_path=scored_point.payload.get("image_path", ""),
                    score=scored_point.score,
                )
            )
        return response

    def _points_iterator(
        self,
        ids: Optional[Iterable[models.ExtendedPointId]],
        metadata: Optional[Iterable[Dict[str, Any]]],
        documents: Optional[Iterable[str]],
        images: Optional[Iterable[PathInput]],
        text_vectors: Optional[Iterable[List[float]]],
        image_vectors: Optional[Iterable[List[float]]],
        sparse_vectors: Optional[Iterable[types.SparseVector]],
        ids_accumulator: list,
    ) -> Iterable[models.PointStruct]:
        assert (documents is None) is (
            text_vectors is None
        )  # docs and vectors should be both None or not None
        assert (sparse_vectors is None) or (
            documents is not None
        )  # sparse vectors should be None if docs are None
        assert (images is None) is (image_vectors is None)

        if ids is None:
            ids = iter(lambda: uuid.uuid4().hex, None)

        if metadata is None:
            metadata = iter(lambda: {}, None)

        if documents is None:
            documents = iter(lambda: None, True)
            text_vectors = iter(lambda: None, True)

        if sparse_vectors is None:
            sparse_vectors = iter(lambda: None, True)

        if images is None:
            images = iter(lambda: None, True)
            image_vectors = iter(lambda: None, True)

        vector_name = self.get_vector_field_name()
        image_vector_name = self.get_image_vector_field_name()
        sparse_vector_name = self.get_sparse_vector_field_name()

        for idx, meta, doc, text_vector, image_path, image_vector, sparse_vector in zip(
            ids, metadata, documents, text_vectors, images, image_vectors, sparse_vectors
        ):
            ids_accumulator.append(idx)
            point_vector: Dict[str, models.Vector] = {}

            if doc is not None:
                assert text_vector is not None or sparse_vector is not None
                meta["document"] = doc

            if text_vector is not None:
                assert vector_name is not None
                point_vector[vector_name] = text_vector

            if image_path is not None:
                assert image_vector is not None
                assert image_vector_name is not None
                meta["image_path"] = image_path
                point_vector[image_vector_name] = image_vector

            if sparse_vector is not None:
                assert sparse_vector_name is not None
                point_vector[sparse_vector_name] = sparse_vector

            yield models.PointStruct(id=idx, payload=meta, vector=point_vector)

    def _validate_collection_info(self, collection_info: models.CollectionInfo) -> None:
        params_map = {}
        if vector_field_name := self.get_vector_field_name():
            params_map[vector_field_name] = self._get_text_model_params(
                model_name=self.embedding_model_name
            )
        if image_vector_field_name := self.get_image_vector_field_name():
            assert self.image_embedding_model_name is not None
            params_map[image_vector_field_name] = self._get_image_model_params(
                model_name=self.image_embedding_model_name
            )

        assert all(
            field in collection_info.config.params.vectors for field in params_map
        ), f"Collection have incompatible vector params: {collection_info.config.params.vectors}, expected {params_map.keys()}"

        for vector_field_name, params in params_map.items():
            vector_params = collection_info.config.params.vectors[vector_field_name]
            embeddings_size, distance = params

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

    def get_fastembed_vector_params(
        self,
        on_disk: Optional[bool] = None,
        quantization_config: Optional[models.QuantizationConfig] = None,
        hnsw_config: Optional[models.HnswConfigDiff] = None,
    ) -> Dict[str, models.VectorParams]:
        """
        Generates vector configuration, compatible with fastembed models.

        Args:
            on_disk: if True, vectors will be stored on disk. If None, default value will be used.
            quantization_config: Quantization configuration. If None, quantization will be disabled.
            hnsw_config: HNSW configuration. If None, default configuration will be used.

        Returns:
            Configuration for `vectors_config` argument in `create_collection` method.
        """
        params = {}

        if text_vector_field_name := self.get_vector_field_name():
            embeddings_size, distance = self._get_text_model_params(
                model_name=self.embedding_model_name
            )
            params[text_vector_field_name] = models.VectorParams(
                size=embeddings_size,
                distance=distance,
                on_disk=on_disk,
                quantization_config=quantization_config,
                hnsw_config=hnsw_config,
            )

        if image_vector_field_name := self.get_image_vector_field_name():
            assert self.image_embedding_model_name is not None
            embeddings_size, distance = self._get_image_model_params(
                model_name=self.image_embedding_model_name
            )
            params[image_vector_field_name] = models.VectorParams(
                size=embeddings_size,
                distance=distance,
                on_disk=on_disk,
                quantization_config=quantization_config,
                hnsw_config=hnsw_config,
            )

        return params

    def get_fastembed_sparse_vector_params(
        self,
        on_disk: Optional[bool] = None,
    ) -> Optional[Dict[str, models.SparseVectorParams]]:
        """
        Generates vector configuration, compatible with fastembed sparse models.

        Args:
            on_disk: if True, vectors will be stored on disk. If None, default value will be used.

        Returns:
            Configuration for `vectors_config` argument in `create_collection` method.
        """
        vector_field_name = self.get_sparse_vector_field_name()
        if vector_field_name is None:
            return None
        return {
            vector_field_name: models.SparseVectorParams(
                index=models.SparseIndexParams(
                    on_disk=on_disk,
                )
            )
        }

    def add(
        self,
        collection_name: str,
        documents: Optional[Iterable[str]] = None,
        metadata: Optional[Iterable[Dict[str, Any]]] = None,
        ids: Optional[Iterable[models.ExtendedPointId]] = None,
        batch_size: int = 32,
        parallel: Optional[int] = None,
        images: Optional[Iterable[PathInput]] = None,
        **kwargs: Any,
    ) -> List[Union[str, int]]:
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
            metadata (Iterable[Dict[str, Any]], optional):
                List of metadata dicts. Defaults to None.
            ids (Iterable[models.ExtendedPointId], optional):
                List of ids to assign to documents.
                If not specified, UUIDs will be generated. Defaults to None.
            batch_size (int, optional):
                How many documents to embed and upload in single request. Defaults to 32.
            parallel (int, optional):
                How many parallel workers to use for embedding. Defaults to None.
                If number is specified, data-parallel process will be used.
            images: (Iterable[Union[str, Path], optional): List of paths to images to embed and add to the collection.

        Raises:
            ImportError: If fastembed is not installed.

        Returns:
            List of IDs of added items. If no ids provided, UUIDs will be randomly generated on client side.

        """

        dense_text_embeddings = None
        sparse_embeddings = None
        if documents is not None:
            documents, documents_iter = tee(documents, 2)
            dense_text_embeddings = self._embed_documents(
                documents=documents_iter,
                embedding_model_name=self.embedding_model_name,
                batch_size=batch_size,
                embed_type="passage",
                parallel=parallel,
            )
            if self.sparse_embedding_model_name:
                documents, documents_iter = tee(documents, 2)
                sparse_embeddings = self._sparse_embed_documents(
                    documents=documents_iter,
                    embedding_model_name=self.sparse_embedding_model_name,
                    batch_size=batch_size,
                    parallel=parallel,
                )

        image_embeddings = None
        if images is not None and self.image_embedding_model_name:
            image_embeddings = self._embed_images(
                images=images,
                embedding_model_name=self.image_embedding_model_name,
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
            documents=documents,
            text_vectors=dense_text_embeddings,
            image_vectors=image_embeddings,
            images=images,
            ids_accumulator=inserted_ids,
            sparse_vectors=sparse_embeddings,
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

    def verify_text_query_vector_name(self, vector_name: str) -> None:
        text_vector_name = self.get_vector_field_name() or self._get_vector_field_name(
            self.DEFAULT_EMBEDDING_MODEL
        )
        image_vector_name = self.get_image_vector_field_name()
        sparse_vector_name = self.get_sparse_vector_field_name()

        if vector_name not in (text_vector_name, image_vector_name, sparse_vector_name):
            raise ValueError(
                f"Vector name {vector_name} does not match any of the available vector names: "
                f"{text_vector_name}, {image_vector_name}, {sparse_vector_name}. Please provide a valid vector name."
            )

    def verify_image_query_vector_name(self, vector_name: str) -> None:
        text_vector_name = self.get_vector_field_name() or self._get_vector_field_name(
            self.DEFAULT_EMBEDDING_MODEL
        )
        image_vector_name = self.get_image_vector_field_name()
        sparse_vector_name = self.get_sparse_vector_field_name()
        if vector_name not in (text_vector_name, image_vector_name, sparse_vector_name):
            raise ValueError(
                f"Vector name {vector_name} does not match any of the available vector names: "
                f"{text_vector_name}, {image_vector_name}, {sparse_vector_name}. Please provide a valid vector name."
            )
        if vector_name == sparse_vector_name:
            raise ValueError(
                f"Sparse vectors are not supported for image queries. Please provide a valid vector name."
            )

    def _query_text(
        self,
        collection_name: str,
        query_text: Optional[Union[str, Dict[str, str]]] = None,
        query_filter: Optional[models.Filter] = None,
        limit: int = 10,
        **kwargs: Any,
    ) -> List[QueryResponse]:
        vector_name = None
        query_vector, sparse_query_vector = None, None
        sparse_vector_name = self.get_sparse_vector_field_name()

        if isinstance(query_text, dict):
            vector_name, query_text = next(iter(query_text.items()))
            self.verify_text_query_vector_name(vector_name)

        if vector_name is None or vector_name != sparse_vector_name:
            embedding_model_inst = (
                self._get_or_init_text_model(  # initialize either provided or default model
                    model_name=self.embedding_model_name
                )
            )
            embeddings = list(embedding_model_inst.query_embed(query=query_text))
            query_vector = embeddings[0].tolist()

        if sparse_vector_name and vector_name in (None, sparse_vector_name):
            sparse_embedding_model_inst = self._get_or_init_sparse_model(
                model_name=self.sparse_embedding_model_name
            )
            sparse_vector = list(sparse_embedding_model_inst.embed(documents=query_text))[0]
            sparse_query_vector = models.SparseVector(
                indices=sparse_vector.indices.tolist(),
                values=sparse_vector.values.tolist(),
            )

        dense_vector_name = vector_name if vector_name else self.get_vector_field_name()

        if sparse_query_vector is None:
            assert query_vector is not None
            return self._scored_points_to_query_responses(
                self.search(
                    collection_name=collection_name,
                    query_vector=models.NamedVector(name=dense_vector_name, vector=query_vector),
                    query_filter=query_filter,
                    limit=limit,
                    with_payload=True,
                    **kwargs,
                )
            )

        if query_vector is None:
            assert sparse_query_vector is not None
            return self._scored_points_to_query_responses(
                self.search(
                    collection_name=collection_name,
                    query_vector=models.NamedSparseVector(
                        name=sparse_vector_name, vector=sparse_query_vector
                    ),
                    query_filter=query_filter,
                    limit=limit,
                    with_payload=True,
                    **kwargs,
                )
            )

        dense_request = models.SearchRequest(
            vector=models.NamedVector(
                name=dense_vector_name,
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

    def _query_image(
        self,
        collection_name: str,
        query_image: Optional[Union[PathInput, Dict[str, PathInput]]] = None,
        query_filter: Optional[models.Filter] = None,
        limit: int = 10,
        **kwargs: Any,
    ):
        if self.image_embedding_model_name is None:
            raise ValueError(
                "Image query is provided, but image embedding model is not set. "
                "Please set image embedding model using `set_image_model` method."
            )

        vector_name = None
        if isinstance(query_image, dict):
            vector_name, query_image = next(iter(query_image.items()))
            self.verify_image_query_vector_name(vector_name)
        vector_name = vector_name if vector_name else self.get_image_vector_field_name()

        embedding_model_inst = self._get_or_init_image_model(
            model_name=self.image_embedding_model_name
        )
        embeddings = list(embedding_model_inst.embed([query_image]))

        query_vector = embeddings[0].tolist()
        return self._scored_points_to_query_responses(
            self.search(
                collection_name=collection_name,
                query_vector=models.NamedVector(name=vector_name, vector=query_vector),
                query_filter=query_filter,
                limit=limit,
                with_payload=True,
                **kwargs,
            )
        )

    def query(
        self,
        collection_name: str,
        query_text: Optional[Union[str, Dict[str, str]]] = None,
        query_image: Optional[Union[PathInput, Dict[str, PathInput]]] = None,
        query_filter: Optional[models.Filter] = None,
        limit: int = 10,
        **kwargs: Any,
    ) -> List[QueryResponse]:
        """
        Search for documents in a collection.
        This method automatically embeds the query text using the specified embedding model.
        If you want to use your own query vector, use `search` method instead.

        Args:
            collection_name: Collection to search in
            query_text:
                Text to search for. This text will be embedded using the specified embedding model.
                And then used as a query vector. Can be set as a dict, where a key is a vector field name to search
                over. Useful in a multimodal setup.
            query_image:
                Path to the image to search for. This image will be embedded using the specified image embedding model.
                And then used as a query vector. Can be set as a dict, where a key is a vector field name to search
                over. Useful in a multimodal setup
            query_filter:
                - Exclude vectors which doesn't fit given conditions.
                - If `None` - search among all vectors
            limit: How many results return
            **kwargs: Additional search parameters. See `qdrant_client.models.SearchRequest` for details.

        Raises:
            ValueError: If neither query_text nor query_image is provided.

        Returns:
            List[types.ScoredPoint]: List of scored points.

        """
        if (query_text is None) is (query_image is None):
            raise ValueError("Either query_text or query_image should be provided")

        if query_text is not None:
            return self._query_text(
                collection_name=collection_name,
                query_text=query_text,
                query_filter=query_filter,
                limit=limit,
                **kwargs,
            )
        else:
            return self._query_image(
                collection_name=collection_name,
                query_image=query_image,
                query_filter=query_filter,
                limit=limit,
                **kwargs,
            )

    def query_batch(
        self,
        collection_name: str,
        query_texts: Optional[Union[List[str], Dict[str, str]]] = None,
        query_images: Optional[Union[List[PathInput], Dict[str, PathInput]]] = None,
        query_filter: Optional[models.Filter] = None,
        limit: int = 10,
        **kwargs: Any,
    ) -> List[List[QueryResponse]]:
        """
        Search for documents in a collection with batched query.
        This method automatically embeds the query text using the specified embedding model.

        Args:
            collection_name: Collection to search in
            query_texts:
                A list of texts to search for. Might be specified as a dict with vector names as keys.
                Each text will be embedded using the specified embedding model.
                And then used as a query vector for a separate search requests.
            query_images:
                A list of paths to the images to search for. Might be specified as a dict with vector names as keys.
                Each image will be embedded using the specified image
                embedding model. And then used as a query vector for a separate search requests.
            query_filter:
                - Exclude vectors which doesn't fit given conditions.
                - If `None` - search among all vectors
                This filter will be applied to all search requests.
            limit: How many results return
            **kwargs: Additional search parameters. See `qdrant_client.models.SearchRequest` for details.

        Returns:
            List[List[QueryResponse]]: List of lists of responses for each query text.

        """

        if self.sparse_embedding_model_name and not self._embedding_model_name:
            raise ValueError(
                "Sparse embedding models are set, but no dense embedding model is set. "
                "Please set dense embedding model using `set_model` method."
            )

        if (query_texts is None) and (query_images is None):
            raise ValueError("Either query_texts or query_images should be provided")

        result = []
        if query_texts is not None:
            result.extend(
                self._query_text_batch(
                    collection_name=collection_name,
                    query_texts=query_texts,
                    query_filter=query_filter,
                    limit=limit,
                    **kwargs,
                )
            )
        if query_images is not None:
            result.extend(
                self._query_image_batch(
                    collection_name=collection_name,
                    query_images=query_images,
                    query_filter=query_filter,
                    limit=limit,
                    **kwargs,
                )
            )

        return result

    def _query_text_batch(
        self,
        collection_name: str,
        query_texts: Union[List[str], Dict[str, str]],
        query_filter: Optional[models.Filter] = None,
        limit: int = 10,
        **kwargs: Any,
    ) -> List[List[QueryResponse]]:
        embedding_model_inst: TextEmbedding = self._get_or_init_text_model(
            model_name=self.embedding_model_name
        )
        vector_names: Union[Iterable, List]

        if isinstance(query_texts, dict):
            vector_names = list(query_texts.keys())
            query_texts = list(query_texts.values())
        else:
            default_vector_name = self.get_vector_field_name()
            vector_names = iter(lambda: default_vector_name, True)

        query_vectors = list(embedding_model_inst.query_embed(query=query_texts))
        requests = []
        for vector_name, vector in zip(vector_names, query_vectors):
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
                indices=sparse_vector.indices.tolist(), values=sparse_vector.values.tolist()
            )
            for sparse_vector in sparse_embedding_model_inst.embed(documents=query_texts)
        ]
        sparse_vector_name = self.get_sparse_vector_field_name()
        for sparse_vector in sparse_query_vectors:
            request = models.SearchRequest(
                vector=models.NamedSparseVector(
                    name=sparse_vector_name,
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

    def _query_image_batch(
        self,
        collection_name: str,
        query_images: Union[List[PathInput], Dict[str, PathInput]],
        query_filter: Optional[models.Filter] = None,
        limit: int = 10,
        **kwargs: Any,
    ) -> List[List[QueryResponse]]:
        if self.image_embedding_model_name is None:
            raise ValueError(
                "Image query is provided, but image embedding model is not set. "
                "Please set image embedding model using `set_image_model` method."
            )
        embedding_model_inst: ImageEmbedding = self._get_or_init_image_model(
            model_name=self.image_embedding_model_name, image=True
        )
        vector_names: Union[Iterable, List]
        if isinstance(query_images, dict):
            vector_names = list(query_images.keys())
            query_images = list(query_images.values())
        else:
            default_vector_name = self.get_vector_field_name()
            vector_names = iter(lambda: default_vector_name, True)

        query_vectors = list(embedding_model_inst.embed(images=query_images))
        requests = []
        for vector_name, vector in zip(vector_names, query_vectors):
            request = models.SearchRequest(
                vector=models.NamedVector(name=vector_name, vector=vector.tolist()),
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

        return [self._scored_points_to_query_responses(response) for response in responses]
