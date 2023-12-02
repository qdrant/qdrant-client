import uuid
from itertools import tee
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

from pydantic import BaseModel

from qdrant_client.client_base import QdrantBase
from qdrant_client.conversions import common_types as types
from qdrant_client.fastembed_common import QueryResponse
from qdrant_client.http import models
from qdrant_client.uploader.uploader import iter_batch

try:
    from fastembed.embedding import DefaultEmbedding
except ImportError:
    DefaultEmbedding = None

SUPPORTED_EMBEDDING_MODELS: Dict[str, Tuple[int, models.Distance]] = {
    "BAAI/bge-base-en": (768, models.Distance.COSINE),
    "sentence-transformers/all-MiniLM-L6-v2": (384, models.Distance.COSINE),
    "BAAI/bge-small-en": (384, models.Distance.COSINE),
    "BAAI/bge-small-en-v1.5": (384, models.Distance.COSINE),
    "BAAI/bge-base-en-v1.5": (768, models.Distance.COSINE),
    "intfloat/multilingual-e5-large": (1024, models.Distance.COSINE),
}


class QdrantFastembedMixin(QdrantBase):
    DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-en"

    embedding_models: Dict[str, "DefaultEmbedding"] = {}

    def __init__(self, **kwargs: Any):
        self.embedding_model_name = self.DEFAULT_EMBEDDING_MODEL
        super().__init__(**kwargs)

    def set_model(self, embedding_model_name: str) -> None:
        """
        Set embedding model to use for encoding documents and queries.
        Args:
            embedding_model_name: One of the supported embedding models. See `SUPPORTED_EMBEDDING_MODELS` for details.

        Raises:
            ValueError: If embedding model is not supported.
            ImportError: If fastembed is not installed.

        Returns:
            None
        """
        self._get_or_init_model(model_name=embedding_model_name)
        self.embedding_model_name = embedding_model_name

    @staticmethod
    def _import_fastembed() -> None:
        try:
            from fastembed.embedding import DefaultEmbedding
        except ImportError:
            # If it's not, ask the user to install it
            raise ImportError(
                "fastembed is not installed."
                " Please install it to enable fast vector indexing with `pip install fastembed`."
            )

    @classmethod
    def _get_model_params(cls, model_name: str) -> Tuple[int, models.Distance]:
        if model_name not in SUPPORTED_EMBEDDING_MODELS:
            raise ValueError(
                f"Unsupported embedding model: {model_name}. Supported models: {SUPPORTED_EMBEDDING_MODELS}"
            )

        return SUPPORTED_EMBEDDING_MODELS[model_name]

    @classmethod
    def _get_or_init_model(
        cls, model_name: str
    ) -> "DefaultEmbedding":  # -> Embedding: # noqa: F821
        if model_name in cls.embedding_models:
            return cls.embedding_models[model_name]

        if model_name not in SUPPORTED_EMBEDDING_MODELS:
            raise ValueError(
                f"Unsupported embedding model: {model_name}. Supported models: {SUPPORTED_EMBEDDING_MODELS}"
            )

        cls._import_fastembed()

        cls.embedding_models[model_name] = DefaultEmbedding(model_name=model_name)
        return cls.embedding_models[model_name]

    def _embed_documents(
        self,
        documents: Iterable[str],
        embedding_model_name: str = DEFAULT_EMBEDDING_MODEL,
        batch_size: int = 32,
        embed_type: str = "default",
        parallel: Optional[int] = None,
    ) -> Iterable[Tuple[str, List[float]]]:
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

    def get_vector_field_name(self) -> str:
        """
        Returns name of the vector field in qdrant collection, used by current fastembed model.
        Returns:
            Name of the vector field.
        """
        model_name = self.embedding_model_name.split("/")[-1].lower()
        return f"fast-{model_name}"

    def _scored_points_to_query_responses(
        self,
        scored_points: List[types.ScoredPoint],
    ) -> List[QueryResponse]:
        response = []
        for scored_point in scored_points:
            embedding = None
            if scored_point.vector is not None:
                embedding = scored_point.vector.get(self.get_vector_field_name(), None)

            response.append(
                QueryResponse(
                    id=scored_point.id,
                    embedding=embedding,
                    metadata=scored_point.payload,
                    document=scored_point.payload.get("document", ""),
                    score=scored_point.score,
                )
            )
        return response

    def _records_iterator(
        self,
        ids: Optional[Iterable[models.ExtendedPointId]],
        metadata: Optional[Iterable[Dict[str, Any]]],
        encoded_docs: Iterable[Tuple[str, List[float]]],
        ids_accumulator: list,
    ) -> Iterable[models.Record]:
        if ids is None:
            ids = iter(lambda: uuid.uuid4().hex, None)

        if metadata is None:
            metadata = iter(lambda: {}, None)

        vector_name = self.get_vector_field_name()

        for idx, meta, (doc, vector) in zip(ids, metadata, encoded_docs):
            ids_accumulator.append(idx)
            payload = {"document": doc, **meta}
            yield models.Record(id=idx, payload=payload, vector={vector_name: vector})

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

    def add(
        self,
        collection_name: str,
        documents: Iterable[str],
        metadata: Optional[Iterable[Dict[str, Any]]] = None,
        ids: Optional[Iterable[models.ExtendedPointId]] = None,
        batch_size: int = 32,
        parallel: Optional[int] = None,
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

        embeddings_size, distance = self._get_model_params(model_name=self.embedding_model_name)
        vector_field_name = self.get_vector_field_name()

        # Check if collection by same name exists, if not, create it
        try:
            collection_info = self.get_collection(collection_name=collection_name)
        except Exception:
            self.create_collection(
                collection_name=collection_name,
                vectors_config=self.get_fastembed_vector_params(),
            )
            collection_info = self.get_collection(collection_name=collection_name)

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

        inserted_ids: list = []

        records = self._records_iterator(
            ids=ids,
            metadata=metadata,
            encoded_docs=encoded_docs,
            ids_accumulator=inserted_ids,
        )

        self.upload_records(
            collection_name=collection_name,
            records=records,
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
    ) -> List[QueryResponse]:
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
            List[types.ScoredPoint]: List of scored points.

        """
        embedding_model_inst = self._get_or_init_model(model_name=self.embedding_model_name)
        embeddings = list(embedding_model_inst.query_embed(query=query_text))
        query_vector = embeddings[0]

        return self._scored_points_to_query_responses(
            self.search(
                collection_name=collection_name,
                query_vector=models.NamedVector(
                    name=self.get_vector_field_name(), vector=query_vector.tolist()
                ),
                query_filter=query_filter,
                limit=limit,
                with_payload=True,
                **kwargs,
            )
        )

    def query_batch(
        self,
        collection_name: str,
        query_texts: List[str],
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
                A list of texts to search for. Each text will be embedded using the specified embedding model.
                And then used as a query vector for a separate search requests.
            query_filter:
                - Exclude vectors which doesn't fit given conditions.
                - If `None` - search among all vectors
                This filter will be applied to all search requests.
            limit: How many results return
            **kwargs: Additional search parameters. See `qdrant_client.models.SearchRequest` for details.

        Returns:
            List[List[QueryResponse]]: List of lists of responses for each query text.

        """
        embedding_model_inst = self._get_or_init_model(model_name=self.embedding_model_name)
        query_vectors = [
            list(embedding_model_inst.query_embed(query=query_text))[0]
            for query_text in query_texts
        ]
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

        return [
            self._scored_points_to_query_responses(response)
            for response in self.search_batch(
                collection_name=collection_name,
                requests=requests,
            )
        ]
