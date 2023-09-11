import uuid
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

from pydantic import BaseModel

from qdrant_client.client_base import QdrantBase
from qdrant_client.conversions import common_types as types
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
}


class QueryResponse(BaseModel, extra="forbid"):  # type: ignore
    id: Union[str, int]
    embedding: Optional[List[float]]
    metadata: Dict[str, Any]
    document: str
    score: float


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
        documents: List[str],
        embedding_model_name: str = DEFAULT_EMBEDDING_MODEL,
        batch_size: int = 32,
        embed_type: str = "default",
    ) -> Iterable[List[float]]:
        embedding_model = self._get_or_init_model(model_name=embedding_model_name)
        for batch_docs in iter_batch(documents, batch_size):
            if embed_type == "passage":
                vectors_batches = embedding_model.passage_embed(batch_docs, batch_size=batch_size)
            elif embed_type == "query":
                vectors_batches = (
                    list(embedding_model.query_embed(query=query))[0] for query in batch_docs
                )
            elif embed_type == "default":
                vectors_batches = embedding_model.embed(batch_docs, batch_size=batch_size)
            else:
                raise ValueError(f"Unknown embed type: {embed_type}")
            for vector in vectors_batches:
                yield vector.tolist()

    def _get_vector_field_name(self) -> str:
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
                embedding = scored_point.vector.get(self._get_vector_field_name(), None)

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

    def add(
        self,
        collection_name: str,
        documents: List[str],
        metadata: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[models.ExtendedPointId]] = None,
        batch_size: int = 32,
        **kwargs: Any,
    ) -> List[str]:
        """
        Adds text documents into qdrant collection.
        If collection does not exist, it will be created with default parameters.
        Metadata in combination with documents will be added as payload.
        Documents will be embedded using the specified embedding model.

        If you want to use your own vectors, use `upsert` method instead.

        Args:
            collection_name (str):
                Name of the collection to add documents to.
            documents (List[str]):
                List of documents to embed and add to the collection.
            metadata (List[Dict[str, Any]], optional):
                List of metadata dicts. Defaults to None.
            ids (List[models.ExtendedPointId], optional):
                List of ids to assign to documents.
                If not specified, UUIDs will be generated. Defaults to None.
            batch_size (int, optional):
                How many documents to embed and upload in single request. Defaults to 32.

        Raises:
            ImportError: If fastembed is not installed.

        Returns:
            List[str]: List of UUIDs of added documents. UUIDs are randomly generated on client side.

        """

        # check if we have fastembed installed
        embeddings = self._embed_documents(
            documents=documents,
            embedding_model_name=self.embedding_model_name,
            batch_size=batch_size,
            embed_type="passage",
        )

        if metadata is None:
            metadata = [{} for _ in range(len(documents))]
        else:
            assert len(metadata) == len(
                documents
            ), f"metadata length mismatch: {len(metadata)} != {len(documents)}"

        payloads = ({"document": doc, **metadata} for doc, metadata in zip(documents, metadata))

        if ids is None:
            ids = [uuid.uuid4().hex for _ in range(len(documents))]

        embeddings_size, distance = self._get_model_params(model_name=self.embedding_model_name)

        vector_field_name = self._get_vector_field_name()

        # Check if collection by same name exists, if not, create it
        try:
            collection_info = self.get_collection(collection_name=collection_name)
        except Exception:
            self.create_collection(
                collection_name=collection_name,
                vectors_config={
                    vector_field_name: models.VectorParams(size=embeddings_size, distance=distance)
                },
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

        records = (
            models.Record(id=idx, payload=payload, vector={vector_field_name: vector})
            for idx, payload, vector in zip(ids, payloads, embeddings)
        )

        self.upload_records(
            collection_name=collection_name,
            records=records,
            wait=True,
            **kwargs,
        )

        return ids

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
                    name=self._get_vector_field_name(), vector=query_vector.tolist()
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
                    name=self._get_vector_field_name(), vector=vector.tolist()
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
