import uuid
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

from qdrant_client.client_base import QdrantBase
from qdrant_client.conversions import common_types as types
from qdrant_client.http import models
from qdrant_client.uploader.uploader import iter_batch

try:
    from fastembed.embedding import DefaultEmbedding
except ImportError:
    pass

SUPPORTED_EMBEDDING_MODELS: Dict[str, Tuple[int, models.Distance]] = {
    "BAAI/bge-base-en": (768, models.Distance.COSINE),
    "sentence-transformers/all-MiniLM-L6-v2": (384, models.Distance.COSINE),
    "BAAI/bge-small-en": (384, models.Distance.COSINE),
}


class QdrantFastembedMixin(QdrantBase):
    DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

    embedding_models: Dict[str, "DefaultEmbedding"] = {}

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
    ) -> Iterable[List[float]]:
        embedding_model = self._get_or_init_model(model_name=embedding_model_name)
        for batch_docs in iter_batch(documents, batch_size):
            vectors_batches = embedding_model.encode(batch_docs, batch_size=batch_size)
            for vector_batch in vectors_batches:
                for vector in vector_batch.tolist():
                    yield vector

    def add(
        self,
        collection_name: str,
        documents: List[str],
        metadata: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[models.ExtendedPointId]] = None,
        batch_size: int = 32,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        **kwargs: Any,
    ) -> List[str]:
        """
        Adds text documents into qdrant collection.
        If collection does not exist, it will be created with default parameters.
        Metadata in combination with documents will be added as payload.
        Documents will be encoded using the specified embedding model.

        If you want to use your own encoded vectors, use `upsert` method instead.

        Args:
            collection_name (str):
                Name of the collection to add documents to.
            documents (List[str]):
                List of documents to encode and add to the collection.
            metadata (List[Dict[str, Any]], optional):
                List of metadata dicts. Defaults to None.
            ids (List[models.ExtendedPointId], optional):
                List of ids to assign to documents.
                If not specified, UUIDs will be generated. Defaults to None.
            batch_size (int, optional):
                How many documents to encode and upload in single request. Defaults to 32.
            embedding_model (str, optional):
                Which embedding model to use. Defaults to "fast-all-MiniLM-L6-v2".

        Raises:
            ImportError: If fastembed is not installed.

        Returns:
            List[str]: List of UUIDs of added documents. UUIDs are randomly generated on client side.

        """

        # check if we have fastembed installed
        embeddings = self._embed_documents(
            documents=documents,
            embedding_model_name=embedding_model,
            batch_size=batch_size,
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

        embeddings_size, distance = self._get_model_params(model_name=embedding_model)

        try:
            collection_info = self.get_collection(collection_name=collection_name)
        except Exception:
            self.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(size=embeddings_size, distance=distance),
            )
            collection_info = self.get_collection(collection_name=collection_name)

        assert not isinstance(
            collection_info.config.params.vectors, dict
        ), f"Collection have incompatible vector params: {collection_info.config.params.vectors}"

        assert (
            embeddings_size == collection_info.config.params.vectors.size
        ), f"Embedding size mismatch: {embeddings_size} != {collection_info.config.params.vectors.size}"

        assert (
            distance == collection_info.config.params.vectors.distance
        ), f"Distance mismatch: {distance} != {collection_info.config.params.vectors.distance}"

        self.upload_collection(
            collection_name=collection_name,
            vectors=embeddings,
            payload=payloads,
            ids=ids,
            wait=True,
            **kwargs,
        )

        return ids

    def query(
        self,
        collection_name: str,
        query_text: str,
        query_filter: Optional[models.Filter] = None,
        search_params: Optional[models.SearchParams] = None,
        limit: int = 10,
        offset: int = 0,
        with_payload: Union[bool, Sequence[str], models.PayloadSelector] = True,
        with_vectors: Union[bool, Sequence[str]] = True,
        score_threshold: Optional[float] = None,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        **kwargs: Any,
    ) -> List[types.ScoredPoint]:
        """
        Search for documents in a collection.
        This method automatically encodes the query text using the specified embedding model.
        If you want to use your own encoded query vector, use `search` method instead.

        Args:
            collection_name: Collection to search in
            query_text:
                Text to search for. This text will be encoded using the specified embedding model.
                And then used as a query vector.
            query_filter:
                - Exclude vectors which doesn't fit given conditions.
                - If `None` - search among all vectors
            search_params: Additional search params
            limit: How many results return
            offset:
                Offset of the first result to return.
                May be used to paginate results.
                Note: large offset values may cause performance issues.
            with_payload:
                - Specify which stored payload should be attached to the result.
                - If `True` - attach all payload
                - If `False` - do not attach any payload
                - If List of string - include only specified fields
                - If `PayloadSelector` - use explicit rules
            with_vectors:
                - If `True` - Attach stored vector to the search result.
                - If `False` - Do not attach vector.
                - If List of string - include only specified fields
                - Default: `False`
            score_threshold:
                Define a minimal score threshold for the result.
                If defined, less similar results will not be returned.
                Score of the returned result might be higher or smaller than the threshold depending
                on the Distance function used.
                E.g. for cosine similarity only higher scores will be returned.
            embedding_model:
                Which embedding model to use. Defaults to "fast-all-MiniLM-L6-v2".
                Models are automatically downloaded by `fastembed` library.

        Returns:
            List[types.ScoredPoint]: List of scored points.

        """
        embedding_model_inst = self._get_or_init_model(model_name=embedding_model)
        embeddings = list(embedding_model_inst.encode(documents=[query_text]))
        query_vector = embeddings[0][0]

        return self.search(
            collection_name=collection_name,
            query_vector=query_vector,
            query_filter=query_filter,
            search_params=search_params,
            limit=limit,
            offset=offset,
            with_payload=with_payload,
            with_vectors=with_vectors,
            score_threshold=score_threshold,
            **kwargs,
        )

    def query_batch(
        self,
        collection_name: str,
        query_texts: List[str],
        query_filter: Optional[models.Filter] = None,
        search_params: Optional[models.SearchParams] = None,
        limit: int = 10,
        offset: int = 0,
        with_payload: Union[bool, Sequence[str], models.PayloadSelector] = True,
        with_vectors: Union[bool, Sequence[str]] = True,
        score_threshold: Optional[float] = None,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        **kwargs: Any,
    ) -> List[List[types.ScoredPoint]]:
        """
        Search for documents in a collection with batched query.
        This method automatically encodes the query text using the specified embedding model.


        """
        embedding_model_inst = self._get_or_init_model(model_name=embedding_model)
        query_vectors = embedding_model_inst.encode(documents=query_texts)

        requests = []
        for vector in query_vectors:
            request = models.SearchRequest(
                vector=vector,
                filter=query_filter,
                params=search_params,
                limit=limit,
                offset=offset,
                with_payload=with_payload,
                with_vectors=with_vectors,
                score_threshold=score_threshold,
            )

            requests.append(request)

        return self.search_batch(
            collection_name=collection_name,
            requests=requests,
            **kwargs,
        )
