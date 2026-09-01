"""Client for Qdrant Serverless.

Serverless exposes the same point-level API as a regular Qdrant cluster (minus
read consistency, shard selection, write ordering and filtered updates), but a
much simpler, tenant-facing collection management API. Point operations are
delegated to the regular gRPC client; collection operations talk to the
serverless CollectionsService.
"""

from typing import Any, Optional, Sequence

from qdrant_client.conversions import common_types as types
from qdrant_client.qdrant_remote import QdrantRemote
from qdrant_client.serverless import models as serverless_models
from qdrant_client.serverless.conversions import (
    collection_config_from_grpc,
    collection_config_to_grpc,
)
from qdrant_client.serverless.grpc import serverless_collections_pb2 as pb2
from qdrant_client.serverless.grpc.serverless_collections_pb2_grpc import CollectionsServiceStub

# Serverless is exposed on the standard TLS port, not on qdrant's 6334.
DEFAULT_SERVERLESS_GRPC_PORT = 443


class QdrantServerless:
    """Entry point to a Qdrant Serverless space.

    Examples:

        >>> client = QdrantServerless(
        ...     url="https://serverless.example.cloud.qdrant.io",
        ...     api_key="<your api key>",
        ... )
        >>> client.create_collection(
        ...     "my-collection",
        ...     dense_vectors=DenseVectorConfig(size=1536, distance=Distance.COSINE),
        ... )

    Args:
        url: Base url of the serverless space, e.g. `https://serverless.example.cloud.qdrant.io`
        api_key: API key of the serverless space, sent as `api-key` metadata with every request
        grpc_port: Port of the gRPC interface. Default: 443
        timeout: Timeout for gRPC requests in seconds. Default: 5 seconds
        grpc_options: Additional low-level gRPC channel options
    """

    def __init__(
        self,
        url: str,
        api_key: Optional[str] = None,
        grpc_port: int = DEFAULT_SERVERLESS_GRPC_PORT,
        timeout: Optional[int] = None,
        grpc_options: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ):
        self._remote = QdrantRemote(
            url=url,
            api_key=api_key,
            grpc_port=grpc_port,
            prefer_grpc=True,
            timeout=timeout,
            grpc_options=grpc_options,
            check_compatibility=False,
            **kwargs,
        )
        self._grpc_collections: Optional[CollectionsServiceStub] = None

    @property
    def _collections(self) -> CollectionsServiceStub:
        if self._grpc_collections is None:
            # reuse the delegate's channel: same host, tls, api-key metadata and options
            self._remote._init_grpc_channel()
            self._grpc_collections = CollectionsServiceStub(self._remote._grpc_channel_pool[0])
        return self._grpc_collections

    def _collections_timeout(self, timeout: Optional[int]) -> int:
        return timeout if timeout is not None else self._remote._timeout

    def close(self, grpc_grace: Optional[float] = None, **kwargs: Any) -> None:
        """Closes the underlying gRPC connections."""
        self._grpc_collections = None
        self._remote.close(grpc_grace=grpc_grace, **kwargs)

    def __enter__(self) -> "QdrantServerless":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    # region collections

    def create_collection(
        self,
        collection_name: str,
        dense_vectors: serverless_models.DenseVectorConfig
        | dict[str, serverless_models.DenseVectorConfig]
        | None = None,
        sparse_vectors: serverless_models.SparseVectorConfig
        | dict[str, serverless_models.SparseVectorConfig]
        | None = None,
        payload_indexes: dict[str, serverless_models.PayloadIndex] | None = None,
        timeout: Optional[int] = None,
    ) -> str:
        """Creates a collection.

        At least one dense or sparse vector is required. A bare (non-dict)
        vector config is registered as the unnamed default vector, like in
        regular qdrant.

        Returns:
            Outcome, e.g. "created" or "already exists"
        """
        if isinstance(dense_vectors, serverless_models.DenseVectorConfig):
            dense_vectors = {"": dense_vectors}
        if isinstance(sparse_vectors, serverless_models.SparseVectorConfig):
            sparse_vectors = {"": sparse_vectors}
        config = serverless_models.CollectionConfig(
            dense_vectors=dense_vectors or {},
            sparse_vectors=sparse_vectors or {},
            payload_indexes=payload_indexes or {},
        )
        response = self._collections.CreateCollection(
            pb2.CreateCollectionRequest(
                collection_name=collection_name,
                config=collection_config_to_grpc(config),
            ),
            timeout=self._collections_timeout(timeout),
        )
        return response.result

    def delete_collection(self, collection_name: str, timeout: Optional[int] = None) -> bool:
        """Deletes a collection and all of its data.

        Returns:
            True if the collection existed and was deleted
        """
        response = self._collections.DeleteCollection(
            pb2.DeleteCollectionRequest(collection_name=collection_name),
            timeout=self._collections_timeout(timeout),
        )
        return response.deleted

    def get_collection(
        self, collection_name: str, timeout: Optional[int] = None
    ) -> serverless_models.CollectionInfo:
        """Returns a collection's configuration and stats.

        Does not raise if the collection is missing: check `.exists`.
        """
        response = self._collections.GetCollection(
            pb2.GetCollectionRequest(collection_name=collection_name),
            timeout=self._collections_timeout(timeout),
        )
        return serverless_models.CollectionInfo(
            exists=response.exists,
            config=collection_config_from_grpc(response.config)
            if response.HasField("config")
            else None,
            point_count=response.point_count if response.HasField("point_count") else None,
        )

    def collection_exists(self, collection_name: str, timeout: Optional[int] = None) -> bool:
        """Checks whether a collection exists."""
        return self.get_collection(collection_name, timeout=timeout).exists

    def get_collections(
        self, timeout: Optional[int] = None
    ) -> list[serverless_models.CollectionSummary]:
        """Lists the collections of the space, ordered by name."""
        response = self._collections.ListCollections(
            pb2.ListCollectionsRequest(),
            timeout=self._collections_timeout(timeout),
        )
        return [
            serverless_models.CollectionSummary(
                collection_name=collection.collection_name,
                point_count=collection.point_count
                if collection.HasField("point_count")
                else None,
            )
            for collection in response.collections
        ]

    # endregion

    # region points
    # Same semantics as the regular client, minus parameters serverless does not
    # support: read consistency, shard selection, write ordering, filtered updates.

    def query_points(
        self,
        collection_name: str,
        query: types.PointId
        | list[float]
        | list[list[float]]
        | types.SparseVector
        | types.Query
        | types.NumpyArray
        | types.Document
        | types.Image
        | types.InferenceObject
        | None = None,
        using: Optional[str] = None,
        prefetch: types.Prefetch | list[types.Prefetch] | None = None,
        query_filter: Optional[types.Filter] = None,
        search_params: Optional[types.SearchParams] = None,
        limit: int = 10,
        offset: Optional[int] = None,
        with_payload: bool | Sequence[str] | types.PayloadSelector = True,
        with_vectors: bool | Sequence[str] = False,
        score_threshold: Optional[float] = None,
        timeout: Optional[int] = None,
    ) -> types.QueryResponse:
        """Universal endpoint to run any available operation, such as search,
        recommendation, discovery, context search. Same as in the regular client."""
        return self._remote.query_points(
            collection_name=collection_name,
            query=query,
            using=using,
            prefetch=prefetch,
            query_filter=query_filter,
            search_params=search_params,
            limit=limit,
            offset=offset,
            with_payload=with_payload,
            with_vectors=with_vectors,
            score_threshold=score_threshold,
            timeout=timeout,
        )

    def retrieve(
        self,
        collection_name: str,
        ids: Sequence[types.PointId],
        with_payload: bool | Sequence[str] | types.PayloadSelector = True,
        with_vectors: bool | Sequence[str] = False,
        timeout: Optional[int] = None,
    ) -> list[types.Record]:
        """Retrieves points by ids."""
        return self._remote.retrieve(
            collection_name=collection_name,
            ids=ids,
            with_payload=with_payload,
            with_vectors=with_vectors,
            timeout=timeout,
        )

    def scroll(
        self,
        collection_name: str,
        scroll_filter: Optional[types.Filter] = None,
        limit: int = 10,
        order_by: Optional[types.OrderBy] = None,
        offset: Optional[types.PointId] = None,
        with_payload: bool | Sequence[str] | types.PayloadSelector = True,
        with_vectors: bool | Sequence[str] = False,
        timeout: Optional[int] = None,
    ) -> tuple[list[types.Record], Optional[types.PointId]]:
        """Iterates over all points, optionally filtered.

        Returns a page of points and the offset of the next page (None if done).
        """
        return self._remote.scroll(
            collection_name=collection_name,
            scroll_filter=scroll_filter,
            limit=limit,
            order_by=order_by,
            offset=offset,
            with_payload=with_payload,
            with_vectors=with_vectors,
            timeout=timeout,
        )

    def count(
        self,
        collection_name: str,
        count_filter: Optional[types.Filter] = None,
        exact: bool = True,
        timeout: Optional[int] = None,
    ) -> types.CountResult:
        """Counts points, optionally filtered."""
        return self._remote.count(
            collection_name=collection_name,
            count_filter=count_filter,
            exact=exact,
            timeout=timeout,
        )

    def upsert(
        self,
        collection_name: str,
        points: types.Points,
        wait: bool = True,
        timeout: Optional[int] = None,
    ) -> types.UpdateResult:
        """Inserts or updates points."""
        return self._remote.upsert(
            collection_name=collection_name,
            points=points,
            wait=wait,
            timeout=timeout,
        )

    def delete(
        self,
        collection_name: str,
        ids: Sequence[types.PointId],
        wait: bool = True,
        timeout: Optional[int] = None,
    ) -> types.UpdateResult:
        """Deletes points by ids. Serverless does not support deletion by filter."""
        return self._remote.delete(
            collection_name=collection_name,
            points_selector=list(ids),
            wait=wait,
            timeout=timeout,
        )

    def set_payload(
        self,
        collection_name: str,
        payload: types.Payload,
        ids: Sequence[types.PointId],
        key: Optional[str] = None,
        wait: bool = True,
        timeout: Optional[int] = None,
    ) -> types.UpdateResult:
        """Merges the given payload into the payload of the given points."""
        return self._remote.set_payload(
            collection_name=collection_name,
            payload=payload,
            points=list(ids),
            key=key,
            wait=wait,
            timeout=timeout,
        )

    def delete_payload(
        self,
        collection_name: str,
        keys: Sequence[str],
        ids: Sequence[types.PointId],
        wait: bool = True,
        timeout: Optional[int] = None,
    ) -> types.UpdateResult:
        """Removes the given payload keys from the given points."""
        return self._remote.delete_payload(
            collection_name=collection_name,
            keys=keys,
            points=list(ids),
            wait=wait,
            timeout=timeout,
        )

    # endregion
