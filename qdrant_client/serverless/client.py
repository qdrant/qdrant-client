"""Client for Qdrant Serverless.

Serverless exposes the same point-level API as a regular Qdrant cluster (minus
read consistency, shard selection, write ordering and filtered updates), but a
much simpler, tenant-facing collection management API. Point operations are
delegated to the regular gRPC client; collection operations talk to the
serverless CollectionsService.
"""

from typing import Any, Optional, Sequence

from qdrant_client.conversions import common_types as types
from qdrant_client.qdrant_fastembed import QdrantFastembedMixin
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

    Point operations behave like in the regular `QdrantClient`, except that
    parameters serverless does not support (read consistency, shard selection,
    write ordering, filtered updates) are not available. Collection management
    uses the simplified serverless API: only the tenant-facing configuration is
    exposed, storage internals (quantization, WAL, segments, ...) are decided
    by the serverless manager.

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
        url: Base url of the serverless space,
            e.g. `https://serverless.example.cloud.qdrant.io`
        api_key: API key of the serverless space,
            sent as `api-key` metadata with every request
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
        """Closes the underlying gRPC connections.

        The client is unusable afterwards; create a new instance to reconnect.

        Args:
            grpc_grace: Grace period for gRPC connection teardown in seconds.
                If `None` - close immediately, cancelling active calls.
        """
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
        """Creates a collection with the given tenant-facing configuration.

        At least one dense or sparse vector is required. Unlike the regular
        client, no storage internals (quantization, WAL, segment number, ...)
        can be configured: the serverless manager decides those.

        Args:
            collection_name: Name of the collection to create
            dense_vectors:
                Dense (embedding) vectors of the collection.
                - If `DenseVectorConfig` - register as the single unnamed
                  default vector, like in regular qdrant.
                - If `dict` - one config per vector name.
            sparse_vectors:
                Sparse vectors of the collection.
                - If `SparseVectorConfig` - register as the single unnamed
                  default vector.
                - If `dict` - one config per vector name.
            payload_indexes:
                Payload indexes to create, keyed by payload field name
                (JSON path, e.g. `user_id` or `meta.tags`). Only the kind of
                filter the field supports is chosen (e.g. `KeywordIndex()`,
                `TextIndex(tokenizer=...)`); index placement is decided by the
                serverless manager. Serverless does not support changing
                payload indexes after creation.
            timeout: Overrides global timeout for this request. Unit is seconds.

        Returns:
            Outcome of the operation, e.g. `"created"`

        Raises:
            grpc.RpcError: with `StatusCode.ALREADY_EXISTS` if the collection already exists
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

        Args:
            collection_name: Name of the collection to delete
            timeout: Overrides global timeout for this request. Unit is seconds.

        Returns:
            `True` if the collection existed and was deleted, `False` if there
            was no such collection
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

        Unlike the regular client, does not raise if the collection is
        missing: check the `exists` field of the result. The returned config
        is the tenant-facing configuration the collection was created with;
        collection internals (segment number, optimizer status, ...) are not
        exposed by serverless.

        Args:
            collection_name: Name of the collection to fetch
            timeout: Overrides global timeout for this request. Unit is seconds.

        Returns:
            `CollectionInfo` with `exists`, the creation-time `config` and an
            eventually consistent `point_count` (absent until stats have been
            written for the collection)
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
        """Checks whether a collection exists.

        Args:
            collection_name: Name of the collection to check
            timeout: Overrides global timeout for this request. Unit is seconds.

        Returns:
            `True` if the collection exists, `False` otherwise
        """
        return self.get_collection(collection_name, timeout=timeout).exists

    def get_collections(
        self, timeout: Optional[int] = None
    ) -> list[serverless_models.CollectionSummary]:
        """Lists the collections of the space.

        Args:
            timeout: Overrides global timeout for this request. Unit is seconds.

        Returns:
            Collection summaries (name and eventually consistent point count),
            ordered by name
        """
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
        recommendation, discovery, context search. Same as in the regular
        client, minus `consistency`, `shard_key_selector` and `lookup_from`,
        which serverless does not support.

        Args:
            collection_name: Collection to search in
            query:
                Query for the chosen search type operation.
                - If `str` - use string as UUID of the existing point as a search query.
                - If `int` - use integer as ID of the existing point as a search query.
                - If `list[float]` - use as a dense vector for nearest search.
                - If `list[list[float]]` - use as a multi-vector for nearest search.
                - If `SparseVector` - use as a sparse vector for nearest search.
                - If `Query` - use as a query for specific search type.
                - If `NumpyArray` - use as a dense vector for nearest search.
                - If `Document` - the server infers the vector from the document text
                  (serverless performs no client-side embedding inference).
                - If `None` - return first `limit` points from the collection.
            using:
                Name of the vectors to use for query.
                If `None` - use default vectors or provided in named vector structures.
            prefetch: Prefetch queries to make a selection of the data to be used with the main query
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
            timeout: Overrides global timeout for this search. Unit is seconds.

        Returns:
            QueryResponse structure containing list of found close points with similarity scores
        """
        # Type resolution only (e.g. a raw list becomes NearestQuery) - no client-side
        # embedding inference: Document/Image inputs go to the server as-is, serverless
        # inference is server-side only.
        query = QdrantFastembedMixin._resolve_query(query)
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
        """Retrieves points by ids.

        Args:
            collection_name: Name of the collection to retrieve from
            ids: List of ids to retrieve
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
            timeout: Overrides global timeout for this request. Unit is seconds.

        Returns:
            List of points. Order of the points is not guaranteed;
            ids that do not exist are silently skipped.
        """
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
        """Scrolls over all points, optionally filtered.

        This method provides a way to iterate over all stored points with some
        optional filtering condition. Scroll does not apply any similarity
        estimations, it will return points sorted by id in ascending order.

        Args:
            collection_name: Name of the collection to scroll
            scroll_filter: If provided - only returns points matching the filtering conditions
            limit: How many points to return
            order_by: Order the records by a payload key. If `None` - order by id.
                Requires a range-capable payload index on the key.
            offset: If provided - skip points with ids less than given `offset`
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
            timeout: Overrides global timeout for this request. Unit is seconds.

        Returns:
            A pair of (List of points) and (optional offset of the next scroll request).
            If the next offset is `None` - there are no more points to scroll.
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
        """Counts points in the collection.

        Counts points matching the filtering conditions, or all points if no
        filter is given.

        Args:
            collection_name: Name of the collection to count points in
            count_filter: Filtering conditions
            exact:
                - If `True` - provide the exact count of points matching the filter.
                - If `False` - provide the approximate count of points matching the filter.
                  Works faster.
            timeout: Overrides global timeout for this request. Unit is seconds.

        Returns:
            Amount of points in the collection matching the filter
        """
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
        """Updates or inserts points into the collection.

        If a point with a given ID already exists - it will be overwritten.
        Same as in the regular client, minus `ordering`, `shard_key_selector`,
        `update_filter` and `update_mode`, which serverless does not support.

        Args:
            collection_name: To which collection to insert
            points: Batch or list of points to insert
            wait: Await for the results to be applied on the server side.
                If `true`, result will be returned only when all changes are applied
            timeout: Overrides global timeout for this request. Unit is seconds.

        Returns:
            Operation Result(UpdateResult)
        """
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
        """Deletes points by ids.

        Unlike the regular client, only deletion by explicit ids is available:
        serverless does not support deletion by filter.

        Args:
            collection_name: Deletes points from this collection
            ids: List of ids of the points to delete
            wait: Await for the results to be applied on the server side.
                If `true`, result will be returned only when all changes are applied
            timeout: Overrides global timeout for this request. Unit is seconds.

        Returns:
            Operation Result(UpdateResult)
        """
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
        """Modifies payload of the given points.

        Only the given payload values are merged into the stored payload;
        other existing keys stay untouched. Unlike the regular client, only
        selection by explicit ids is available: serverless does not support
        payload updates by filter.

        Args:
            collection_name: Name of the collection to set payload in
            payload: Key-value pairs of payload to assign
            ids: List of ids of the points to modify
            key: Path to the nested field in the payload to modify.
                If `None` - modify the root of the payload.
            wait: Await for the results to be applied on the server side.
                If `true`, result will be returned only when all changes are applied
            timeout: Overrides global timeout for this request. Unit is seconds.

        Returns:
            Operation Result(UpdateResult)
        """
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
        """Removes the given payload keys from the given points.

        Unlike the regular client, only selection by explicit ids is
        available: serverless does not support payload updates by filter.

        Args:
            collection_name: Name of the collection to delete payload from
            keys: List of payload keys to remove
            ids: List of ids of the points to modify
            wait: Await for the results to be applied on the server side.
                If `true`, result will be returned only when all changes are applied
            timeout: Overrides global timeout for this request. Unit is seconds.

        Returns:
            Operation Result(UpdateResult)
        """
        return self._remote.delete_payload(
            collection_name=collection_name,
            keys=keys,
            points=list(ids),
            wait=wait,
            timeout=timeout,
        )

    # endregion
