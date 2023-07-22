from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

from qdrant_client import grpc as grpc
from qdrant_client.client_base import QdrantBase
from qdrant_client.conversions import common_types as types
from qdrant_client.http import ApiClient, SyncApis
from qdrant_client.local.qdrant_local import QdrantLocal
from qdrant_client.qdrant_remote import QdrantRemote


class QdrantClient(QdrantBase):
    """Entry point to communicate with Qdrant service via REST or gPRC API.

    It combines interface classes and endpoint implementation.
    Additionally, it provides custom implementations for frequently used methods like initial collection upload.

    All methods in QdrantClient accept both gRPC and REST structures as an input.
    Conversion will be performed automatically.

    .. note::
        This module methods are wrappers around generated client code for gRPC and REST methods.
        If you need lower-level access to generated clients, use following properties:

        - :meth:`QdrantClient.grpc_points`
        - :meth:`QdrantClient.grpc_collections`
        - :meth:`QdrantClient.rest`

    .. note::
        If you need to use async versions of API, please consider using raw implementations of clients directly:

        - For REST: :class:`~qdrant_client.http.api_client.AsyncApis`
        - For gRPC: :class:`~qdrant_client.grpc.PointsStub` and :class:`~qdrant_client.grpc.CollectionsStub`

    Args:
        location:
            If `:memory:` - use in-memory Qdrant instance.
            If `str` - use it as a `url` parameter.
            If `None` - use default values for `host` and `port`.
        url: either host or str of "Optional[scheme], host, Optional[port], Optional[prefix]".
            Default: `None`
        port: Port of the REST API interface. Default: 6333
        grpc_port: Port of the gRPC interface. Default: 6334
        prefer_grpc: If `true` - use gPRC interface whenever possible in custom methods.
        https: If `true` - use HTTPS(SSL) protocol. Default: `None`
        api_key: API key for authentication in Qdrant Cloud. Default: `None`
        prefix:
            If not `None` - add `prefix` to the REST URL path.
            Example: `service/v1` will result in `http://localhost:6333/service/v1/{qdrant-endpoint}` for REST API.
            Default: `None`
        timeout:
            Timeout for REST and gRPC API requests.
            Default: 5.0 seconds for REST and unlimited for gRPC
        host: Host name of Qdrant service. If url and host are None, set to 'localhost'.
            Default: `None`
        path: Persistence path for QdrantLocal. Default: `None`
        **kwargs: Additional arguments passed directly into REST client initialization

    """

    def __init__(
        self,
        location: Optional[str] = None,
        url: Optional[str] = None,
        port: Optional[int] = 6333,
        grpc_port: int = 6334,
        prefer_grpc: bool = False,
        https: Optional[bool] = None,
        api_key: Optional[str] = None,
        prefix: Optional[str] = None,
        timeout: Optional[float] = None,
        host: Optional[str] = None,
        path: Optional[str] = None,
        **kwargs: Any,
    ):
        self._client: QdrantBase

        if location == ":memory:":
            self._client = QdrantLocal(location=location)
        else:
            if path is not None:
                self._client = QdrantLocal(location=path)
            else:
                if location is not None and url is None:
                    url = location
                self._client = QdrantRemote(
                    url=url,
                    port=port,
                    grpc_port=grpc_port,
                    prefer_grpc=prefer_grpc,
                    https=https,
                    api_key=api_key,
                    prefix=prefix,
                    timeout=timeout,
                    host=host,
                    **kwargs,
                )

    @property
    def grpc_collections(self) -> grpc.CollectionsStub:
        """gRPC client for collections methods

        Returns:
            An instance of raw gRPC client, generated from Protobuf
        """
        if isinstance(self._client, QdrantRemote):
            return self._client.grpc_collections

        raise NotImplementedError(f"gRPC client is not supported for {type(self._client)}")

    @property
    def grpc_points(self) -> grpc.PointsStub:
        """gRPC client for points methods

        Returns:
            An instance of raw gRPC client, generated from Protobuf
        """
        if isinstance(self._client, QdrantRemote):
            return self._client.grpc_points

        raise NotImplementedError(f"gRPC client is not supported for {type(self._client)}")

    @property
    def async_grpc_points(self) -> grpc.PointsStub:
        """gRPC client for points methods

        Returns:
            An instance of raw gRPC client, generated from Protobuf
        """
        if isinstance(self._client, QdrantRemote):
            return self._client.async_grpc_points

        raise NotImplementedError(f"gRPC client is not supported for {type(self._client)}")

    @property
    def async_grpc_collections(self) -> grpc.CollectionsStub:
        """gRPC client for collections methods

        Returns:
            An instance of raw gRPC client, generated from Protobuf
        """
        if isinstance(self._client, QdrantRemote):
            return self._client.async_grpc_collections

        raise NotImplementedError(f"gRPC client is not supported for {type(self._client)}")

    @property
    def rest(self) -> SyncApis[ApiClient]:
        """REST Client

        Returns:
            An instance of raw REST API client, generated from OpenAPI schema
        """
        if isinstance(self._client, QdrantRemote):
            return self._client.rest

        raise NotImplementedError(f"REST client is not supported for {type(self._client)}")

    @property
    def http(self) -> SyncApis[ApiClient]:
        """REST Client

        Returns:
            An instance of raw REST API client, generated from OpenAPI schema
        """
        if isinstance(self._client, QdrantRemote):
            return self._client.http

        raise NotImplementedError(f"REST client is not supported for {type(self._client)}")

    def search_batch(
        self,
        collection_name: str,
        requests: Sequence[types.SearchRequest],
        consistency: Optional[types.ReadConsistency] = None,
        **kwargs: Any,
    ) -> List[List[types.ScoredPoint]]:
        """Search for points in multiple collections

        Args:
            collection_name: Name of the collection
            requests: List of search requests
            consistency:
                Read consistency of the search. Defines how many replicas should be queried before returning the result.
                Values:
                - int - number of replicas to query, values should present in all queried replicas
                - 'majority' - query all replicas, but return values present in the majority of replicas
                - 'quorum' - query the majority of replicas, return values present in all of them
                - 'all' - query all replicas, and return values present in all replicas

        Returns:
            List of search responses
        """
        assert len(kwargs) == 0, f"Unknown arguments: {list(kwargs.keys())}"

        return self._client.search_batch(
            collection_name=collection_name,
            requests=requests,
            consistency=consistency,
            **kwargs,
        )

    def search(
        self,
        collection_name: str,
        query_vector: Union[
            types.NumpyArray,
            Sequence[float],
            Tuple[str, List[float]],
            types.NamedVector,
        ],
        query_filter: Optional[types.Filter] = None,
        search_params: Optional[types.SearchParams] = None,
        limit: int = 10,
        offset: int = 0,
        with_payload: Union[bool, Sequence[str], types.PayloadSelector] = True,
        with_vectors: Union[bool, Sequence[str]] = False,
        score_threshold: Optional[float] = None,
        append_payload: bool = True,
        consistency: Optional[types.ReadConsistency] = None,
        **kwargs: Any,
    ) -> List[types.ScoredPoint]:
        """Search for closest vectors in collection taking into account filtering conditions

        Args:
            collection_name: Collection to search in
            query_vector:
                Search for vectors closest to this.
                Can be either a vector itself, or a named vector, or a tuple of vector name and vector itself
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
            append_payload: Same as `with_payload`. Deprecated.
            consistency:
                Read consistency of the search. Defines how many replicas should be queried before returning the result.
                Values:
                - int - number of replicas to query, values should present in all queried replicas
                - 'majority' - query all replicas, but return values present in the majority of replicas
                - 'quorum' - query the majority of replicas, return values present in all of them
                - 'all' - query all replicas, and return values present in all replicas

        Examples:

        `Search with filter`::

            qdrant.search(
                collection_name="test_collection",
                query_vector=[1.0, 0.1, 0.2, 0.7],
                query_filter=Filter(
                    must=[
                        FieldCondition(
                            key='color',
                            range=Match(
                                value="red"
                            )
                        )
                    ]
                )
            )

        Returns:
            List of found close points with similarity scores.
        """
        assert len(kwargs) == 0, f"Unknown arguments: {list(kwargs.keys())}"

        return self._client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            query_filter=query_filter,
            search_params=search_params,
            limit=limit,
            offset=offset,
            with_payload=with_payload,
            with_vectors=with_vectors,
            score_threshold=score_threshold,
            append_payload=append_payload,
            consistency=consistency,
            **kwargs,
        )

    def search_groups(
        self,
        collection_name: str,
        query_vector: Union[
            types.NumpyArray,
            Sequence[float],
            Tuple[str, List[float]],
            types.NamedVector,
        ],
        group_by: str,
        query_filter: Optional[types.Filter] = None,
        search_params: Optional[types.SearchParams] = None,
        limit: int = 10,
        group_size: int = 1,
        with_payload: Union[bool, Sequence[str], types.PayloadSelector] = True,
        with_vectors: Union[bool, Sequence[str]] = False,
        score_threshold: Optional[float] = None,
        with_lookup: Optional[types.WithLookupInterface] = None,
        consistency: Optional[types.ReadConsistency] = None,
        **kwargs: Any,
    ) -> types.GroupsResult:
        """Search for closest vectors grouped by payload field.

        Searches best matches for query vector grouped by the value of payload field.
        Useful to obtain most relevant results for each category, deduplicate results,
        finding the best representation vector for the same entity.

        Args:
            collection_name: Collection to search in
            query_vector:
                Search for vectors closest to this.
                Can be either a vector itself, or a named vector, or a tuple of vector name and vector itself
            group_by: Name of the payload field to group by.
                Field must be of type "keyword" or "integer".
                Nested fields are specified using dot notation, e.g. "nested_field.subfield".
            query_filter:
                - Exclude vectors which doesn't fit given conditions.
                - If `None` - search among all vectors
            search_params: Additional search params
            limit: How many groups return
            group_size: How many results return for each group
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
            score_threshold: Minimal score threshold for the result.
                If defined, less similar results will not be returned.
                Score of the returned result might be higher or smaller than the threshold depending
                on the Distance function used.
                E.g. for cosine similarity only higher scores will be returned.
            with_lookup:
                Look for points in another collection using the group ids.
                If specified, each group will contain a record from the specified collection
                with the same id as the group id. In addition, the parameter allows to specify
                which parts of the record should be returned, like in `with_payload` and `with_vectors` parameters.
            consistency:
                Read consistency of the search. Defines how many replicas should be queried before returning the result.
                Values:
                - int - number of replicas to query, values should present in all queried replicas
                - 'majority' - query all replicas, but return values present in the majority of replicas
                - 'quorum' - query the majority of replicas, return values present in all of them
                - 'all' - query all replicas, and return values present in all replicas

        Returns:
            List of groups with not more than `group_size` hits in each group.
            Each group also contains an id of the group, which is the value of the payload field.
        """
        assert len(kwargs) == 0, f"Unknown arguments: {list(kwargs.keys())}"

        return self._client.search_groups(
            collection_name=collection_name,
            query_vector=query_vector,
            group_by=group_by,
            query_filter=query_filter,
            search_params=search_params,
            limit=limit,
            group_size=group_size,
            with_payload=with_payload,
            with_vectors=with_vectors,
            score_threshold=score_threshold,
            consistency=consistency,
            with_lookup=with_lookup,
            **kwargs,
        )

    def recommend_batch(
        self,
        collection_name: str,
        requests: Sequence[types.RecommendRequest],
        consistency: Optional[types.ReadConsistency] = None,
        **kwargs: Any,
    ) -> List[List[types.ScoredPoint]]:
        """Perform multiple recommend requests in batch mode

        Args:
            collection_name: Name of the collection
            requests: List of recommend requests
            consistency:
                Read consistency of the search. Defines how many replicas should be queried before returning the result.
                Values:
                - int - number of replicas to query, values should present in all queried replicas
                - 'majority' - query all replicas, but return values present in the majority of replicas
                - 'quorum' - query the majority of replicas, return values present in all of them
                - 'all' - query all replicas, and return values present in all replicas

        Returns:
            List of recommend responses
        """
        assert len(kwargs) == 0, f"Unknown arguments: {list(kwargs.keys())}"

        return self._client.recommend_batch(
            collection_name=collection_name,
            requests=requests,
            consistency=consistency,
            **kwargs,
        )

    def recommend(
        self,
        collection_name: str,
        positive: Sequence[types.PointId],
        negative: Optional[Sequence[types.PointId]] = None,
        query_filter: Optional[types.Filter] = None,
        search_params: Optional[types.SearchParams] = None,
        limit: int = 10,
        offset: int = 0,
        with_payload: Union[bool, List[str], types.PayloadSelector] = True,
        with_vectors: Union[bool, List[str]] = False,
        score_threshold: Optional[float] = None,
        using: Optional[str] = None,
        lookup_from: Optional[types.LookupLocation] = None,
        consistency: Optional[types.ReadConsistency] = None,
        **kwargs: Any,
    ) -> List[types.ScoredPoint]:
        """Recommend points: search for similar points based on already stored in Qdrant examples.

        Provide IDs of the stored points, and Qdrant will perform search based on already existing vectors.
        This functionality is especially useful for recommendation over existing collection of points.

        Args:
            collection_name: Collection to search in
            positive:
                List of stored point IDs, which should be used as reference for similarity search.
                If there is only one ID provided - this request is equivalent to the regular search with vector of that point.
                If there are more than one IDs, Qdrant will attempt to search for similar to all of them.
                Recommendation for multiple vectors is experimental. Its behaviour may change in the future.
            negative:
                List of stored point IDs, which should be dissimilar to the search result.
                Negative examples is an experimental functionality. Its behaviour may change in the future.
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
            using:
                Name of the vectors to use for recommendations.
                If `None` - use default vectors.
            lookup_from:
                Defines a location (collection and vector field name), used to lookup vectors for recommendations.
                If `None` - use current collection will be used.
            consistency:
                Read consistency of the search. Defines how many replicas should be queried before returning the result.
                Values:
                - int - number of replicas to query, values should present in all queried replicas
                - 'majority' - query all replicas, but return values present in the majority of replicas
                - 'quorum' - query the majority of replicas, return values present in all of them
                - 'all' - query all replicas, and return values present in all replicas

        Returns:
            List of recommended points with similarity scores.
        """
        assert len(kwargs) == 0, f"Unknown arguments: {list(kwargs.keys())}"

        return self._client.recommend(
            collection_name=collection_name,
            positive=positive,
            negative=negative,
            query_filter=query_filter,
            search_params=search_params,
            limit=limit,
            offset=offset,
            with_payload=with_payload,
            with_vectors=with_vectors,
            score_threshold=score_threshold,
            using=using,
            lookup_from=lookup_from,
            consistency=consistency,
            **kwargs,
        )

    def recommend_groups(
        self,
        collection_name: str,
        group_by: str,
        positive: Sequence[types.PointId],
        negative: Optional[Sequence[types.PointId]] = None,
        query_filter: Optional[types.Filter] = None,
        search_params: Optional[types.SearchParams] = None,
        limit: int = 10,
        group_size: int = 1,
        score_threshold: Optional[float] = None,
        with_payload: Union[bool, Sequence[str], types.PayloadSelector] = True,
        with_vectors: Union[bool, Sequence[str]] = False,
        using: Optional[str] = None,
        lookup_from: Optional[types.LookupLocation] = None,
        with_lookup: Optional[types.WithLookupInterface] = None,
        consistency: Optional[types.ReadConsistency] = None,
        **kwargs: Any,
    ) -> types.GroupsResult:
        """Recommend point groups: search for similar points based on already stored in Qdrant examples
        and groups by payload field.

        Recommend best matches for given stored examples grouped by the value of payload field.
        Useful to obtain most relevant results for each category, deduplicate results,
        finding the best representation vector for the same entity.

        Args:
            collection_name: Collection to search in
            positive:
                List of stored point IDs, which should be used as reference for similarity search.
                If there is only one ID provided - this request is equivalent to the regular search with vector of that point.
                If there are more than one IDs, Qdrant will attempt to search for similar to all of them.
                Recommendation for multiple vectors is experimental. Its behaviour may change in the future.
            negative:
                List of stored point IDs, which should be dissimilar to the search result.
                Negative examples is an experimental functionality. Its behaviour may change in the future.
            group_by: Name of the payload field to group by.
                Field must be of type "keyword" or "integer".
                Nested fields are specified using dot notation, e.g. "nested_field.subfield".
            query_filter:
                - Exclude vectors which doesn't fit given conditions.
                - If `None` - search among all vectors
            search_params: Additional search params
            limit: How many groups return
            group_size: How many results return for each group
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
            using:
                Name of the vectors to use for recommendations.
                If `None` - use default vectors.
            lookup_from:
                Defines a location (collection and vector field name), used to lookup vectors for recommendations.
                If `None` - use current collection will be used.
            with_lookup:
                Look for points in another collection using the group ids.
                If specified, each group will contain a record from the specified collection
                with the same id as the group id. In addition, the parameter allows to specify
                which parts of the record should be returned, like in `with_payload` and `with_vectors` parameters.
            consistency:
                Read consistency of the search. Defines how many replicas should be queried before returning the result.
                Values:
                - int - number of replicas to query, values should present in all queried replicas
                - 'majority' - query all replicas, but return values present in the majority of replicas
                - 'quorum' - query the majority of replicas, return values present in all of them
                - 'all' - query all replicas, and return values present in all replicas

        Returns:
            List of groups with not more than `group_size` hits in each group.
            Each group also contains an id of the group, which is the value of the payload field.

        """
        assert len(kwargs) == 0, f"Unknown arguments: {list(kwargs.keys())}"

        return self._client.recommend_groups(
            collection_name=collection_name,
            group_by=group_by,
            positive=positive,
            negative=negative,
            query_filter=query_filter,
            search_params=search_params,
            limit=limit,
            group_size=group_size,
            score_threshold=score_threshold,
            with_payload=with_payload,
            with_vectors=with_vectors,
            using=using,
            lookup_from=lookup_from,
            consistency=consistency,
            with_lookup=with_lookup,
            **kwargs,
        )

    def scroll(
        self,
        collection_name: str,
        scroll_filter: Optional[types.Filter] = None,
        limit: int = 10,
        offset: Optional[types.PointId] = None,
        with_payload: Union[bool, Sequence[str], types.PayloadSelector] = True,
        with_vectors: Union[bool, Sequence[str]] = False,
        consistency: Optional[types.ReadConsistency] = None,
        **kwargs: Any,
    ) -> Tuple[List[types.Record], Optional[types.PointId]]:
        """Scroll over all (matching) points in the collection.

        This method provides a way to iterate over all stored points with some optional filtering condition.
        Scroll does not apply any similarity estimations, it will return points sorted by id in ascending order.

        Args:
            collection_name: Name of the collection
            scroll_filter: If provided - only returns points matching filtering conditions
            limit: How many points to return
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
            consistency:
                Read consistency of the search. Defines how many replicas should be queried before returning the result.
                Values:
                - int - number of replicas to query, values should present in all queried replicas
                - 'majority' - query all replicas, but return values present in the majority of replicas
                - 'quorum' - query the majority of replicas, return values present in all of them
                - 'all' - query all replicas, and return values present in all replicas

        Returns:
            A pair of (List of points) and (optional offset for the next scroll request).
            If next page offset is `None` - there is no more points in the collection to scroll.
        """
        assert len(kwargs) == 0, f"Unknown arguments: {list(kwargs.keys())}"

        return self._client.scroll(
            collection_name=collection_name,
            scroll_filter=scroll_filter,
            limit=limit,
            offset=offset,
            with_payload=with_payload,
            with_vectors=with_vectors,
            consistency=consistency,
            **kwargs,
        )

    def count(
        self,
        collection_name: str,
        count_filter: Optional[types.Filter] = None,
        exact: bool = True,
        **kwargs: Any,
    ) -> types.CountResult:
        """Count points in the collection.

        Count points in the collection matching the given filter.

        Args:
            collection_name: name of the collection to count points in
            count_filter: filtering conditions
            exact:
                If `True` - provide the exact count of points matching the filter.
                If `False` - provide the approximate count of points matching the filter. Works faster.

        Returns:
            Amount of points in the collection matching the filter.
        """
        assert len(kwargs) == 0, f"Unknown arguments: {list(kwargs.keys())}"

        return self._client.count(
            collection_name=collection_name,
            count_filter=count_filter,
            exact=exact,
            **kwargs,
        )

    def upsert(
        self,
        collection_name: str,
        points: types.Points,
        wait: bool = True,
        ordering: Optional[types.WriteOrdering] = None,
        **kwargs: Any,
    ) -> types.UpdateResult:
        """Update or insert a new point into the collection.

        If point with given ID already exists - it will be overwritten.

        Args:
            collection_name: To which collection to insert
            wait: Await for the results to be processed.

                - If `true`, result will be returned only when all changes are applied
                - If `false`, result will be returned immediately after the confirmation of receiving.
            points: Batch or list of points to insert
            ordering:
                Define strategy for ordering of the points. Possible values:
                - 'weak' - write operations may be reordered, works faster, default
                - 'medium' - write operations go through dynamically selected leader,
                    may be inconsistent for a short period of time in case of leader change
                - 'strong' - Write operations go through the permanent leader,
                    consistent, but may be unavailable if leader is down

        Returns:
            Operation result
        """
        assert len(kwargs) == 0, f"Unknown arguments: {list(kwargs.keys())}"

        return self._client.upsert(
            collection_name=collection_name,
            points=points,
            wait=wait,
            ordering=ordering,
            **kwargs,
        )

    def update_vectors(
        self,
        collection_name: str,
        vectors: Sequence[types.PointVectors],
        wait: bool = True,
        ordering: Optional[types.WriteOrdering] = None,
        **kwargs: Any,
    ) -> types.UpdateResult:
        """Update specified vectors in the collection. Keeps payload and unspecified vectors unchanged.

        Args:
            collection_name: Name of the collection to update vectors in
            vectors: List of (id, vector) pairs to update. Vector might be a list of numbers or a dict of named vectors.
                Example
                - `PointVectors(id=1, vector=[1, 2, 3])`
                - `PointVectors(id=2, vector={'vector_1': [1, 2, 3], 'vector_2': [4, 5, 6]})`
            wait: Await for the results to be processed.
            ordering: Define strategy for ordering of the points. Possible values:
                - 'weak' - write operations may be reordered, works faster, default
                - 'medium' - write operations go through dynamically selected leader,
                    may be inconsistent for a short period of time in case of leader change
                - 'strong' - Write operations go through the permanent leader,
                    consistent, but may be unavailable if leader is down

        Returns:
            Operation result
        """
        assert len(kwargs) == 0, f"Unknown arguments: {list(kwargs.keys())}"

        return self._client.update_vectors(
            collection_name=collection_name,
            vectors=vectors,
            wait=wait,
            ordering=ordering,
        )

    def delete_vectors(
        self,
        collection_name: str,
        vectors: Sequence[str],
        points: types.PointsSelector,
        wait: bool = True,
        ordering: Optional[types.WriteOrdering] = None,
        **kwargs: Any,
    ) -> types.UpdateResult:
        """Delete specified vector from the collection. Does not affect payload.

        Args:
            collection_name: Name of the collection to delete vector from
            vectors:
                List of names of the vectors to delete.
                Use `""` to delete the default vector.
                At least one vector should be specified.
            points: Selects points based on list of IDs or filter
                 Examples
                    - `points=[1, 2, 3, "cd3b53f0-11a7-449f-bc50-d06310e7ed90"]`
                    - `points=Filter(must=[FieldCondition(key='rand_number', range=Range(gte=0.7))])`
            wait: Await for the results to be processed.
            ordering: Define strategy for ordering of the points. Possible values:
                - 'weak' - write operations may be reordered, works faster, default
                - 'medium' - write operations go through dynamically selected leader,
                    may be inconsistent for a short period of time in case of leader change
                - 'strong' - Write operations go through the permanent leader,
                    consistent, but may be unavailable if leader is down

        Returns:
            Operation result
        """
        assert len(kwargs) == 0, f"Unknown arguments: {list(kwargs.keys())}"

        return self._client.delete_vectors(
            collection_name=collection_name,
            vectors=vectors,
            points=points,
            wait=wait,
            ordering=ordering,
        )

    def retrieve(
        self,
        collection_name: str,
        ids: Sequence[types.PointId],
        with_payload: Union[bool, Sequence[str], types.PayloadSelector] = True,
        with_vectors: Union[bool, Sequence[str]] = False,
        consistency: Optional[types.ReadConsistency] = None,
        **kwargs: Any,
    ) -> List[types.Record]:
        """Retrieve stored points by IDs

        Args:
            collection_name: Name of the collection to lookup in
            ids: list of IDs to lookup
            with_payload:
                - Specify which stored payload should be attached to the result.
                - If `True` - attach all payload
                - If `False` - do not attach any payload
                - If List of string - include only specified fields
                - If `PayloadSelector` - use explicit rules
            with_vectors:
                - If `True` - Attach stored vector to the search result.
                - If `False` - Do not attach vector.
                - If List of string - Attach only specified vectors.
                - Default: `False`
            consistency:
                Read consistency of the search. Defines how many replicas should be queried before returning the result.
                Values:
                - int - number of replicas to query, values should present in all queried replicas
                - 'majority' - query all replicas, but return values present in the majority of replicas
                - 'quorum' - query the majority of replicas, return values present in all of them
                - 'all' - query all replicas, and return values present in all replicas

        Returns:
            List of points
        """
        assert len(kwargs) == 0, f"Unknown arguments: {list(kwargs.keys())}"

        return self._client.retrieve(
            collection_name=collection_name,
            ids=ids,
            with_payload=with_payload,
            with_vectors=with_vectors,
            consistency=consistency,
            **kwargs,
        )

    def delete(
        self,
        collection_name: str,
        points_selector: types.PointsSelector,
        wait: bool = True,
        ordering: Optional[types.WriteOrdering] = None,
        **kwargs: Any,
    ) -> types.UpdateResult:
        """Deletes selected points from collection

        Args:
            collection_name: Name of the collection
            wait: Await for the results to be processed.

                - If `true`, result will be returned only when all changes are applied
                - If `false`, result will be returned immediately after the confirmation of receiving.
            points_selector: Selects points based on list of IDs or filter
                Examples
                    - `points=[1, 2, 3, "cd3b53f0-11a7-449f-bc50-d06310e7ed90"]`
                    - `points=Filter(must=[FieldCondition(key='rand_number', range=Range(gte=0.7))])`
            ordering: Define strategy for ordering of the points. Possible values:

                - 'weak' - write operations may be reordered, works faster, default
                - 'medium' - write operations go through dynamically selected leader,
                    may be inconsistent for a short period of time in case of leader change
                - 'strong' - Write operations go through the permanent leader,
                    consistent, but may be unavailable if leader is down

        Returns:
            Operation result
        """
        assert len(kwargs) == 0, f"Unknown arguments: {list(kwargs.keys())}"

        return self._client.delete(
            collection_name=collection_name,
            points_selector=points_selector,
            wait=wait,
            ordering=ordering,
            **kwargs,
        )

    def set_payload(
        self,
        collection_name: str,
        payload: types.Payload,
        points: types.PointsSelector,
        wait: bool = True,
        ordering: Optional[types.WriteOrdering] = None,
        **kwargs: Any,
    ) -> types.UpdateResult:
        """Modifies payload of the specified points

        Examples:

        `Set payload`::

            # Assign payload value with key `"key"` to points 1, 2, 3.
            # If payload value with specified key already exists - it will be overwritten
            qdrant_client.set_payload(
                collection_name="test_collection",
                wait=True,
                payload={
                    "key": "value"
                },
                points=[1,2,3]
            )

        Args:
            collection_name: Name of the collection
            wait: Await for the results to be processed.

                - If `true`, result will be returned only when all changes are applied
                - If `false`, result will be returned immediately after the confirmation of receiving.
            payload: Key-value pairs of payload to assign
            points: List of affected points, filter or points selector.
             Example
                - `points=[1, 2, 3, "cd3b53f0-11a7-449f-bc50-d06310e7ed90"]`
                - `points=Filter(must=[FieldCondition(key='rand_number', range=Range(gte=0.7))])`
            ordering:
                Define strategy for ordering of the points. Possible values:
                - 'weak' - write operations may be reordered, works faster, default
                - 'medium' - write operations go through dynamically selected leader,
                    may be inconsistent for a short period of time in case of leader change
                - 'strong' - Write operations go through the permanent leader,
                    consistent, but may be unavailable if leader is down

        Returns:
            Operation result
        """
        assert len(kwargs) == 0, f"Unknown arguments: {list(kwargs.keys())}"

        return self._client.set_payload(
            collection_name=collection_name,
            payload=payload,
            points=points,
            wait=wait,
            ordering=ordering,
            **kwargs,
        )

    def overwrite_payload(
        self,
        collection_name: str,
        payload: types.Payload,
        points: types.PointsSelector,
        wait: bool = True,
        ordering: Optional[types.WriteOrdering] = None,
        **kwargs: Any,
    ) -> types.UpdateResult:
        """Overwrites payload of the specified points
        After this operation is applied, only the specified payload will be present in the point.
        The existing payload, even if the key is not specified in the payload, will be deleted.

        Examples:

        `Set payload`::

            # Overwrite payload value with key `"key"` to points 1, 2, 3.
            # If any other valid payload value exists - it will be deleted
            qdrant_client.overwrite_payload(
                collection_name="test_collection",
                wait=True,
                payload={
                    "key": "value"
                },
                points=[1,2,3]
            )

        Args:
            collection_name: Name of the collection
            wait: Await for the results to be processed.

                - If `true`, result will be returned only when all changes are applied
                - If `false`, result will be returned immediately after the confirmation of receiving.
            payload: Key-value pairs of payload to assign
            points: List of affected points, filter or points selector.
             Example
                - `points=[1, 2, 3, "cd3b53f0-11a7-449f-bc50-d06310e7ed90"]`
                - `points=Filter(must=[FieldCondition(key='rand_number', range=Range(gte=0.7))])`
            ordering:
                Define strategy for ordering of the points. Possible values:
                - 'weak' - write operations may be reordered, works faster, default
                - 'medium' - write operations go through dynamically selected leader,
                    may be inconsistent for a short period of time in case of leader change
                - 'strong' - Write operations go through the permanent leader,
                    consistent, but may be unavailable if leader is down

        Returns:
            Operation result
        """
        assert len(kwargs) == 0, f"Unknown arguments: {list(kwargs.keys())}"

        return self._client.overwrite_payload(
            collection_name=collection_name,
            payload=payload,
            points=points,
            wait=wait,
            ordering=ordering,
            **kwargs,
        )

    def delete_payload(
        self,
        collection_name: str,
        keys: Sequence[str],
        points: types.PointsSelector,
        wait: bool = True,
        ordering: Optional[types.WriteOrdering] = None,
        **kwargs: Any,
    ) -> types.UpdateResult:
        """Remove values from point's payload

        Args:
            collection_name: Name of the collection
            wait: Await for the results to be processed.

                - If `true`, result will be returned only when all changes are applied
                - If `false`, result will be returned immediately after the confirmation of receiving.
            keys: List of payload keys to remove
            points: List of affected points, filter or points selector.
                Example
                   - `points=[1, 2, 3, "cd3b53f0-11a7-449f-bc50-d06310e7ed90"]`
                   - `points=Filter(must=[FieldCondition(key='rand_number', range=Range(gte=0.7))])`
            ordering:
                Define strategy for ordering of the points. Possible values:
                - 'weak' - write operations may be reordered, works faster, default
                - 'medium' - write operations go through dynamically selected leader,
                    may be inconsistent for a short period of time in case of leader change
                - 'strong' - Write operations go through the permanent leader,
                    consistent, but may be unavailable if leader is down

        Returns:
            Operation result
        """
        assert len(kwargs) == 0, f"Unknown arguments: {list(kwargs.keys())}"

        return self._client.delete_payload(
            collection_name=collection_name,
            keys=keys,
            points=points,
            wait=wait,
            ordering=ordering,
            **kwargs,
        )

    def clear_payload(
        self,
        collection_name: str,
        points_selector: types.PointsSelector,
        wait: bool = True,
        ordering: Optional[types.WriteOrdering] = None,
        **kwargs: Any,
    ) -> types.UpdateResult:
        """Delete all payload for selected points

        Args:
            collection_name: Name of the collection
            wait: Await for the results to be processed.

                - If `true`, result will be returned only when all changes are applied
                - If `false`, result will be returned immediately after the confirmation of receiving.
            points_selector: List of affected points, filter or points selector.
                Example
                   - `points=[1, 2, 3, "cd3b53f0-11a7-449f-bc50-d06310e7ed90"]`
                   - `points=Filter(must=[FieldCondition(key='rand_number', range=Range(gte=0.7))])`
            ordering:
                Define strategy for ordering of the points. Possible values:
                - 'weak' - write operations may be reordered, works faster, default
                - 'medium' - write operations go through dynamically selected leader,
                    may be inconsistent for a short period of time in case of leader change
                - 'strong' - Write operations go through the permanent leader,
                    consistent, but may be unavailable if leader is down

        Returns:
            Operation result
        """
        assert len(kwargs) == 0, f"Unknown arguments: {list(kwargs.keys())}"

        return self._client.clear_payload(
            collection_name=collection_name,
            points_selector=points_selector,
            wait=wait,
            ordering=ordering,
            **kwargs,
        )

    def update_collection_aliases(
        self,
        change_aliases_operations: Sequence[types.AliasOperations],
        timeout: Optional[int] = None,
        **kwargs: Any,
    ) -> bool:
        """Operation for performing changes of collection aliases.

        Alias changes are atomic, meaning that no collection modifications can happen between alias operations.

        Args:
            change_aliases_operations: List of operations to perform
            timeout:
                Wait for operation commit timeout in seconds.
                If timeout is reached - request will return with service error.

        Returns:
            Operation result
        """
        assert len(kwargs) == 0, f"Unknown arguments: {list(kwargs.keys())}"

        return self._client.update_collection_aliases(
            change_aliases_operations=change_aliases_operations,
            timeout=timeout,
            **kwargs,
        )

    def get_collection_aliases(
        self, collection_name: str, **kwargs: Any
    ) -> types.CollectionsAliasesResponse:
        """Get collection aliases

        Args:
            collection_name: Name of the collection

        Returns:
            Collection aliases
        """
        assert len(kwargs) == 0, f"Unknown arguments: {list(kwargs.keys())}"

        return self._client.get_collection_aliases(collection_name=collection_name, **kwargs)

    def get_aliases(self, **kwargs: Any) -> types.CollectionsAliasesResponse:
        """Get all aliases

        Returns:
            All aliases of all collections
        """
        assert len(kwargs) == 0, f"Unknown arguments: {list(kwargs.keys())}"

        return self._client.get_aliases(**kwargs)

    def get_collections(self, **kwargs: Any) -> types.CollectionsResponse:
        """Get list name of all existing collections

        Returns:
            List of the collections
        """
        assert len(kwargs) == 0, f"Unknown arguments: {list(kwargs.keys())}"

        return self._client.get_collections(**kwargs)

    def get_collection(self, collection_name: str, **kwargs: Any) -> types.CollectionInfo:
        """Get detailed information about specified existing collection

        Args:
            collection_name: Name of the collection

        Returns:
            Detailed information about the collection
        """
        assert len(kwargs) == 0, f"Unknown arguments: {list(kwargs.keys())}"

        return self._client.get_collection(collection_name=collection_name, **kwargs)

    def update_collection(
        self,
        collection_name: str,
        optimizers_config: Optional[types.OptimizersConfigDiff] = None,
        collection_params: Optional[types.CollectionParamsDiff] = None,
        timeout: Optional[int] = None,
        **kwargs: Any,
    ) -> bool:
        """Update parameters of the collection

        Args:
            collection_name: Name of the collection
            optimizers_config: Override for optimizer configuration
            collection_params: Override for collection parameters
            timeout:
                Wait for operation commit timeout in seconds.
                If timeout is reached - request will return with service error.
        Returns:
            Operation result
        """
        if "optimizer_config" in kwargs and optimizers_config is not None:
            raise ValueError(
                "Only one of optimizer_config and optimizers_config should be specified"
            )

        if "optimizer_config" in kwargs:
            optimizers_config = kwargs.pop("optimizer_config")

        assert len(kwargs) == 0, f"Unknown arguments: {list(kwargs.keys())}"

        return self._client.update_collection(
            collection_name=collection_name,
            optimizers_config=optimizers_config,
            collection_params=collection_params,
            timeout=timeout,
            **kwargs,
        )

    def delete_collection(
        self, collection_name: str, timeout: Optional[int] = None, **kwargs: Any
    ) -> bool:
        """Removes collection and all it's data

        Args:
            collection_name: Name of the collection to delete
            timeout:
                Wait for operation commit timeout in seconds.
                If timeout is reached - request will return with service error.

        Returns:
            Operation result
        """
        assert len(kwargs) == 0, f"Unknown arguments: {list(kwargs.keys())}"

        return self._client.delete_collection(
            collection_name=collection_name, timeout=timeout, **kwargs
        )

    def create_collection(
        self,
        collection_name: str,
        vectors_config: Union[types.VectorParams, Mapping[str, types.VectorParams]],
        shard_number: Optional[int] = None,
        replication_factor: Optional[int] = None,
        write_consistency_factor: Optional[int] = None,
        on_disk_payload: Optional[bool] = None,
        hnsw_config: Optional[types.HnswConfigDiff] = None,
        optimizers_config: Optional[types.OptimizersConfigDiff] = None,
        wal_config: Optional[types.WalConfigDiff] = None,
        quantization_config: Optional[types.QuantizationConfig] = None,
        init_from: Optional[types.InitFrom] = None,
        timeout: Optional[int] = None,
        **kwargs: Any,
    ) -> bool:
        """Create empty collection with given parameters

        Args:
            collection_name: Name of the collection to recreate
            vectors_config:
                Configuration of the vector storage. Vector params contains size and distance for the vector storage.
                If dict is passed, service will create a vector storage for each key in the dict.
                If single VectorParams is passed, service will create a single anonymous vector storage.
            shard_number: Number of shards in collection. Default is 1, minimum is 1.
            replication_factor:
                Replication factor for collection. Default is 1, minimum is 1.
                Defines how many copies of each shard will be created.
                Have effect only in distributed mode.
            write_consistency_factor:
                Write consistency factor for collection. Default is 1, minimum is 1.
                Defines how many replicas should apply the operation for us to consider it successful.
                Increasing this number will make the collection more resilient to inconsistencies, but will
                also make it fail if not enough replicas are available.
                Does not have any performance impact.
                Have effect only in distributed mode.
            on_disk_payload:
                If true - point`s payload will not be stored in memory.
                It will be read from the disk every time it is requested.
                This setting saves RAM by (slightly) increasing the response time.
                Note: those payload values that are involved in filtering and are indexed - remain in RAM.
            hnsw_config: Params for HNSW index
            optimizers_config: Params for optimizer
            wal_config: Params for Write-Ahead-Log
            quantization_config: Params for quantization, if None - quantization will be disabled
            init_from: Use data stored in another collection to initialize this collection
            timeout:
                Wait for operation commit timeout in seconds.
                If timeout is reached - request will return with service error.

        Returns:
            Operation result
        """
        assert len(kwargs) == 0, f"Unknown arguments: {list(kwargs.keys())}"

        return self._client.create_collection(
            collection_name=collection_name,
            vectors_config=vectors_config,
            shard_number=shard_number,
            replication_factor=replication_factor,
            write_consistency_factor=write_consistency_factor,
            on_disk_payload=on_disk_payload,
            hnsw_config=hnsw_config,
            optimizers_config=optimizers_config,
            wal_config=wal_config,
            quantization_config=quantization_config,
            init_from=init_from,
            timeout=timeout,
            **kwargs,
        )

    def recreate_collection(
        self,
        collection_name: str,
        vectors_config: Union[types.VectorParams, Mapping[str, types.VectorParams]],
        shard_number: Optional[int] = None,
        replication_factor: Optional[int] = None,
        write_consistency_factor: Optional[int] = None,
        on_disk_payload: Optional[bool] = None,
        hnsw_config: Optional[types.HnswConfigDiff] = None,
        optimizers_config: Optional[types.OptimizersConfigDiff] = None,
        wal_config: Optional[types.WalConfigDiff] = None,
        quantization_config: Optional[types.QuantizationConfig] = None,
        init_from: Optional[types.InitFrom] = None,
        timeout: Optional[int] = None,
        **kwargs: Any,
    ) -> bool:
        """Delete and create empty collection with given parameters

        Args:
            collection_name: Name of the collection to recreate
            vectors_config:
                Configuration of the vector storage. Vector params contains size and distance for the vector storage.
                If dict is passed, service will create a vector storage for each key in the dict.
                If single VectorParams is passed, service will create a single anonymous vector storage.
            shard_number: Number of shards in collection. Default is 1, minimum is 1.
            replication_factor:
                Replication factor for collection. Default is 1, minimum is 1.
                Defines how many copies of each shard will be created.
                Have effect only in distributed mode.
            write_consistency_factor:
                Write consistency factor for collection. Default is 1, minimum is 1.
                Defines how many replicas should apply the operation for us to consider it successful.
                Increasing this number will make the collection more resilient to inconsistencies, but will
                also make it fail if not enough replicas are available.
                Does not have any performance impact.
                Have effect only in distributed mode.
            on_disk_payload:
                If true - point`s payload will not be stored in memory.
                It will be read from the disk every time it is requested.
                This setting saves RAM by (slightly) increasing the response time.
                Note: those payload values that are involved in filtering and are indexed - remain in RAM.
            hnsw_config: Params for HNSW index
            optimizers_config: Params for optimizer
            wal_config: Params for Write-Ahead-Log
            quantization_config: Params for quantization, if None - quantization will be disabled
            init_from: Use data stored in another collection to initialize this collection
            timeout:
                Wait for operation commit timeout in seconds.
                If timeout is reached - request will return with service error.

        Returns:
            Operation result
        """
        assert len(kwargs) == 0, f"Unknown arguments: {list(kwargs.keys())}"

        return self._client.recreate_collection(
            collection_name=collection_name,
            vectors_config=vectors_config,
            shard_number=shard_number,
            replication_factor=replication_factor,
            write_consistency_factor=write_consistency_factor,
            on_disk_payload=on_disk_payload,
            hnsw_config=hnsw_config,
            optimizers_config=optimizers_config,
            wal_config=wal_config,
            quantization_config=quantization_config,
            init_from=init_from,
            timeout=timeout,
            **kwargs,
        )

    def upload_records(
        self,
        collection_name: str,
        records: Iterable[types.Record],
        batch_size: int = 64,
        parallel: int = 1,
        method: Optional[str] = None,
        max_retries: int = 3,
        **kwargs: Any,
    ) -> None:
        """Upload records to the collection

        Similar to `upload_collection` method, but operates with records, rather than vector and payload individually.

        Args:
            collection_name:  Name of the collection to upload to
            records: Iterator over records to upload
            batch_size: How many vectors upload per-request, Default: 64
            parallel: Number of parallel processes of upload
            method: Start method for parallel processes, Default: forkserver
            max_retries: maximum number of retries in case of a failure
                during the upload of a batch

        """
        assert len(kwargs) == 0, f"Unknown arguments: {list(kwargs.keys())}"

        return self._client.upload_records(
            collection_name=collection_name,
            records=records,
            batch_size=batch_size,
            parallel=parallel,
            method=method,
            max_retries=max_retries,
            **kwargs,
        )

    def upload_collection(
        self,
        collection_name: str,
        vectors: Union[
            Dict[str, types.NumpyArray], types.NumpyArray, Iterable[types.VectorStruct]
        ],
        payload: Optional[Iterable[Dict[Any, Any]]] = None,
        ids: Optional[Iterable[types.PointId]] = None,
        batch_size: int = 64,
        parallel: int = 1,
        method: Optional[str] = None,
        max_retries: int = 3,
        **kwargs: Any,
    ) -> None:
        """Upload vectors and payload to the collection.
        This method will perform automatic batching of the data.
        If you need to perform a single update, use `upsert` method.
        Note: use `upload_records` method if you want to upload multiple vectors with single payload.

        Args:
            collection_name:  Name of the collection to upload to
            vectors: np.ndarray or an iterable over vectors to upload. Might be mmaped
            payload: Iterable of vectors payload, Optional, Default: None
            ids: Iterable of custom vectors ids, Optional, Default: None
            batch_size: How many vectors upload per-request, Default: 64
            parallel: Number of parallel processes of upload
            method: Start method for parallel processes, Default: forkserver
            max_retries: maximum number of retries in case of a failure
                during the upload of a batch
        """
        assert len(kwargs) == 0, f"Unknown arguments: {list(kwargs.keys())}"

        return self._client.upload_collection(
            collection_name=collection_name,
            vectors=vectors,
            payload=payload,
            ids=ids,
            batch_size=batch_size,
            parallel=parallel,
            method=method,
            max_retries=max_retries,
            **kwargs,
        )

    def create_payload_index(
        self,
        collection_name: str,
        field_name: str,
        field_schema: Optional[types.PayloadSchemaType] = None,
        field_type: Optional[types.PayloadSchemaType] = None,
        wait: bool = True,
        ordering: Optional[types.WriteOrdering] = None,
        **kwargs: Any,
    ) -> types.UpdateResult:
        """Creates index for a given payload field.
        Indexed fields allow to perform filtered search operations faster.

        Args:
            collection_name: Name of the collection
            field_name: Name of the payload field
            field_schema: Type of data to index
            field_type: Same as field_schema, but deprecated
            wait: Await for the results to be processed.

                - If `true`, result will be returned only when all changes are applied
                - If `false`, result will be returned immediately after the confirmation of receiving.
            ordering:
                Define strategy for ordering of the points. Possible values:
                - 'weak' - write operations may be reordered, works faster, default
                - 'medium' - write operations go through dynamically selected leader,
                    may be inconsistent for a short period of time in case of leader change
                - 'strong' - Write operations go through the permanent leader,
                    consistent, but may be unavailable if leader is down

        Returns:
            Operation Result
        """
        assert len(kwargs) == 0, f"Unknown arguments: {list(kwargs.keys())}"

        return self._client.create_payload_index(
            collection_name=collection_name,
            field_name=field_name,
            field_schema=field_schema,
            field_type=field_type,
            wait=wait,
            ordering=ordering,
            **kwargs,
        )

    def delete_payload_index(
        self,
        collection_name: str,
        field_name: str,
        wait: bool = True,
        ordering: Optional[types.WriteOrdering] = None,
        **kwargs: Any,
    ) -> types.UpdateResult:
        """Removes index for a given payload field.

        Args:
            collection_name: Name of the collection
            field_name: Name of the payload field
            wait: Await for the results to be processed.

                - If `true`, result will be returned only when all changes are applied
                - If `false`, result will be returned immediately after the confirmation of receiving.
            ordering:
                Define strategy for ordering of the points. Possible values:
                - 'weak' - write operations may be reordered, works faster, default
                - 'medium' - write operations go through dynamically selected leader,
                    may be inconsistent for a short period of time in case of leader change
                - 'strong' - Write operations go through the permanent leader,
                    consistent, but may be unavailable if leader is down

        Returns:
            Operation Result
        """
        assert len(kwargs) == 0, f"Unknown arguments: {list(kwargs.keys())}"

        return self._client.delete_payload_index(
            collection_name=collection_name,
            field_name=field_name,
            wait=wait,
            ordering=ordering,
            **kwargs,
        )

    def list_snapshots(
        self, collection_name: str, **kwargs: Any
    ) -> List[types.SnapshotDescription]:
        """List all snapshots for a given collection.

        Args:
            collection_name: Name of the collection

        Returns:
            List of snapshots
        """
        assert len(kwargs) == 0, f"Unknown arguments: {list(kwargs.keys())}"

        return self._client.list_snapshots(collection_name=collection_name, **kwargs)

    def create_snapshot(
        self, collection_name: str, **kwargs: Any
    ) -> Optional[types.SnapshotDescription]:
        """Create snapshot for a given collection.

        Args:
            collection_name: Name of the collection

        Returns:
            Snapshot description
        """
        assert len(kwargs) == 0, f"Unknown arguments: {list(kwargs.keys())}"

        return self._client.create_snapshot(collection_name=collection_name, **kwargs)

    def delete_snapshot(self, collection_name: str, snapshot_name: str, **kwargs: Any) -> bool:
        """Delete snapshot for a given collection.

        Args:
            collection_name: Name of the collection
            snapshot_name: Snapshot id

        Returns:
            True if snapshot was deleted
        """
        assert len(kwargs) == 0, f"Unknown arguments: {list(kwargs.keys())}"

        return self._client.delete_snapshot(
            collection_name=collection_name, snapshot_name=snapshot_name, **kwargs
        )

    def list_full_snapshots(self, **kwargs: Any) -> List[types.SnapshotDescription]:
        """List all snapshots for a whole storage

        Returns:
            List of snapshots
        """
        assert len(kwargs) == 0, f"Unknown arguments: {list(kwargs.keys())}"

        return self._client.list_full_snapshots(**kwargs)

    def create_full_snapshot(self, **kwargs: Any) -> types.SnapshotDescription:
        """Create snapshot for a whole storage.

        Returns:
            Snapshot description
        """
        assert len(kwargs) == 0, f"Unknown arguments: {list(kwargs.keys())}"

        return self._client.create_full_snapshot(**kwargs)

    def delete_full_snapshot(self, snapshot_name: str, **kwargs: Any) -> bool:
        """Delete snapshot for a whole storage.

        Args:
            snapshot_name: Snapshot name

        Returns:
            True if snapshot was deleted
        """
        assert len(kwargs) == 0, f"Unknown arguments: {list(kwargs.keys())}"

        return self._client.delete_full_snapshot(snapshot_name=snapshot_name, **kwargs)

    def recover_snapshot(
        self,
        collection_name: str,
        location: str,
        priority: Optional[types.SnapshotPriority] = None,
        **kwargs: Any,
    ) -> bool:
        """Recover collection from snapshot.

        Args:
            collection_name: Name of the collection
            location:
                URL of the snapshot.
                Example:
                    - URL `http://localhost:8080/collections/my_collection/snapshots/my_snapshot`
                    - Local path `file:///qdrant/snapshots/test_collection-2022-08-04-10-49-10.snapshot`
            priority:
                Defines source of truth for snapshot recovery
                    - `snapshot` means - prefer snapshot data over the current state
                    - `replica` means - prefer existing data over the snapshot
                Default: `replica`

        """
        assert len(kwargs) == 0, f"Unknown arguments: {list(kwargs.keys())}"

        return self._client.recover_snapshot(
            collection_name=collection_name,
            location=location,
            priority=priority,
            **kwargs,
        )

    def lock_storage(self, reason: str, **kwargs: Any) -> types.LocksOption:
        """Lock storage for writing."""
        assert len(kwargs) == 0, f"Unknown arguments: {list(kwargs.keys())}"

        return self._client.lock_storage(reason=reason, **kwargs)

    def unlock_storage(self, **kwargs: Any) -> types.LocksOption:
        """Unlock storage for writing."""
        assert len(kwargs) == 0, f"Unknown arguments: {list(kwargs.keys())}"

        return self._client.unlock_storage(**kwargs)

    def get_locks(self, **kwargs: Any) -> types.LocksOption:
        """Get current locks state."""
        assert len(kwargs) == 0, f"Unknown arguments: {list(kwargs.keys())}"

        return self._client.get_locks(**kwargs)
