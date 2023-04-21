import logging
import warnings
from multiprocessing import get_all_start_methods
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
)

import httpx
import numpy as np
from urllib3.util import Url, parse_url

from qdrant_client import grpc as grpc
from qdrant_client.client_base import QdrantBase
from qdrant_client.connection import get_async_channel, get_channel
from qdrant_client.conversions import common_types as types
from qdrant_client.conversions.conversion import GrpcToRest, RestToGrpc
from qdrant_client.http import ApiClient, SyncApis
from qdrant_client.http import models as rest_models
from qdrant_client.parallel_processor import ParallelWorkerPool
from qdrant_client.uploader.grpc_uploader import GrpcBatchUploader
from qdrant_client.uploader.rest_uploader import RestBatchUploader
from qdrant_client.uploader.uploader import BaseUploader


class QdrantRemote(QdrantBase):
    def __init__(
        self,
        url: Optional[str] = None,
        port: Optional[int] = 6333,
        grpc_port: int = 6334,
        prefer_grpc: bool = False,
        https: Optional[bool] = None,
        api_key: Optional[str] = None,
        prefix: Optional[str] = None,
        timeout: Optional[float] = None,
        host: Optional[str] = None,
        **kwargs: Any,
    ):
        self._prefer_grpc = prefer_grpc
        self._grpc_port = grpc_port
        self._https = https if https is not None else api_key is not None
        self._scheme = "https" if self._https else "http"

        self._prefix = prefix or ""
        if len(self._prefix) > 0 and self._prefix[0] != "/":
            self._prefix = f"/{self._prefix}"

        if url is not None and host is not None:
            raise ValueError(f"Only one of (url, host) can be set. url is {url}, host is {host}")

        if host is not None and (host.startswith("http://") or host.startswith("https://")):
            raise ValueError(
                f"`host` param is not expected to contain protocol (http:// or https://). "
                f"Try to use `url` parameter instead."
            )

        elif url:
            parsed_url: Url = parse_url(url)
            self._host, self._port = parsed_url.host, parsed_url.port

            if parsed_url.scheme:
                self._https = parsed_url.scheme == "https"
                self._scheme = parsed_url.scheme

            self._port = self._port if self._port else port

            if self._prefix and parsed_url.path:
                raise ValueError(
                    "Prefix can be set either in `url` or in `prefix`. "
                    f"url is {url}, prefix is {parsed_url.path}"
                )

            if self._scheme not in ("http", "https"):
                raise ValueError(f"Unknown scheme: {self._scheme}")
        else:
            self._host = host or "localhost"
            self._port = port

        self._timeout = timeout
        self._api_key = api_key

        limits = kwargs.pop("limits", None)
        if limits is None:
            if self._host in ["localhost", "127.0.0.1"]:
                # Disable keep-alive for local connections
                # Cause in some cases, it may cause extra delays
                limits = httpx.Limits(max_connections=None, max_keepalive_connections=0)

        http2 = kwargs.pop("http2", False)
        self._grpc_headers = []
        self._rest_headers = kwargs.pop("metadata", {})
        if api_key is not None:
            if self._scheme == "http":
                warnings.warn("Api key is used with unsecure connection.")

            http2 = True

            self._rest_headers["api-key"] = api_key
            self._grpc_headers.append(("api-key", api_key))

        address = f"{self._host}:{self._port}" if self._port is not None else self._host
        self.rest_uri = f"{self._scheme}://{address}{self._prefix}"

        self._rest_args = {"headers": self._rest_headers, "http2": http2, **kwargs}

        if limits is not None:
            self._rest_args["limits"] = limits

        if self._timeout is not None:
            self._rest_args["timeout"] = self._timeout

        self.openapi_client: SyncApis[ApiClient] = SyncApis(host=self.rest_uri, **self._rest_args)

        self._grpc_channel = None
        self._grpc_points_client: Optional[grpc.PointsStub] = None
        self._grpc_collections_client: Optional[grpc.CollectionsStub] = None

        self._aio_grpc_channel = None
        self._aio_grpc_points_client: Optional[grpc.PointsStub] = None
        self._aio_grpc_collections_client: Optional[grpc.CollectionsStub] = None

    def __del__(self) -> None:
        if hasattr(self, "_grpc_channel") and self._grpc_channel is not None:
            self._grpc_channel.close()

    @staticmethod
    def _parse_url(url: str) -> Tuple[Optional[str], str, Optional[int], Optional[str]]:
        parse_result: Url = parse_url(url)
        scheme, host, port, prefix = (
            parse_result.scheme,
            parse_result.host,
            parse_result.port,
            parse_result.path,
        )
        return scheme, host, port, prefix

    def _init_grpc_points_client(self) -> None:
        if self._grpc_channel is None:
            self._grpc_channel = get_channel(
                host=self._host, port=self._grpc_port, ssl=self._https, metadata=self._grpc_headers
            )
        self._grpc_points_client = grpc.PointsStub(self._grpc_channel)

    def _init_grpc_collections_client(self) -> None:
        if self._grpc_channel is None:
            self._grpc_channel = get_channel(
                host=self._host, port=self._grpc_port, ssl=self._https, metadata=self._grpc_headers
            )
        self._grpc_collections_client = grpc.CollectionsStub(self._grpc_channel)

    def _init_async_grpc_points_client(self) -> None:
        if self._aio_grpc_channel is None:
            self._aio_grpc_channel = get_async_channel(
                host=self._host, port=self._grpc_port, ssl=self._https, metadata=self._grpc_headers
            )
        self._aio_grpc_points_client = grpc.PointsStub(self._aio_grpc_channel)

    def _init_async_grpc_collections_client(self) -> None:
        if self._aio_grpc_channel is None:
            self._aio_grpc_channel = get_async_channel(
                host=self._host, port=self._grpc_port, ssl=self._https, metadata=self._grpc_headers
            )
        self._aio_grpc_collections_client = grpc.CollectionsStub(self._aio_grpc_channel)

    @property
    def async_grpc_collections(self) -> grpc.CollectionsStub:
        """gRPC client for collections methods

        Returns:
            An instance of raw gRPC client, generated from Protobuf
        """
        if self._aio_grpc_collections_client is None:
            self._init_async_grpc_collections_client()
        return self._aio_grpc_collections_client

    @property
    def async_grpc_points(self) -> grpc.PointsStub:
        """gRPC client for points methods

        Returns:
            An instance of raw gRPC client, generated from Protobuf
        """
        if self._aio_grpc_points_client is None:
            self._init_async_grpc_points_client()
        return self._aio_grpc_points_client

    @property
    def grpc_collections(self) -> grpc.CollectionsStub:
        """gRPC client for collections methods

        Returns:
            An instance of raw gRPC client, generated from Protobuf
        """
        if self._grpc_collections_client is None:
            self._init_grpc_collections_client()
        return self._grpc_collections_client

    @property
    def grpc_points(self) -> grpc.PointsStub:
        """gRPC client for points methods

        Returns:
            An instance of raw gRPC client, generated from Protobuf
        """
        if self._grpc_points_client is None:
            self._init_grpc_points_client()
        return self._grpc_points_client

    @property
    def rest(self) -> SyncApis[ApiClient]:
        """REST Client

        Returns:
            An instance of raw REST API client, generated from OpenAPI schema
        """
        return self.openapi_client

    @property
    def http(self) -> SyncApis[ApiClient]:
        """REST Client

        Returns:
            An instance of raw REST API client, generated from OpenAPI schema
        """
        return self.openapi_client

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
        if self._prefer_grpc:
            requests = [
                RestToGrpc.convert_search_request(r, collection_name)
                if isinstance(r, rest_models.SearchRequest)
                else r
                for r in requests
            ]

            if isinstance(consistency, (rest_models.ReadConsistencyType, int)):
                consistency = RestToGrpc.convert_read_consistency(consistency)

            grpc_res: grpc.SearchBatchResponse = self.grpc_points.SearchBatch(
                grpc.SearchBatchPoints(
                    collection_name=collection_name,
                    search_points=requests,
                    read_consistency=consistency,
                ),
                timeout=self._timeout,
            )

            return [
                [GrpcToRest.convert_scored_point(hit) for hit in r.result] for r in grpc_res.result
            ]
        else:
            requests = [
                GrpcToRest.convert_search_points(r) if isinstance(r, grpc.SearchPoints) else r
                for r in requests
            ]
            http_res: List[
                List[rest_models.ScoredPoint]
            ] = self.http.points_api.search_batch_points(
                collection_name=collection_name,
                consistency=consistency,
                search_request_batch=rest_models.SearchRequestBatch(searches=requests),
            ).result
            return http_res

    def search(
        self,
        collection_name: str,
        query_vector: Union[
            types.NumpyArray, Sequence[float], Tuple[str, List[float]], types.NamedVector
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

        if not append_payload:
            logging.warning(
                "Usage of `append_payload` is deprecated. Please consider using `with_payload` instead"
            )
            with_payload = append_payload

        if isinstance(query_vector, np.ndarray):
            query_vector = query_vector.tolist()

        if self._prefer_grpc:
            vector_name = None

            if isinstance(query_vector, types.NamedVector):
                vector = query_vector.vector
                vector_name = query_vector.name
            elif isinstance(query_vector, tuple):
                vector_name = query_vector[0]
                vector = query_vector[1]
            else:
                vector = list(query_vector)

            if isinstance(query_filter, rest_models.Filter):
                query_filter = RestToGrpc.convert_filter(model=query_filter)

            if isinstance(search_params, rest_models.SearchParams):
                search_params = RestToGrpc.convert_search_params(search_params)

            if isinstance(
                with_payload,
                (
                    bool,
                    list,
                    rest_models.PayloadSelectorInclude,
                    rest_models.PayloadSelectorExclude,
                ),
            ):
                with_payload = RestToGrpc.convert_with_payload_interface(with_payload)

            if isinstance(
                with_vectors,
                (
                    bool,
                    list,
                ),
            ):
                with_vectors = RestToGrpc.convert_with_vectors(with_vectors)

            if isinstance(consistency, (rest_models.ReadConsistencyType, int)):
                consistency = RestToGrpc.convert_read_consistency(consistency)

            res: grpc.SearchResponse = self.grpc_points.Search(
                grpc.SearchPoints(
                    collection_name=collection_name,
                    vector=vector,
                    vector_name=vector_name,
                    filter=query_filter,
                    limit=limit,
                    offset=offset,
                    with_vectors=with_vectors,
                    with_payload=with_payload,
                    params=search_params,
                    score_threshold=score_threshold,
                    read_consistency=consistency,
                ),
                timeout=self._timeout,
            )

            return [GrpcToRest.convert_scored_point(hit) for hit in res.result]

        else:
            if isinstance(query_vector, tuple):
                query_vector = types.NamedVector(name=query_vector[0], vector=query_vector[1])

            if isinstance(query_filter, grpc.Filter):
                query_filter = GrpcToRest.convert_filter(model=query_filter)

            if isinstance(search_params, grpc.SearchParams):
                search_params = GrpcToRest.convert_search_params(search_params)

            if isinstance(with_payload, grpc.WithPayloadSelector):
                with_payload = GrpcToRest.convert_with_payload_selector(with_payload)

            search_result = self.http.points_api.search_points(
                collection_name=collection_name,
                consistency=consistency,
                search_request=rest_models.SearchRequest(
                    vector=query_vector,
                    filter=query_filter,
                    limit=limit,
                    offset=offset,
                    params=search_params,
                    with_vector=with_vectors,
                    with_payload=with_payload,
                    score_threshold=score_threshold,
                ),
            )
            result: Optional[List[types.ScoredPoint]] = search_result.result
            assert result is not None, "Search returned None"
            return result

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
        if self._prefer_grpc:
            requests = [
                RestToGrpc.convert_recommend_request(r, collection_name)
                if isinstance(r, rest_models.RecommendRequest)
                else r
                for r in requests
            ]

            if isinstance(consistency, (rest_models.ReadConsistencyType, int)):
                consistency = RestToGrpc.convert_read_consistency(consistency)

            grpc_res: grpc.SearchBatchResponse = self.grpc_points.RecommendBatch(
                grpc.RecommendBatchPoints(
                    collection_name=collection_name,
                    recommend_points=requests,
                    read_consistency=consistency,
                ),
                timeout=self._timeout,
            )

            return [
                [GrpcToRest.convert_scored_point(hit) for hit in r.result] for r in grpc_res.result
            ]
        else:
            requests = [
                GrpcToRest.convert_recommend_points(r)
                if isinstance(r, grpc.RecommendPoints)
                else r
                for r in requests
            ]
            http_res: List[
                List[rest_models.ScoredPoint]
            ] = self.http.points_api.recommend_batch_points(
                collection_name=collection_name,
                consistency=consistency,
                recommend_request_batch=rest_models.RecommendRequestBatch(searches=requests),
            ).result
            return http_res

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

        if negative is None:
            negative = []

        if self._prefer_grpc:
            positive = [
                RestToGrpc.convert_extended_point_id(point_id)
                if isinstance(point_id, (str, int))
                else point_id
                for point_id in positive
            ]

            negative = [
                RestToGrpc.convert_extended_point_id(point_id)
                if isinstance(point_id, (str, int))
                else point_id
                for point_id in negative
            ]

            if isinstance(query_filter, rest_models.Filter):
                query_filter = RestToGrpc.convert_filter(model=query_filter)

            if isinstance(search_params, rest_models.SearchParams):
                search_params = RestToGrpc.convert_search_params(search_params)

            if isinstance(
                with_payload,
                (
                    bool,
                    list,
                    rest_models.PayloadSelectorInclude,
                    rest_models.PayloadSelectorExclude,
                ),
            ):
                with_payload = RestToGrpc.convert_with_payload_interface(with_payload)

            if isinstance(
                with_vectors,
                (
                    bool,
                    list,
                ),
            ):
                with_vectors = RestToGrpc.convert_with_vectors(with_vectors)

            if isinstance(lookup_from, rest_models.LookupLocation):
                lookup_from = RestToGrpc.convert_lookup_location(lookup_from)

            if isinstance(consistency, (rest_models.ReadConsistencyType, int)):
                consistency = RestToGrpc.convert_read_consistency(consistency)

            res: grpc.SearchResponse = self.grpc_points.Recommend(
                grpc.RecommendPoints(
                    collection_name=collection_name,
                    positive=positive,
                    negative=negative,
                    filter=query_filter,
                    limit=limit,
                    offset=offset,
                    with_vectors=with_vectors,
                    with_payload=with_payload,
                    params=search_params,
                    score_threshold=score_threshold,
                    using=using,
                    lookup_from=lookup_from,
                    read_consistency=consistency,
                ),
                timeout=self._timeout,
            )

            return [GrpcToRest.convert_scored_point(hit) for hit in res.result]
        else:
            positive = [
                GrpcToRest.convert_point_id(point_id)
                if isinstance(point_id, grpc.PointId)
                else point_id
                for point_id in positive
            ]

            negative = [
                GrpcToRest.convert_point_id(point_id)
                if isinstance(point_id, grpc.PointId)
                else point_id
                for point_id in negative
            ]

            if isinstance(query_filter, grpc.Filter):
                query_filter = GrpcToRest.convert_filter(model=query_filter)

            if isinstance(search_params, grpc.SearchParams):
                search_params = GrpcToRest.convert_search_params(search_params)

            if isinstance(with_payload, grpc.WithPayloadSelector):
                with_payload = GrpcToRest.convert_with_payload_selector(with_payload)

            if isinstance(lookup_from, grpc.LookupLocation):
                lookup_from = GrpcToRest.convert_lookup_location(lookup_from)

            result = self.openapi_client.points_api.recommend_points(
                collection_name=collection_name,
                consistency=consistency,
                recommend_request=rest_models.RecommendRequest(
                    filter=query_filter,
                    negative=negative,
                    params=search_params,
                    positive=positive,
                    limit=limit,
                    offset=offset,
                    with_payload=with_payload,
                    with_vector=with_vectors,
                    score_threshold=score_threshold,
                    lookup_from=lookup_from,
                    using=using,
                ),
            ).result
            assert result is not None, "Recommend points API returned None"
            return result

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
        if self._prefer_grpc:
            if isinstance(offset, (int, str)):
                offset = RestToGrpc.convert_extended_point_id(offset)

            if isinstance(scroll_filter, rest_models.Filter):
                scroll_filter = RestToGrpc.convert_filter(model=scroll_filter)

            if isinstance(
                with_payload,
                (
                    bool,
                    list,
                    rest_models.PayloadSelectorInclude,
                    rest_models.PayloadSelectorExclude,
                ),
            ):
                with_payload = RestToGrpc.convert_with_payload_interface(with_payload)

            if isinstance(
                with_vectors,
                (
                    bool,
                    list,
                ),
            ):
                with_vectors = RestToGrpc.convert_with_vectors(with_vectors)

            if isinstance(consistency, (rest_models.ReadConsistencyType, int)):
                consistency = RestToGrpc.convert_read_consistency(consistency)

            res: grpc.ScrollResponse = self.grpc_points.Scroll(
                grpc.ScrollPoints(
                    collection_name=collection_name,
                    filter=scroll_filter,
                    offset=offset,
                    with_vectors=with_vectors,
                    with_payload=with_payload,
                    limit=limit,
                    read_consistency=consistency,
                ),
                timeout=self._timeout,
            )

            return [
                GrpcToRest.convert_retrieved_point(point) for point in res.result
            ], GrpcToRest.convert_point_id(res.next_page_offset) if res.HasField(
                "next_page_offset"
            ) else None
        else:
            if isinstance(offset, grpc.PointId):
                offset = GrpcToRest.convert_point_id(offset)

            if isinstance(scroll_filter, grpc.Filter):
                scroll_filter = GrpcToRest.convert_filter(model=scroll_filter)

            if isinstance(with_payload, grpc.WithPayloadSelector):
                with_payload = GrpcToRest.convert_with_payload_selector(with_payload)

            scroll_result: Optional[
                rest_models.ScrollResult
            ] = self.openapi_client.points_api.scroll_points(
                collection_name=collection_name,
                consistency=consistency,
                scroll_request=rest_models.ScrollRequest(
                    filter=scroll_filter,
                    limit=limit,
                    offset=offset,
                    with_payload=with_payload,
                    with_vector=with_vectors,
                ),
            ).result
            assert scroll_result is not None, "Scroll points API returned None result"

            return scroll_result.points, scroll_result.next_page_offset

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
        if isinstance(count_filter, grpc.Filter):
            count_filter = GrpcToRest.convert_filter(model=count_filter)

        count_result = self.openapi_client.points_api.count_points(
            collection_name=collection_name,
            count_request=rest_models.CountRequest(filter=count_filter, exact=exact),
        ).result
        assert count_result is not None, "Count points returned None result"
        return count_result

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
        if self._prefer_grpc:
            if isinstance(points, rest_models.Batch):
                vectors_batch: List[grpc.Vectors] = RestToGrpc.convert_batch_vector_struct(
                    points.vectors, len(points.ids)
                )
                points = [
                    grpc.PointStruct(
                        id=RestToGrpc.convert_extended_point_id(points.ids[idx]),
                        vectors=vectors_batch[idx],
                        payload=RestToGrpc.convert_payload(points.payloads[idx])
                        if points.payloads is not None
                        else None,
                    )
                    for idx in range(len(points.ids))
                ]
            if isinstance(points, list):
                points = [
                    RestToGrpc.convert_point_struct(point)
                    if isinstance(point, rest_models.PointStruct)
                    else point
                    for point in points
                ]

            if isinstance(ordering, rest_models.WriteOrdering):
                ordering = RestToGrpc.convert_write_ordering(ordering)

            grpc_result = self.grpc_points.Upsert(
                grpc.UpsertPoints(
                    collection_name=collection_name, wait=wait, points=points, ordering=ordering
                ),
                timeout=self._timeout,
            ).result

            assert grpc_result is not None, "Upsert returned None result"
            return GrpcToRest.convert_update_result(grpc_result)
        else:
            if isinstance(points, list):
                points = [
                    GrpcToRest.convert_point_struct(point)
                    if isinstance(point, grpc.PointStruct)
                    else point
                    for point in points
                ]

                points = rest_models.PointsList(points=points)

            if isinstance(points, rest_models.Batch):
                points = rest_models.PointsBatch(batch=points)

            http_result = self.openapi_client.points_api.upsert_points(
                collection_name=collection_name,
                wait=wait,
                point_insert_operations=points,
                ordering=ordering,
            ).result
            assert http_result is not None, "Upsert returned None result"
            return http_result

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
        if self._prefer_grpc:
            if isinstance(
                with_payload,
                (
                    bool,
                    list,
                    rest_models.PayloadSelectorInclude,
                    rest_models.PayloadSelectorExclude,
                ),
            ):
                with_payload = RestToGrpc.convert_with_payload_interface(with_payload)

            ids = [
                RestToGrpc.convert_extended_point_id(idx) if isinstance(idx, (int, str)) else idx
                for idx in ids
            ]

            with_vectors = RestToGrpc.convert_with_vectors(with_vectors)

            if isinstance(consistency, (rest_models.ReadConsistencyType, int)):
                consistency = RestToGrpc.convert_read_consistency(consistency)

            result = self.grpc_points.Get(
                grpc.GetPoints(
                    collection_name=collection_name,
                    ids=ids,
                    with_payload=with_payload,
                    with_vectors=with_vectors,
                    read_consistency=consistency,
                ),
                timeout=self._timeout,
            ).result

            assert result is not None, "Retrieve returned None result"

            return [GrpcToRest.convert_retrieved_point(record) for record in result]

        else:
            if isinstance(with_payload, grpc.WithPayloadSelector):
                with_payload = GrpcToRest.convert_with_payload_selector(with_payload)

            ids = [
                GrpcToRest.convert_point_id(idx) if isinstance(idx, grpc.PointId) else idx
                for idx in ids
            ]

            http_result = self.openapi_client.points_api.get_points(
                collection_name=collection_name,
                consistency=consistency,
                point_request=rest_models.PointRequest(
                    ids=ids, with_payload=with_payload, with_vector=with_vectors
                ),
            ).result
            assert http_result is not None, "Retrieve API returned None result"
            return http_result

    @classmethod
    def _try_argument_to_grpc_selector(cls, points: types.PointsSelector) -> grpc.PointsSelector:
        if isinstance(points, list):
            points_selector = grpc.PointsSelector(
                points=grpc.PointsIdsList(
                    ids=[
                        RestToGrpc.convert_extended_point_id(idx)
                        if isinstance(idx, (int, str))
                        else idx
                        for idx in points
                    ]
                )
            )
        elif isinstance(points, grpc.PointsSelector):
            points_selector = points
        elif isinstance(points, (rest_models.PointIdsList, rest_models.FilterSelector)):
            points_selector = RestToGrpc.convert_points_selector(points)
        elif isinstance(points, rest_models.Filter):
            points_selector = RestToGrpc.convert_points_selector(
                rest_models.FilterSelector.construct(filter=points)
            )
        elif isinstance(points, grpc.Filter):
            points_selector = grpc.PointsSelector(filter=points)
        else:
            raise ValueError(f"Unsupported points selector type: {type(points)}")
        return points_selector

    @classmethod
    def _try_argument_to_rest_selector(
        cls, points: types.PointsSelector
    ) -> rest_models.PointsSelector:
        if isinstance(points, list):
            _points = [
                GrpcToRest.convert_point_id(idx) if isinstance(idx, grpc.PointId) else idx
                for idx in points
            ]
            points_selector = rest_models.PointIdsList.construct(points=_points)
        elif isinstance(points, grpc.PointsSelector):
            points_selector = GrpcToRest.convert_points_selector(points)
        elif isinstance(points, (rest_models.PointIdsList, rest_models.FilterSelector)):
            points_selector = points
        elif isinstance(points, rest_models.Filter):
            points_selector = rest_models.FilterSelector.construct(filter=points)
        elif isinstance(points, grpc.Filter):
            points_selector = rest_models.FilterSelector.construct(
                filter=GrpcToRest.convert_filter(points)
            )
        else:
            raise ValueError(f"Unsupported points selector type: {type(points)}")
        return points_selector

    @classmethod
    def _points_selector_to_points_list(
        cls, points_selector: grpc.PointsSelector
    ) -> List[grpc.PointId]:
        name = points_selector.WhichOneof("points_selector_one_of")
        val = getattr(points_selector, name)

        if name == "points":
            return list(val.ids)
        return []

    @classmethod
    def _try_argument_to_rest_points_and_filter(
        cls, points: types.PointsSelector
    ) -> Tuple[Optional[List[rest_models.ExtendedPointId]], Optional[rest_models.Filter]]:
        _points = None
        _filter = None
        if isinstance(points, list):
            _points = [
                GrpcToRest.convert_point_id(idx) if isinstance(idx, grpc.PointId) else idx
                for idx in points
            ]
        elif isinstance(points, grpc.PointsSelector):
            selector = GrpcToRest.convert_points_selector(points)
            if isinstance(selector, rest_models.PointIdsList):
                _points = selector.points
            elif isinstance(selector, rest_models.FilterSelector):
                _filter = selector.filter
        elif isinstance(points, rest_models.PointIdsList):
            _points = points.points
        elif isinstance(points, rest_models.FilterSelector):
            _filter = points.filter
        elif isinstance(points, rest_models.Filter):
            _filter = points
        elif isinstance(points, grpc.Filter):
            _filter = GrpcToRest.convert_filter(points)
        else:
            raise ValueError(f"Unsupported points selector type: {type(points)}")

        return _points, _filter

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
                Example:
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
        if self._prefer_grpc:
            points_selector = self._try_argument_to_grpc_selector(points_selector)

            if isinstance(ordering, rest_models.WriteOrdering):
                ordering = RestToGrpc.convert_write_ordering(ordering)

            return GrpcToRest.convert_update_result(
                self.grpc_points.Delete(
                    grpc.DeletePoints(
                        collection_name=collection_name,
                        wait=wait,
                        points=points_selector,
                        ordering=ordering,
                    ),
                    timeout=self._timeout,
                ).result
            )
        else:
            points_selector = self._try_argument_to_rest_selector(points_selector)
            result: Optional[types.UpdateResult] = self.openapi_client.points_api.delete_points(
                collection_name=collection_name,
                wait=wait,
                points_selector=points_selector,
                ordering=ordering,
            ).result
            assert result is not None, "Delete points returned None"
            return result

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
             Example:
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
        if self._prefer_grpc:
            points_selector = self._try_argument_to_grpc_selector(points)

            if isinstance(ordering, rest_models.WriteOrdering):
                ordering = RestToGrpc.convert_write_ordering(ordering)

            return GrpcToRest.convert_update_result(
                self.grpc_points.SetPayload(
                    grpc.SetPayloadPoints(
                        collection_name=collection_name,
                        wait=wait,
                        payload=RestToGrpc.convert_payload(payload),
                        points_selector=points_selector,
                        ordering=ordering,
                    ),
                    timeout=self._timeout,
                ).result
            )
        else:
            _points, _filter = self._try_argument_to_rest_points_and_filter(points)
            result: Optional[types.UpdateResult] = self.openapi_client.points_api.set_payload(
                collection_name=collection_name,
                wait=wait,
                ordering=ordering,
                set_payload=rest_models.SetPayload(
                    payload=payload,
                    points=_points,
                    filter=_filter,
                ),
            ).result
            assert result is not None, "Set payload returned None"
            return result

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
             Example:
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
        if self._prefer_grpc:
            points_selector = self._try_argument_to_grpc_selector(points)

            if isinstance(ordering, rest_models.WriteOrdering):
                ordering = RestToGrpc.convert_write_ordering(ordering)

            return GrpcToRest.convert_update_result(
                self.grpc_points.OverwritePayload(
                    grpc.SetPayloadPoints(
                        collection_name=collection_name,
                        wait=wait,
                        payload=RestToGrpc.convert_payload(payload),
                        points_selector=points_selector,
                        ordering=ordering,
                    ),
                    timeout=self._timeout,
                ).result
            )
        else:
            _points, _filter = self._try_argument_to_rest_points_and_filter(points)
            result: Optional[
                types.UpdateResult
            ] = self.openapi_client.points_api.overwrite_payload(
                collection_name=collection_name,
                wait=wait,
                ordering=ordering,
                set_payload=rest_models.SetPayload(
                    payload=payload,
                    points=_points,
                    filter=_filter,
                ),
            ).result
            assert result is not None, "Overwrite payload returned None"
            return result

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
                Example:
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
        if self._prefer_grpc:
            points_selector = self._try_argument_to_grpc_selector(points)
            if isinstance(ordering, rest_models.WriteOrdering):
                ordering = RestToGrpc.convert_write_ordering(ordering)
            return GrpcToRest.convert_update_result(
                self.grpc_points.DeletePayload(
                    grpc.DeletePayloadPoints(
                        collection_name=collection_name,
                        wait=wait,
                        keys=keys,
                        points_selector=points_selector,
                        ordering=ordering,
                    ),
                    timeout=self._timeout,
                ).result
            )
        else:
            _points, _filter = self._try_argument_to_rest_points_and_filter(points)
            result: Optional[types.UpdateResult] = self.openapi_client.points_api.delete_payload(
                collection_name=collection_name,
                wait=wait,
                ordering=ordering,
                delete_payload=rest_models.DeletePayload(
                    keys=keys,
                    points=_points,
                    filter=_filter,
                ),
            ).result
            assert result is not None, "Delete payload returned None"
            return result

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
                Example:
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
        if self._prefer_grpc:
            points_selector = self._try_argument_to_grpc_selector(points_selector)

            if isinstance(ordering, rest_models.WriteOrdering):
                ordering = RestToGrpc.convert_write_ordering(ordering)

            return GrpcToRest.convert_update_result(
                self.grpc_points.ClearPayload(
                    grpc.ClearPayloadPoints(
                        collection_name=collection_name,
                        wait=wait,
                        points=points_selector,
                        ordering=ordering,
                    ),
                    timeout=self._timeout,
                ).result
            )
        else:
            points_selector = self._try_argument_to_rest_selector(points_selector)
            result: Optional[types.UpdateResult] = self.openapi_client.points_api.clear_payload(
                collection_name=collection_name,
                wait=wait,
                ordering=ordering,
                points_selector=points_selector,
            ).result
            assert result is not None, "Clear payload returned None"
            return result

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
        change_aliases_operation = [
            GrpcToRest.convert_alias_operations(operation)
            if isinstance(operation, grpc.AliasOperations)
            else operation
            for operation in change_aliases_operations
        ]
        result: Optional[bool] = self.http.collections_api.update_aliases(
            timeout=timeout,
            change_aliases_operation=rest_models.ChangeAliasesOperation(
                actions=change_aliases_operation
            ),
        ).result
        assert result is not None, "Update aliases returned None"
        return result

    def get_collection_aliases(
        self, collection_name: str, **kwargs: Any
    ) -> types.CollectionsAliasesResponse:
        """Get collection aliases

        Args:
            collection_name: Name of the collection

        Returns:
            Collection aliases
        """
        result: Optional[
            types.CollectionsAliasesResponse
        ] = self.http.collections_api.get_collection_aliases(
            collection_name=collection_name
        ).result
        assert result is not None, "Get collection aliases returned None"
        return result

    def get_aliases(self, **kwargs: Any) -> types.CollectionsAliasesResponse:
        """Get all aliases

        Returns:
            All aliases of all collections
        """
        result: Optional[
            types.CollectionsAliasesResponse
        ] = self.http.collections_api.get_collections_aliases().result
        assert result is not None, "Get aliases returned None"
        return result

    def get_collections(self, **kwargs: Any) -> types.CollectionsResponse:
        """Get list name of all existing collections

        Returns:
            List of the collections
        """
        if self._prefer_grpc:
            response = self.grpc_collections.List(
                grpc.ListCollectionsRequest(), timeout=self._timeout
            ).collections
            return types.CollectionsResponse(
                collections=[
                    GrpcToRest.convert_collection_description(description)
                    for description in response
                ]
            )

        result: Optional[
            types.CollectionsResponse
        ] = self.http.collections_api.get_collections().result
        assert result is not None, "Get collections returned None"
        return result

    def get_collection(self, collection_name: str, **kwargs: Any) -> types.CollectionInfo:
        """Get detailed information about specified existing collection

        Args:
            collection_name: Name of the collection

        Returns:
            Detailed information about the collection
        """
        if self._prefer_grpc:
            return GrpcToRest.convert_collection_info(
                self.grpc_collections.Get(
                    grpc.GetCollectionInfoRequest(collection_name=collection_name),
                    timeout=self._timeout,
                ).result
            )
        result: Optional[types.CollectionInfo] = self.http.collections_api.get_collection(
            collection_name=collection_name
        ).result
        assert result is not None, "Get collection returned None"
        return result

    def update_collection(
        self,
        collection_name: str,
        optimizer_config: Optional[types.OptimizersConfigDiff] = None,
        collection_params: Optional[types.CollectionParamsDiff] = None,
        timeout: Optional[int] = None,
        **kwargs: Any,
    ) -> bool:
        """Update parameters of the collection

        Args:
            collection_name: Name of the collection
            optimizer_config: Override for optimizer configuration
            collection_params: Override for collection parameters
            timeout:
                Wait for operation commit timeout in seconds.
                If timeout is reached - request will return with service error.

        Returns:
            Operation result
        """
        if isinstance(optimizer_config, grpc.OptimizersConfigDiff):
            optimizer_config = GrpcToRest.convert_optimizers_config_diff(optimizer_config)

        if isinstance(collection_params, grpc.CollectionParamsDiff):
            collection_params = GrpcToRest.convert_collection_params_diff(collection_params)

        result: Optional[bool] = self.http.collections_api.update_collection(
            collection_name,
            update_collection=rest_models.UpdateCollection(
                optimizers_config=optimizer_config, params=collection_params
            ),
            timeout=timeout,
        ).result
        assert result is not None, "Update collection returned None"
        return result

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
        result: Optional[bool] = self.http.collections_api.delete_collection(
            collection_name, timeout=timeout
        ).result
        assert result is not None, "Delete collection returned None"
        return result

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
        if isinstance(hnsw_config, grpc.HnswConfigDiff):
            hnsw_config = GrpcToRest.convert_hnsw_config_diff(hnsw_config)

        if isinstance(optimizers_config, grpc.OptimizersConfigDiff):
            optimizers_config = GrpcToRest.convert_optimizers_config_diff(optimizers_config)

        if isinstance(wal_config, grpc.WalConfigDiff):
            wal_config = GrpcToRest.convert_wal_config_diff(wal_config)

        if isinstance(quantization_config, grpc.QuantizationConfig):
            quantization_config = GrpcToRest.convert_quantization_config(quantization_config)

        create_collection_request = rest_models.CreateCollection(
            vectors=vectors_config,
            shard_number=shard_number,
            replication_factor=replication_factor,
            write_consistency_factor=write_consistency_factor,
            on_disk_payload=on_disk_payload,
            hnsw_config=hnsw_config,
            optimizers_config=optimizers_config,
            wal_config=wal_config,
            quantization_config=quantization_config,
            init_from=init_from,
        )

        result: Optional[bool] = self.http.collections_api.create_collection(
            collection_name=collection_name,
            create_collection=create_collection_request,
            timeout=timeout,
        ).result

        assert result is not None, "Create collection returned None"
        return result

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

        self.delete_collection(collection_name, timeout=timeout)

        return self.create_collection(
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
            timeout=timeout,
        )

    @property
    def _updater_class(self) -> Type[BaseUploader]:
        if self._prefer_grpc:
            return GrpcBatchUploader
        else:
            return RestBatchUploader

    def _upload_collection(
        self,
        batches_iterator: Iterable,
        collection_name: str,
        max_retries: int,
        parallel: int = 1,
        method: Optional[str] = None,
    ) -> None:
        if method is not None:
            if method in get_all_start_methods():
                start_method = method
            else:
                raise ValueError(
                    f"Start methods {method} is not available, available methods: {get_all_start_methods()}"
                )
        else:
            start_method = "forkserver" if "forkserver" in get_all_start_methods() else "spawn"

        if self._prefer_grpc:
            updater_kwargs = {
                "collection_name": collection_name,
                "host": self._host,
                "port": self._grpc_port,
                "max_retries": max_retries,
                "ssl": self._https,
                "metadata": self._grpc_headers,
            }
        else:
            updater_kwargs = {
                "collection_name": collection_name,
                "uri": self.rest_uri,
                "max_retries": max_retries,
                **self._rest_args,
            }

        if parallel == 1:
            updater = self._updater_class.start(**updater_kwargs)
            for _ in updater.process(batches_iterator):
                pass
        else:
            pool = ParallelWorkerPool(parallel, self._updater_class, start_method=start_method)
            for _ in pool.unordered_map(batches_iterator, **updater_kwargs):
                pass

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

        batches_iterator = self._updater_class.iterate_records_batches(
            records=records, batch_size=batch_size
        )
        self._upload_collection(batches_iterator, collection_name, max_retries, parallel, method)

    def upload_collection(
        self,
        collection_name: str,
        vectors: Union[types.NumpyArray, Dict[str, types.NumpyArray], Iterable[List[float]]],
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
        batches_iterator = self._updater_class.iterate_batches(
            vectors=vectors, payload=payload, ids=ids, batch_size=batch_size
        )
        self._upload_collection(batches_iterator, collection_name, max_retries, parallel, method)

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
        if field_type is not None:
            warnings.warn("field_type is deprecated, use field_schema instead", DeprecationWarning)
            field_schema = field_type

        if isinstance(field_schema, int):  # type(grpc.PayloadSchemaType) == int
            field_schema = GrpcToRest.convert_payload_schema_type(field_schema)

        result: Optional[
            types.UpdateResult
        ] = self.openapi_client.collections_api.create_field_index(
            collection_name=collection_name,
            create_field_index=rest_models.CreateFieldIndex(
                field_name=field_name, field_schema=field_schema
            ),
            wait=wait,
            ordering=ordering,
        ).result
        assert result is not None, "Create field index returned None"
        return result

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
        result: Optional[
            types.UpdateResult
        ] = self.openapi_client.collections_api.delete_field_index(
            collection_name=collection_name,
            field_name=field_name,
            wait=wait,
            ordering=ordering,
        ).result
        assert result is not None, "Delete field index returned None"
        return result

    def list_snapshots(
        self, collection_name: str, **kwargs: Any
    ) -> List[types.SnapshotDescription]:
        """List all snapshots for a given collection.

        Args:
            collection_name: Name of the collection

        Returns:
            List of snapshots
        """
        snapshots = self.openapi_client.collections_api.list_snapshots(
            collection_name=collection_name
        ).result
        assert snapshots is not None, "List snapshots API returned None result"
        return snapshots

    def create_snapshot(
        self, collection_name: str, **kwargs: Any
    ) -> Optional[types.SnapshotDescription]:
        """Create snapshot for a given collection.

        Args:
            collection_name: Name of the collection

        Returns:
            Snapshot description
        """
        return self.openapi_client.collections_api.create_snapshot(
            collection_name=collection_name
        ).result

    def delete_snapshot(self, collection_name: str, snapshot_name: str, **kwargs: Any) -> bool:
        """Delete snapshot for a given collection.

        Args:
            collection_name: Name of the collection
            snapshot_name: Snapshot id

        Returns:
            True if snapshot was deleted
        """
        result: Optional[bool] = self.openapi_client.collections_api.delete_snapshot(
            collection_name=collection_name,
            snapshot_name=snapshot_name,
        ).result
        assert result is not None, "Delete snapshot API returned None"
        return result

    def list_full_snapshots(self, **kwargs: Any) -> List[types.SnapshotDescription]:
        """List all snapshots for a whole storage

        Returns:
            List of snapshots
        """
        snapshots = self.openapi_client.snapshots_api.list_full_snapshots().result
        assert snapshots is not None, "List full snapshots API returned None result"
        return snapshots

    def create_full_snapshot(self, **kwargs: Any) -> types.SnapshotDescription:
        """Create snapshot for a whole storage.

        Returns:
            Snapshot description
        """
        snapshot_description = self.openapi_client.snapshots_api.create_full_snapshot().result
        assert snapshot_description is not None, "Create full snapshot API returned None result"
        return snapshot_description

    def delete_full_snapshot(self, snapshot_name: str, **kwargs: Any) -> bool:
        """Delete snapshot for a whole storage.

        Args:
            snapshot_name: Snapshot name

        Returns:
            True if snapshot was deleted
        """
        result: Optional[bool] = self.openapi_client.snapshots_api.delete_full_snapshot(
            snapshot_name=snapshot_name,
        ).result
        assert result is not None, "Delete full snapshot API returned None"
        return result

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
        success = self.openapi_client.snapshots_api.recover_from_snapshot(
            collection_name=collection_name,
            snapshot_recover=rest_models.SnapshotRecover(location=location, priority=priority),
        ).result
        assert success is not None, "Recover from snapshot API returned None result"
        return success

    def lock_storage(self, reason: str, **kwargs: Any) -> types.LocksOption:
        """Lock storage for writing."""
        result: Optional[types.LocksOption] = self.openapi_client.service_api.post_locks(
            rest_models.LocksOption(error_message=reason, write=True)
        ).result
        assert result is not None, "Lock storage returned None"
        return result

    def unlock_storage(self, **kwargs: Any) -> types.LocksOption:
        """Unlock storage for writing."""
        result: Optional[types.LocksOption] = self.openapi_client.service_api.post_locks(
            rest_models.LocksOption(write=False)
        ).result
        assert result is not None, "Post locks returned None"
        return result

    def get_locks(self, **kwargs: Any) -> types.LocksOption:
        """Get current locks state."""
        result: Optional[types.LocksOption] = self.openapi_client.service_api.get_locks().result
        assert result is not None, "Get locks returned None"
        return result
