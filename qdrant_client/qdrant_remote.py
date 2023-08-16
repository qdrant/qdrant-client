import asyncio
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
from qdrant_client._pydantic_compat import construct
from qdrant_client.client_base import QdrantBase
from qdrant_client.connection import get_async_channel, get_channel
from qdrant_client.conversions import common_types as types
from qdrant_client.conversions.conversion import (
    GrpcToRest,
    RestToGrpc,
    grpc_payload_schema_to_field_type,
)
from qdrant_client.http import ApiClient, SyncApis
from qdrant_client.http import models
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
            if url.startswith("localhost"):
                # Handle for a special case when url is localhost:port
                # Which is not parsed correctly by urllib
                url = f"//{url}"

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
        self._grpc_snapshots_client: Optional[grpc.SnapshotsStub] = None

        self._aio_grpc_channel = None
        self._aio_grpc_points_client: Optional[grpc.PointsStub] = None
        self._aio_grpc_collections_client: Optional[grpc.CollectionsStub] = None
        self._aio_grpc_snapshots_client: Optional[grpc.SnapshotsStub] = None

        self._closed: bool = False

    @property
    def closed(self) -> bool:
        return self._closed

    def close(self, grpc_grace: Optional[float] = None, **kwargs: Any) -> None:
        if hasattr(self, "_grpc_channel") and self._grpc_channel is not None:
            try:
                self._grpc_channel.close()
            except AttributeError:
                logging.warning(
                    "Unable to close grpc_channel. Connection was interrupted on the server side"
                )

        if hasattr(self, "_aio_grpc_channel") and self._aio_grpc_channel is not None:
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(self._aio_grpc_channel.close(grace=grpc_grace))
            except AttributeError:
                logging.warning(
                    "Unable to close aio_grpc_channel. Connection was interrupted on the server side"
                )
            except RuntimeError:
                pass

        try:
            self.openapi_client.close()
        except Exception:
            logging.warning(
                "Unable to close http connection. Connection was interrupted on the server side"
            )

        self._closed = True

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

    def _init_grpc_channel(self) -> None:
        if self._closed:
            raise RuntimeError("Client was closed. Please create a new QdrantClient instance.")

        if self._grpc_channel is None:
            self._grpc_channel = get_channel(
                host=self._host,
                port=self._grpc_port,
                ssl=self._https,
                metadata=self._grpc_headers,
            )

    def _init_async_grpc_channel(self) -> None:
        if self._closed:
            raise RuntimeError("Client was closed. Please create a new QdrantClient instance.")

        if self._aio_grpc_channel is None:
            self._aio_grpc_channel = get_async_channel(
                host=self._host,
                port=self._grpc_port,
                ssl=self._https,
                metadata=self._grpc_headers,
            )

    def _init_grpc_points_client(self) -> None:
        self._init_grpc_channel()
        self._grpc_points_client = grpc.PointsStub(self._grpc_channel)

    def _init_grpc_collections_client(self) -> None:
        self._init_grpc_channel()
        self._grpc_collections_client = grpc.CollectionsStub(self._grpc_channel)

    def _init_grpc_snapshots_client(self) -> None:
        self._init_grpc_channel()
        self._grpc_snapshots_client = grpc.SnapshotsStub(self._grpc_channel)

    def _init_async_grpc_points_client(self) -> None:
        self._init_async_grpc_channel()
        self._aio_grpc_points_client = grpc.PointsStub(self._aio_grpc_channel)

    def _init_async_grpc_collections_client(self) -> None:
        self._init_async_grpc_channel()
        self._aio_grpc_collections_client = grpc.CollectionsStub(self._aio_grpc_channel)

    def _init_async_grpc_snapshots_client(self) -> None:
        self._init_async_grpc_channel()
        self._aio_grpc_snapshots_client = grpc.SnapshotsStub(self._aio_grpc_channel)

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
    def async_grpc_snapshots(self) -> grpc.SnapshotsStub:
        """gRPC client for snapshots methods

        Returns:
            An instance of raw gRPC client, generated from Protobuf
        """
        if self._aio_grpc_snapshots_client is None:
            self._init_async_grpc_snapshots_client()
        return self._aio_grpc_snapshots_client

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
    def grpc_snapshots(self) -> grpc.SnapshotsStub:
        """gRPC client for snapshots methods

        Returns:
            An instance of raw gRPC client, generated from Protobuf
        """
        if self._grpc_snapshots_client is None:
            self._init_grpc_snapshots_client()
        return self._grpc_snapshots_client

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
        query_filter: Optional[models.Filter] = None,
        search_params: Optional[models.SearchParams] = None,
        limit: int = 10,
        group_size: int = 1,
        with_payload: Union[bool, Sequence[str], models.PayloadSelector] = True,
        with_vectors: Union[bool, Sequence[str]] = False,
        score_threshold: Optional[float] = None,
        with_lookup: Optional[types.WithLookupInterface] = None,
        consistency: Optional[types.ReadConsistency] = None,
        **kwargs: Any,
    ) -> types.GroupsResult:
        if self._prefer_grpc:
            vector_name = None

            if isinstance(with_lookup, rest_models.WithLookup):
                with_lookup = RestToGrpc.convert_with_lookup(with_lookup)

            if isinstance(with_lookup, str):
                with_lookup = grpc.WithLookup(lookup=with_lookup)

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

            result: grpc.GroupsResult = self.grpc_points.SearchGroups(
                grpc.SearchPointGroups(
                    collection_name=collection_name,
                    vector=vector,
                    vector_name=vector_name,
                    filter=query_filter,
                    limit=limit,
                    group_size=group_size,
                    with_vectors=with_vectors,
                    with_payload=with_payload,
                    params=search_params,
                    score_threshold=score_threshold,
                    group_by=group_by,
                    read_consistency=consistency,
                    with_lookup=with_lookup,
                ),
                timeout=self._timeout,
            ).result

            return GrpcToRest.convert_groups_result(result)
        else:
            if isinstance(with_lookup, grpc.WithLookup):
                with_lookup = GrpcToRest.convert_with_lookup(with_lookup)

            if isinstance(query_vector, tuple):
                query_vector = construct(
                    rest_models.NamedVector, name=query_vector[0], vector=query_vector[1]
                )

            if isinstance(query_vector, np.ndarray):
                query_vector = query_vector.tolist()

            if isinstance(query_filter, grpc.Filter):
                query_filter = GrpcToRest.convert_filter(model=query_filter)

            if isinstance(search_params, grpc.SearchParams):
                search_params = GrpcToRest.convert_search_params(search_params)

            if isinstance(with_payload, grpc.WithPayloadSelector):
                with_payload = GrpcToRest.convert_with_payload_selector(with_payload)

            search_groups_request = construct(
                rest_models.SearchGroupsRequest,
                vector=query_vector,
                filter=query_filter,
                params=search_params,
                with_payload=with_payload,
                with_vector=with_vectors,
                score_threshold=score_threshold,
                group_by=group_by,
                group_size=group_size,
                limit=limit,
                with_lookup=with_lookup,
            )

            return self.openapi_client.points_api.search_point_groups(
                search_groups_request=search_groups_request,
                collection_name=collection_name,
                consistency=consistency,
            ).result

    def recommend_batch(
        self,
        collection_name: str,
        requests: Sequence[types.RecommendRequest],
        consistency: Optional[types.ReadConsistency] = None,
        **kwargs: Any,
    ) -> List[List[types.ScoredPoint]]:
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

    def recommend_groups(
        self,
        collection_name: str,
        group_by: str,
        positive: Sequence[types.PointId],
        negative: Optional[Sequence[types.PointId]] = None,
        query_filter: Optional[models.Filter] = None,
        search_params: Optional[models.SearchParams] = None,
        limit: int = 10,
        group_size: int = 1,
        score_threshold: Optional[float] = None,
        with_payload: Union[bool, Sequence[str], models.PayloadSelector] = True,
        with_vectors: Union[bool, Sequence[str]] = False,
        using: Optional[str] = None,
        lookup_from: Optional[models.LookupLocation] = None,
        with_lookup: Optional[types.WithLookupInterface] = None,
        consistency: Optional[models.ReadConsistencyType] = None,
        **kwargs: Any,
    ) -> types.GroupsResult:
        if negative is None:
            negative = []

        if self._prefer_grpc:
            if isinstance(with_lookup, rest_models.WithLookup):
                with_lookup = RestToGrpc.convert_with_lookup(with_lookup)

            if isinstance(with_lookup, str):
                with_lookup = grpc.WithLookup(lookup_index=with_lookup)

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

            res: grpc.GroupsResult = self.grpc_points.RecommendGroups(
                grpc.RecommendPointGroups(
                    collection_name=collection_name,
                    positive=positive,
                    negative=negative,
                    filter=query_filter,
                    group_by=group_by,
                    limit=limit,
                    group_size=group_size,
                    with_vectors=with_vectors,
                    with_payload=with_payload,
                    params=search_params,
                    score_threshold=score_threshold,
                    using=using,
                    lookup_from=lookup_from,
                    read_consistency=consistency,
                    with_lookup=with_lookup,
                ),
                timeout=self._timeout,
            ).result

            assert res is not None, "Recommend groups API returned None"
            return GrpcToRest.convert_groups_result(res)
        else:
            if isinstance(with_lookup, grpc.WithLookup):
                with_lookup = GrpcToRest.convert_with_lookup(with_lookup)

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

            result = self.openapi_client.points_api.recommend_point_groups(
                collection_name=collection_name,
                consistency=consistency,
                recommend_groups_request=construct(
                    rest_models.RecommendGroupsRequest,
                    positive=positive,
                    negative=negative,
                    filter=query_filter,
                    group_by=group_by,
                    limit=limit,
                    group_size=group_size,
                    params=search_params,
                    with_payload=with_payload,
                    with_vector=with_vectors,
                    score_threshold=score_threshold,
                    lookup_from=lookup_from,
                    using=using,
                    with_lookup=with_lookup,
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
        if self._prefer_grpc:
            if isinstance(count_filter, rest_models.Filter):
                count_filter = RestToGrpc.convert_filter(model=count_filter)
            response = self.grpc_points.Count(
                grpc.CountPoints(
                    collection_name=collection_name, filter=count_filter, exact=exact
                ),
                timeout=self._timeout,
            ).result
            return GrpcToRest.convert_count_result(response)

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
                    collection_name=collection_name,
                    wait=wait,
                    points=points,
                    ordering=ordering,
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

    def update_vectors(
        self,
        collection_name: str,
        vectors: Sequence[types.PointVectors],
        wait: bool = True,
        ordering: Optional[types.WriteOrdering] = None,
        **kwargs: Any,
    ) -> types.UpdateResult:
        if self._prefer_grpc:
            vectors = [RestToGrpc.convert_point_vectors(vector) for vector in vectors]

            if isinstance(ordering, rest_models.WriteOrdering):
                ordering = RestToGrpc.convert_write_ordering(ordering)

            grpc_result = self.grpc_points.UpdateVectors(
                grpc.UpdatePointVectors(
                    collection_name=collection_name,
                    wait=wait,
                    vectors=vectors,
                    ordering=ordering,
                )
            ).result
            assert grpc_result is not None, "Upsert returned None result"
            return GrpcToRest.convert_update_result(grpc_result)
        else:
            return self.openapi_client.points_api.update_vectors(
                collection_name=collection_name,
                wait=wait,
                update_vectors=rest_models.UpdateVectors(points=vectors),
                ordering=ordering,
            ).result

    def delete_vectors(
        self,
        collection_name: str,
        vectors: Sequence[str],
        points: types.PointsSelector,
        wait: bool = True,
        ordering: Optional[types.WriteOrdering] = None,
        **kwargs: Any,
    ) -> types.UpdateResult:
        if self._prefer_grpc:
            points = self._try_argument_to_grpc_selector(points)

            if isinstance(ordering, rest_models.WriteOrdering):
                ordering = RestToGrpc.convert_write_ordering(ordering)

            grpc_result = self.grpc_points.DeleteVectors(
                grpc.DeletePointVectors(
                    collection_name=collection_name,
                    wait=wait,
                    vectors=grpc.VectorsSelector(
                        names=vectors,
                    ),
                    points=points,
                    ordering=ordering,
                )
            ).result

            assert grpc_result is not None, "Delete vectors returned None result"

            return GrpcToRest.convert_update_result(grpc_result)
        else:
            _points, _filter = self._try_argument_to_rest_points_and_filter(points)
            return self.openapi_client.points_api.delete_vectors(
                collection_name=collection_name,
                wait=wait,
                ordering=ordering,
                delete_vectors=construct(
                    rest_models.DeleteVectors,
                    vector=vectors,
                    points=_points,
                    filter=_filter,
                ),
            ).result

    def retrieve(
        self,
        collection_name: str,
        ids: Sequence[types.PointId],
        with_payload: Union[bool, Sequence[str], types.PayloadSelector] = True,
        with_vectors: Union[bool, Sequence[str]] = False,
        consistency: Optional[types.ReadConsistency] = None,
        **kwargs: Any,
    ) -> List[types.Record]:
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
                construct(rest_models.FilterSelector, filter=points)
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
            points_selector = construct(rest_models.PointIdsList, points=_points)
        elif isinstance(points, grpc.PointsSelector):
            points_selector = GrpcToRest.convert_points_selector(points)
        elif isinstance(points, (rest_models.PointIdsList, rest_models.FilterSelector)):
            points_selector = points
        elif isinstance(points, rest_models.Filter):
            points_selector = construct(rest_models.FilterSelector, filter=points)
        elif isinstance(points, grpc.Filter):
            points_selector = construct(
                rest_models.FilterSelector, filter=GrpcToRest.convert_filter(points)
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
        if self._prefer_grpc:
            change_aliases_operation = [
                RestToGrpc.convert_alias_operations(operation)
                if not isinstance(operation, grpc.AliasOperations)
                else operation
                for operation in change_aliases_operations
            ]
            return self.grpc_collections.UpdateAliases(
                grpc.ChangeAliases(
                    timeout=timeout,
                    actions=change_aliases_operation,
                ),
                timeout=self._timeout,
            ).result

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
        if self._prefer_grpc:
            response = self.grpc_collections.ListCollectionAliases(
                grpc.ListCollectionAliasesRequest(collection_name=collection_name),
                timeout=self._timeout,
            ).aliases
            return types.CollectionsAliasesResponse(
                aliases=[
                    GrpcToRest.convert_alias_description(description) for description in response
                ]
            )

        result: Optional[
            types.CollectionsAliasesResponse
        ] = self.http.collections_api.get_collection_aliases(
            collection_name=collection_name
        ).result
        assert result is not None, "Get collection aliases returned None"
        return result

    def get_aliases(self, **kwargs: Any) -> types.CollectionsAliasesResponse:
        if self._prefer_grpc:
            response = self.grpc_collections.ListAliases(
                grpc.ListAliasesRequest(), timeout=self._timeout
            ).aliases
            return types.CollectionsAliasesResponse(
                aliases=[
                    GrpcToRest.convert_alias_description(description) for description in response
                ]
            )
        result: Optional[
            types.CollectionsAliasesResponse
        ] = self.http.collections_api.get_collections_aliases().result
        assert result is not None, "Get aliases returned None"
        return result

    def get_collections(self, **kwargs: Any) -> types.CollectionsResponse:
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
        optimizers_config: Optional[types.OptimizersConfigDiff] = None,
        collection_params: Optional[types.CollectionParamsDiff] = None,
        vectors_config: Optional[types.VectorsConfigDiff] = None,
        hnsw_config: Optional[types.HnswConfigDiff] = None,
        quantization_config: Optional[types.QuantizationConfigDiff] = None,
        timeout: Optional[int] = None,
        **kwargs: Any,
    ) -> bool:
        if self._prefer_grpc:
            if isinstance(optimizers_config, rest_models.OptimizersConfigDiff):
                optimizers_config = RestToGrpc.convert_optimizers_config_diff(optimizers_config)

            if isinstance(collection_params, rest_models.CollectionParamsDiff):
                collection_params = RestToGrpc.convert_collection_params_diff(collection_params)

            if isinstance(vectors_config, dict):
                vectors_config = RestToGrpc.convert_vectors_config_diff(vectors_config)

            if isinstance(hnsw_config, rest_models.HnswConfigDiff):
                hnsw_config = RestToGrpc.convert_hnsw_config_diff(hnsw_config)

            # if isinstance(quantization_config, get_args(rest_models.QuantizationConfigDiff)): <-- deprecate python 3.7
            if isinstance(
                quantization_config,
                (
                    rest_models.ScalarQuantization,
                    rest_models.ProductQuantization,
                    rest_models.Disabled,
                ),
            ):
                quantization_config = RestToGrpc.convert_quantization_config_diff(
                    quantization_config
                )

            return self.grpc_collections.Update(
                grpc.UpdateCollection(
                    collection_name=collection_name,
                    optimizers_config=optimizers_config,
                    params=collection_params,
                    vectors_config=vectors_config,
                    hnsw_config=hnsw_config,
                    quantization_config=quantization_config,
                ),
                timeout=self._timeout,
            ).result

        if isinstance(optimizers_config, grpc.OptimizersConfigDiff):
            optimizers_config = GrpcToRest.convert_optimizers_config_diff(optimizers_config)

        if isinstance(collection_params, grpc.CollectionParamsDiff):
            collection_params = GrpcToRest.convert_collection_params_diff(collection_params)

        if isinstance(vectors_config, grpc.VectorsConfigDiff):
            vectors_config = GrpcToRest.convert_vectors_config_diff(vectors_config)

        if isinstance(hnsw_config, grpc.HnswConfigDiff):
            hnsw_config = GrpcToRest.convert_hnsw_config_diff(hnsw_config)

        if isinstance(quantization_config, grpc.QuantizationConfigDiff):
            quantization_config = GrpcToRest.convert_quantization_config_diff(quantization_config)

        result: Optional[bool] = self.http.collections_api.update_collection(
            collection_name,
            update_collection=rest_models.UpdateCollection(
                optimizers_config=optimizers_config,
                params=collection_params,
                vectors=vectors_config,
                hnsw_config=hnsw_config,
                quantization_config=quantization_config,
            ),
            timeout=timeout,
        ).result
        assert result is not None, "Update collection returned None"
        return result

    def delete_collection(
        self, collection_name: str, timeout: Optional[int] = None, **kwargs: Any
    ) -> bool:
        if self._prefer_grpc:
            return self.grpc_collections.Delete(
                grpc.DeleteCollection(collection_name=collection_name),
                timeout=self._timeout,
            ).result

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
        if self._prefer_grpc:
            if isinstance(vectors_config, (rest_models.VectorParams, dict)):
                vectors_config = RestToGrpc.convert_vectors_config(vectors_config)

            if isinstance(hnsw_config, rest_models.HnswConfigDiff):
                hnsw_config = RestToGrpc.convert_hnsw_config_diff(hnsw_config)

            if isinstance(optimizers_config, rest_models.OptimizersConfigDiff):
                optimizers_config = RestToGrpc.convert_optimizers_config_diff(optimizers_config)

            if isinstance(wal_config, rest_models.WalConfigDiff):
                wal_config = RestToGrpc.convert_wal_config_diff(wal_config)

            if isinstance(
                quantization_config,
                (rest_models.ScalarQuantization, rest_models.ProductQuantization),
            ):
                quantization_config = RestToGrpc.convert_quantization_config(quantization_config)

            create_collection = grpc.CreateCollection(
                collection_name=collection_name,
                hnsw_config=hnsw_config,
                wal_config=wal_config,
                optimizers_config=optimizers_config,
                shard_number=shard_number,
                on_disk_payload=on_disk_payload,
                timeout=timeout,
                vectors_config=vectors_config,
                replication_factor=replication_factor,
                write_consistency_factor=write_consistency_factor,
                init_from_collection=init_from,
                quantization_config=quantization_config,
            )
            return self.grpc_collections.Create(create_collection).result

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
            init_from=init_from,
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
        batches_iterator = self._updater_class.iterate_records_batches(
            records=records, batch_size=batch_size
        )
        self._upload_collection(batches_iterator, collection_name, max_retries, parallel, method)

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
        if field_type is not None:
            warnings.warn("field_type is deprecated, use field_schema instead", DeprecationWarning)
            field_schema = field_type

        if self._prefer_grpc:
            field_index_params = None
            if isinstance(field_schema, rest_models.PayloadSchemaType):
                field_schema = RestToGrpc.convert_payload_schema_type(field_schema)

            if isinstance(field_schema, int):
                # There are no means to distinguish grpc.PayloadSchemaType and grpc.FieldType,
                # as both of them are just ints
                # method signature assumes that grpc.PayloadSchemaType is passed,
                # otherwise the value will be corrupted
                field_schema = grpc_payload_schema_to_field_type(field_schema)

            if isinstance(field_schema, rest_models.TextIndexParams):
                field_index_params = grpc.PayloadIndexParams(
                    text_index_params=RestToGrpc.convert_text_index_params(field_schema)
                )
                field_schema = grpc.FieldType.FieldTypeText

            if isinstance(field_schema, grpc.PayloadIndexParams):
                field_index_params = field_schema
                field_schema = grpc.FieldType.FieldTypeText

            request = grpc.CreateFieldIndexCollection(
                collection_name=collection_name,
                field_name=field_name,
                field_type=field_schema,
                field_index_params=field_index_params,
                wait=wait,
                ordering=ordering,
            )
            return GrpcToRest.convert_update_result(
                self.grpc_points.CreateFieldIndex(request).result
            )

        if isinstance(field_schema, int):  # type(grpc.PayloadSchemaType) == int
            field_schema = GrpcToRest.convert_payload_schema_type(field_schema)

        if isinstance(field_schema, grpc.PayloadIndexParams):
            field_schema = GrpcToRest.convert_payload_schema_params(field_schema)

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
        if self._prefer_grpc:
            request = grpc.DeleteFieldIndexCollection(
                collection_name=collection_name,
                field_name=field_name,
                wait=wait,
                ordering=ordering,
            )
            return GrpcToRest.convert_update_result(
                self.grpc_points.DeleteFieldIndex(request).result
            )

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
        if self._prefer_grpc:
            snapshots = self.grpc_snapshots.List(
                grpc.ListSnapshotsRequest(collection_name=collection_name)
            ).snapshot_descriptions
            return [GrpcToRest.convert_snapshot_description(snapshot) for snapshot in snapshots]

        snapshots = self.openapi_client.collections_api.list_snapshots(
            collection_name=collection_name
        ).result
        assert snapshots is not None, "List snapshots API returned None result"
        return snapshots

    def create_snapshot(
        self, collection_name: str, **kwargs: Any
    ) -> Optional[types.SnapshotDescription]:
        if self._prefer_grpc:
            snapshot = self.grpc_snapshots.Create(
                grpc.CreateSnapshotRequest(collection_name=collection_name)
            ).snapshot_description
            return GrpcToRest.convert_snapshot_description(snapshot)

        return self.openapi_client.collections_api.create_snapshot(
            collection_name=collection_name
        ).result

    def delete_snapshot(self, collection_name: str, snapshot_name: str, **kwargs: Any) -> bool:
        if self._prefer_grpc:
            self.grpc_snapshots.Delete(
                grpc.DeleteSnapshotRequest(
                    collection_name=collection_name, snapshot_name=snapshot_name
                )
            )
            return True

        result: Optional[bool] = self.openapi_client.collections_api.delete_snapshot(
            collection_name=collection_name,
            snapshot_name=snapshot_name,
        ).result
        assert result is not None, "Delete snapshot API returned None"
        return result

    def list_full_snapshots(self, **kwargs: Any) -> List[types.SnapshotDescription]:
        if self._prefer_grpc:
            snapshots = self.grpc_snapshots.ListFull(
                grpc.ListFullSnapshotsRequest()
            ).snapshot_descriptions
            return [GrpcToRest.convert_snapshot_description(snapshot) for snapshot in snapshots]

        snapshots = self.openapi_client.snapshots_api.list_full_snapshots().result
        assert snapshots is not None, "List full snapshots API returned None result"
        return snapshots

    def create_full_snapshot(self, **kwargs: Any) -> types.SnapshotDescription:
        if self._prefer_grpc:
            snapshot_description = self.grpc_snapshots.CreateFull(
                grpc.CreateFullSnapshotRequest()
            ).snapshot_description
            return GrpcToRest.convert_snapshot_description(snapshot_description)

        snapshot_description = self.openapi_client.snapshots_api.create_full_snapshot().result
        assert snapshot_description is not None, "Create full snapshot API returned None result"
        return snapshot_description

    def delete_full_snapshot(self, snapshot_name: str, **kwargs: Any) -> bool:
        if self._prefer_grpc:
            self.grpc_snapshots.DeleteFull(
                grpc.DeleteFullSnapshotRequest(snapshot_name=snapshot_name)
            )
            return True
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
        success = self.openapi_client.snapshots_api.recover_from_snapshot(
            collection_name=collection_name,
            snapshot_recover=rest_models.SnapshotRecover(location=location, priority=priority),
        ).result
        assert success is not None, "Recover from snapshot API returned None result"
        return success

    def lock_storage(self, reason: str, **kwargs: Any) -> types.LocksOption:
        result: Optional[types.LocksOption] = self.openapi_client.service_api.post_locks(
            rest_models.LocksOption(error_message=reason, write=True)
        ).result
        assert result is not None, "Lock storage returned None"
        return result

    def unlock_storage(self, **kwargs: Any) -> types.LocksOption:
        result: Optional[types.LocksOption] = self.openapi_client.service_api.post_locks(
            rest_models.LocksOption(write=False)
        ).result
        assert result is not None, "Post locks returned None"
        return result

    def get_locks(self, **kwargs: Any) -> types.LocksOption:
        result: Optional[types.LocksOption] = self.openapi_client.service_api.get_locks().result
        assert result is not None, "Get locks returned None"
        return result
