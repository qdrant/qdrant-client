# flake8: noqa E501
from typing import TYPE_CHECKING, Any, Dict, Set, TypeVar, Union

from pydantic import BaseModel
from pydantic.main import BaseModel
from pydantic.version import VERSION as PYDANTIC_VERSION
from qdrant_client.http.models import *
from qdrant_client.http.models import models as m

PYDANTIC_V2 = PYDANTIC_VERSION.startswith("2.")
Model = TypeVar("Model", bound="BaseModel")

SetIntStr = Set[Union[int, str]]
DictIntStrAny = Dict[Union[int, str], Any]
file = None


def to_json(model: BaseModel, *args: Any, **kwargs: Any) -> str:
    if PYDANTIC_V2:
        return model.model_dump_json(*args, **kwargs)
    else:
        return model.json(*args, **kwargs)


def jsonable_encoder(
    obj: Any,
    include: Union[SetIntStr, DictIntStrAny] = None,
    exclude=None,
    by_alias: bool = True,
    skip_defaults: bool = None,
    exclude_unset: bool = True,
    exclude_none: bool = True,
):
    if hasattr(obj, "json") or hasattr(obj, "model_dump_json"):
        return to_json(
            obj,
            include=include,
            exclude=exclude,
            by_alias=by_alias,
            exclude_unset=bool(exclude_unset or skip_defaults),
            exclude_none=exclude_none,
        )

    return obj


if TYPE_CHECKING:
    from qdrant_client.http.api_client import ApiClient


class _SearchApi:
    def __init__(self, api_client: "Union[ApiClient, AsyncApiClient]"):
        self.api_client = api_client

    def _build_for_query_batch_points(
        self,
        collection_name: str,
        consistency: m.ReadConsistency = None,
        timeout: int = None,
        query_request_batch: m.QueryRequestBatch = None,
    ):
        """
        Universally query points in batch. This endpoint covers all capabilities of search, recommend, discover, filters. But also enables hybrid and multi-stage queries.
        """
        path_params = {
            "collection_name": str(collection_name),
        }

        query_params = {}
        if consistency is not None:
            query_params["consistency"] = str(consistency)
        if timeout is not None:
            query_params["timeout"] = str(timeout)

        headers = {}
        body = jsonable_encoder(query_request_batch)
        if "Content-Type" not in headers:
            headers["Content-Type"] = "application/json"
        return self.api_client.request(
            type_=m.InlineResponse20023,
            method="POST",
            url="/collections/{collection_name}/points/query/batch",
            headers=headers if headers else None,
            path_params=path_params,
            params=query_params,
            content=body,
        )

    def _build_for_query_points(
        self,
        collection_name: str,
        consistency: m.ReadConsistency = None,
        timeout: int = None,
        query_request: m.QueryRequest = None,
    ):
        """
        Universally query points. This endpoint covers all capabilities of search, recommend, discover, filters. But also enables hybrid and multi-stage queries.
        """
        path_params = {
            "collection_name": str(collection_name),
        }

        query_params = {}
        if consistency is not None:
            query_params["consistency"] = str(consistency)
        if timeout is not None:
            query_params["timeout"] = str(timeout)

        headers = {}
        body = jsonable_encoder(query_request)
        if "Content-Type" not in headers:
            headers["Content-Type"] = "application/json"
        return self.api_client.request(
            type_=m.InlineResponse20022,
            method="POST",
            url="/collections/{collection_name}/points/query",
            headers=headers if headers else None,
            path_params=path_params,
            params=query_params,
            content=body,
        )

    def _build_for_query_points_groups(
        self,
        collection_name: str,
        consistency: m.ReadConsistency = None,
        timeout: int = None,
        query_groups_request: m.QueryGroupsRequest = None,
    ):
        """
        Universally query points, grouped by a given payload field
        """
        path_params = {
            "collection_name": str(collection_name),
        }

        query_params = {}
        if consistency is not None:
            query_params["consistency"] = str(consistency)
        if timeout is not None:
            query_params["timeout"] = str(timeout)

        headers = {}
        body = jsonable_encoder(query_groups_request)
        if "Content-Type" not in headers:
            headers["Content-Type"] = "application/json"
        return self.api_client.request(
            type_=m.InlineResponse20024,
            method="POST",
            url="/collections/{collection_name}/points/query/groups",
            headers=headers if headers else None,
            path_params=path_params,
            params=query_params,
            content=body,
        )

    def _build_for_search_matrix_offsets(
        self,
        collection_name: str,
        consistency: m.ReadConsistency = None,
        timeout: int = None,
        search_matrix_request: m.SearchMatrixRequest = None,
    ):
        """
        Compute distance matrix for sampled points with an offset based output format
        """
        path_params = {
            "collection_name": str(collection_name),
        }

        query_params = {}
        if consistency is not None:
            query_params["consistency"] = str(consistency)
        if timeout is not None:
            query_params["timeout"] = str(timeout)

        headers = {}
        body = jsonable_encoder(search_matrix_request)
        if "Content-Type" not in headers:
            headers["Content-Type"] = "application/json"
        return self.api_client.request(
            type_=m.InlineResponse20026,
            method="POST",
            url="/collections/{collection_name}/points/search/matrix/offsets",
            headers=headers if headers else None,
            path_params=path_params,
            params=query_params,
            content=body,
        )

    def _build_for_search_matrix_pairs(
        self,
        collection_name: str,
        consistency: m.ReadConsistency = None,
        timeout: int = None,
        search_matrix_request: m.SearchMatrixRequest = None,
    ):
        """
        Compute distance matrix for sampled points with a pair based output format
        """
        path_params = {
            "collection_name": str(collection_name),
        }

        query_params = {}
        if consistency is not None:
            query_params["consistency"] = str(consistency)
        if timeout is not None:
            query_params["timeout"] = str(timeout)

        headers = {}
        body = jsonable_encoder(search_matrix_request)
        if "Content-Type" not in headers:
            headers["Content-Type"] = "application/json"
        return self.api_client.request(
            type_=m.InlineResponse20025,
            method="POST",
            url="/collections/{collection_name}/points/search/matrix/pairs",
            headers=headers if headers else None,
            path_params=path_params,
            params=query_params,
            content=body,
        )


class AsyncSearchApi(_SearchApi):
    async def query_batch_points(
        self,
        collection_name: str,
        consistency: m.ReadConsistency = None,
        timeout: int = None,
        query_request_batch: m.QueryRequestBatch = None,
    ) -> m.InlineResponse20023:
        """
        Universally query points in batch. This endpoint covers all capabilities of search, recommend, discover, filters. But also enables hybrid and multi-stage queries.
        """
        return await self._build_for_query_batch_points(
            collection_name=collection_name,
            consistency=consistency,
            timeout=timeout,
            query_request_batch=query_request_batch,
        )

    async def query_points(
        self,
        collection_name: str,
        consistency: m.ReadConsistency = None,
        timeout: int = None,
        query_request: m.QueryRequest = None,
    ) -> m.InlineResponse20022:
        """
        Universally query points. This endpoint covers all capabilities of search, recommend, discover, filters. But also enables hybrid and multi-stage queries.
        """
        return await self._build_for_query_points(
            collection_name=collection_name,
            consistency=consistency,
            timeout=timeout,
            query_request=query_request,
        )

    async def query_points_groups(
        self,
        collection_name: str,
        consistency: m.ReadConsistency = None,
        timeout: int = None,
        query_groups_request: m.QueryGroupsRequest = None,
    ) -> m.InlineResponse20024:
        """
        Universally query points, grouped by a given payload field
        """
        return await self._build_for_query_points_groups(
            collection_name=collection_name,
            consistency=consistency,
            timeout=timeout,
            query_groups_request=query_groups_request,
        )

    async def search_matrix_offsets(
        self,
        collection_name: str,
        consistency: m.ReadConsistency = None,
        timeout: int = None,
        search_matrix_request: m.SearchMatrixRequest = None,
    ) -> m.InlineResponse20026:
        """
        Compute distance matrix for sampled points with an offset based output format
        """
        return await self._build_for_search_matrix_offsets(
            collection_name=collection_name,
            consistency=consistency,
            timeout=timeout,
            search_matrix_request=search_matrix_request,
        )

    async def search_matrix_pairs(
        self,
        collection_name: str,
        consistency: m.ReadConsistency = None,
        timeout: int = None,
        search_matrix_request: m.SearchMatrixRequest = None,
    ) -> m.InlineResponse20025:
        """
        Compute distance matrix for sampled points with a pair based output format
        """
        return await self._build_for_search_matrix_pairs(
            collection_name=collection_name,
            consistency=consistency,
            timeout=timeout,
            search_matrix_request=search_matrix_request,
        )


class SyncSearchApi(_SearchApi):
    def query_batch_points(
        self,
        collection_name: str,
        consistency: m.ReadConsistency = None,
        timeout: int = None,
        query_request_batch: m.QueryRequestBatch = None,
    ) -> m.InlineResponse20023:
        """
        Universally query points in batch. This endpoint covers all capabilities of search, recommend, discover, filters. But also enables hybrid and multi-stage queries.
        """
        return self._build_for_query_batch_points(
            collection_name=collection_name,
            consistency=consistency,
            timeout=timeout,
            query_request_batch=query_request_batch,
        )

    def query_points(
        self,
        collection_name: str,
        consistency: m.ReadConsistency = None,
        timeout: int = None,
        query_request: m.QueryRequest = None,
    ) -> m.InlineResponse20022:
        """
        Universally query points. This endpoint covers all capabilities of search, recommend, discover, filters. But also enables hybrid and multi-stage queries.
        """
        return self._build_for_query_points(
            collection_name=collection_name,
            consistency=consistency,
            timeout=timeout,
            query_request=query_request,
        )

    def query_points_groups(
        self,
        collection_name: str,
        consistency: m.ReadConsistency = None,
        timeout: int = None,
        query_groups_request: m.QueryGroupsRequest = None,
    ) -> m.InlineResponse20024:
        """
        Universally query points, grouped by a given payload field
        """
        return self._build_for_query_points_groups(
            collection_name=collection_name,
            consistency=consistency,
            timeout=timeout,
            query_groups_request=query_groups_request,
        )

    def search_matrix_offsets(
        self,
        collection_name: str,
        consistency: m.ReadConsistency = None,
        timeout: int = None,
        search_matrix_request: m.SearchMatrixRequest = None,
    ) -> m.InlineResponse20026:
        """
        Compute distance matrix for sampled points with an offset based output format
        """
        return self._build_for_search_matrix_offsets(
            collection_name=collection_name,
            consistency=consistency,
            timeout=timeout,
            search_matrix_request=search_matrix_request,
        )

    def search_matrix_pairs(
        self,
        collection_name: str,
        consistency: m.ReadConsistency = None,
        timeout: int = None,
        search_matrix_request: m.SearchMatrixRequest = None,
    ) -> m.InlineResponse20025:
        """
        Compute distance matrix for sampled points with a pair based output format
        """
        return self._build_for_search_matrix_pairs(
            collection_name=collection_name,
            consistency=consistency,
            timeout=timeout,
            search_matrix_request=search_matrix_request,
        )
