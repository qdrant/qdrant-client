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


class _PointsApi:
    def __init__(self, api_client: "Union[ApiClient, AsyncApiClient]"):
        self.api_client = api_client

    def _build_for_facet(
        self,
        collection_name: str,
        timeout: int = None,
        consistency: m.ReadConsistency = None,
        facet_request: m.FacetRequest = None,
    ):
        """
        Count points that satisfy the given filter for each unique value of a payload key.
        """
        path_params = {
            "collection_name": str(collection_name),
        }

        query_params = {}
        if timeout is not None:
            query_params["timeout"] = str(timeout)
        if consistency is not None:
            query_params["consistency"] = str(consistency)

        headers = {}
        body = jsonable_encoder(facet_request)
        if "Content-Type" not in headers:
            headers["Content-Type"] = "application/json"
        return self.api_client.request(
            type_=m.InlineResponse20020,
            method="POST",
            url="/collections/{collection_name}/facet",
            headers=headers if headers else None,
            path_params=path_params,
            params=query_params,
            content=body,
        )

    def _build_for_search_points_matrix_offsets(
        self,
        collection_name: str,
        consistency: m.ReadConsistency = None,
        timeout: int = None,
        search_groups_request: m.SearchGroupsRequest = None,
    ):
        """
        Retrieve closest points based on vector similarity and given filtering conditions, grouped by a given payload field
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
        body = jsonable_encoder(search_groups_request)
        if "Content-Type" not in headers:
            headers["Content-Type"] = "application/json"
        return self.api_client.request(
            type_=m.InlineResponse20018,
            method="POST",
            url="/collections/{collection_name}/points/search/groups",
            headers=headers if headers else None,
            path_params=path_params,
            params=query_params,
            content=body,
        )

    def _build_for_search_points(
        self,
        collection_name: str,
        consistency: m.ReadConsistency = None,
        timeout: int = None,
        search_request: m.SearchRequest = None,
    ):
        """
        Retrieve closest points based on vector similarity and given filtering conditions
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
        body = jsonable_encoder(search_request)
        if "Content-Type" not in headers:
            headers["Content-Type"] = "application/json"
        return self.api_client.request(
            type_=m.InlineResponse20016,
            method="POST",
            url="/collections/{collection_name}/points/search",
            headers=headers if headers else None,
            path_params=path_params,
            params=query_params,
            content=body,
        )


class AsyncPointsApi(_PointsApi):
    async def facet(
        self,
        collection_name: str,
        timeout: int = None,
        consistency: m.ReadConsistency = None,
        facet_request: m.FacetRequest = None,
    ) -> m.InlineResponse20020:
        """
        Count points that satisfy the given filter for each unique value of a payload key.
        """
        return await self._build_for_facet(
            collection_name=collection_name,
            timeout=timeout,
            consistency=consistency,
            facet_request=facet_request,
        )

    async def search_points_matrix_offsets(
        self,
        collection_name: str,
        consistency: m.ReadConsistency = None,
        timeout: int = None,
        search_groups_request: m.SearchGroupsRequest = None,
    ) -> m.InlineResponse20018:
        """
        Retrieve closest points based on vector similarity and given filtering conditions, grouped by a given payload field
        """
        return await self._build_for_search_point_groups(
            collection_name=collection_name,
            consistency=consistency,
            timeout=timeout,
            search_groups_request=search_groups_request,
        )

    async def search_points(
        self,
        collection_name: str,
        consistency: m.ReadConsistency = None,
        timeout: int = None,
        search_request: m.SearchRequest = None,
    ) -> m.InlineResponse20016:
        """
        Retrieve closest points based on vector similarity and given filtering conditions
        """
        return await self._build_for_search_points(
            collection_name=collection_name,
            consistency=consistency,
            timeout=timeout,
            search_request=search_request,
        )


class SyncPointsApi(_PointsApi):
    def facet(
        self,
        collection_name: str,
        timeout: int = None,
        consistency: m.ReadConsistency = None,
        facet_request: m.FacetRequest = None,
    ) -> m.InlineResponse20020:
        """
        Count points that satisfy the given filter for each unique value of a payload key.
        """
        return self._build_for_facet(
            collection_name=collection_name,
            timeout=timeout,
            consistency=consistency,
            facet_request=facet_request,
        )

    def search_points_matrix_offsets(
        self,
        collection_name: str,
        consistency: m.ReadConsistency = None,
        timeout: int = None,
        search_groups_request: m.SearchGroupsRequest = None,
    ) -> m.InlineResponse20018:
        """
        Retrieve closest points based on vector similarity and given filtering conditions, grouped by a given payload field
        """
        return self._build_for_search_point_groups(
            collection_name=collection_name,
            consistency=consistency,
            timeout=timeout,
            search_groups_request=search_groups_request,
        )

    def search_points(
        self,
        collection_name: str,
        consistency: m.ReadConsistency = None,
        timeout: int = None,
        search_request: m.SearchRequest = None,
    ) -> m.InlineResponse20016:
        """
        Retrieve closest points based on vector similarity and given filtering conditions
        """
        return self._build_for_search_points(
            collection_name=collection_name,
            consistency=consistency,
            timeout=timeout,
            search_request=search_request,
        )
