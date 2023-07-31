# flake8: noqa E501
from typing import TYPE_CHECKING, Any, Dict, Set, Union

from qdrant_client._pydantic_compat import to_json
from qdrant_client.http.models import *
from qdrant_client.http.models import models as m

SetIntStr = Set[Union[int, str]]
DictIntStrAny = Dict[Union[int, str], Any]
file = None


def jsonable_encoder(
    obj: Any,
    include: Union[SetIntStr, DictIntStrAny] = None,
    exclude=None,
    by_alias: bool = True,
    skip_defaults: bool = None,
    exclude_unset: bool = False,
):
    if hasattr(obj, "json") or hasattr(obj, "model_dump_json"):
        return to_json(
            obj,
            include=include,
            exclude=exclude,
            by_alias=by_alias,
            exclude_unset=bool(exclude_unset or skip_defaults),
        )

    return obj


if TYPE_CHECKING:
    from qdrant_client.http.api_client import ApiClient


class _PointsApi:
    def __init__(self, api_client: "Union[ApiClient, AsyncApiClient]"):
        self.api_client = api_client

    def _build_for_clear_payload(
        self,
        collection_name: str,
        wait: bool = None,
        ordering: WriteOrdering = None,
        points_selector: m.PointsSelector = None,
    ):
        """
        Remove all payload for specified points
        """
        path_params = {
            "collection_name": str(collection_name),
        }

        query_params = {}
        if wait is not None:
            query_params["wait"] = str(wait).lower()
        if ordering is not None:
            query_params["ordering"] = str(ordering)

        headers = {}
        body = jsonable_encoder(points_selector)
        if "Content-Type" not in headers:
            headers["Content-Type"] = "application/json"
        return self.api_client.request(
            type_=m.InlineResponse2006,
            method="POST",
            url="/collections/{collection_name}/points/payload/clear",
            headers=headers if headers else None,
            path_params=path_params,
            params=query_params,
            data=body,
        )

    def _build_for_count_points(
        self,
        collection_name: str,
        count_request: m.CountRequest = None,
    ):
        """
        Count points which matches given filtering condition
        """
        path_params = {
            "collection_name": str(collection_name),
        }

        headers = {}
        body = jsonable_encoder(count_request)
        if "Content-Type" not in headers:
            headers["Content-Type"] = "application/json"
        return self.api_client.request(
            type_=m.InlineResponse20017,
            method="POST",
            url="/collections/{collection_name}/points/count",
            headers=headers if headers else None,
            path_params=path_params,
            data=body,
        )

    def _build_for_delete_payload(
        self,
        collection_name: str,
        wait: bool = None,
        ordering: WriteOrdering = None,
        delete_payload: m.DeletePayload = None,
    ):
        """
        Delete specified key payload for points
        """
        path_params = {
            "collection_name": str(collection_name),
        }

        query_params = {}
        if wait is not None:
            query_params["wait"] = str(wait).lower()
        if ordering is not None:
            query_params["ordering"] = str(ordering)

        headers = {}
        body = jsonable_encoder(delete_payload)
        if "Content-Type" not in headers:
            headers["Content-Type"] = "application/json"
        return self.api_client.request(
            type_=m.InlineResponse2006,
            method="POST",
            url="/collections/{collection_name}/points/payload/delete",
            headers=headers if headers else None,
            path_params=path_params,
            params=query_params,
            data=body,
        )

    def _build_for_delete_points(
        self,
        collection_name: str,
        wait: bool = None,
        ordering: WriteOrdering = None,
        points_selector: m.PointsSelector = None,
    ):
        """
        Delete points
        """
        path_params = {
            "collection_name": str(collection_name),
        }

        query_params = {}
        if wait is not None:
            query_params["wait"] = str(wait).lower()
        if ordering is not None:
            query_params["ordering"] = str(ordering)

        headers = {}
        body = jsonable_encoder(points_selector)
        if "Content-Type" not in headers:
            headers["Content-Type"] = "application/json"
        return self.api_client.request(
            type_=m.InlineResponse2006,
            method="POST",
            url="/collections/{collection_name}/points/delete",
            headers=headers if headers else None,
            path_params=path_params,
            params=query_params,
            data=body,
        )

    def _build_for_delete_vectors(
        self,
        collection_name: str,
        wait: bool = None,
        ordering: WriteOrdering = None,
        delete_vectors: m.DeleteVectors = None,
    ):
        """
        Delete named vectors from the given points.
        """
        path_params = {
            "collection_name": str(collection_name),
        }

        query_params = {}
        if wait is not None:
            query_params["wait"] = str(wait).lower()
        if ordering is not None:
            query_params["ordering"] = str(ordering)

        headers = {}
        body = jsonable_encoder(delete_vectors)
        if "Content-Type" not in headers:
            headers["Content-Type"] = "application/json"
        return self.api_client.request(
            type_=m.InlineResponse2006,
            method="POST",
            url="/collections/{collection_name}/points/vectors/delete",
            headers=headers if headers else None,
            path_params=path_params,
            params=query_params,
            data=body,
        )

    def _build_for_get_point(
        self,
        collection_name: str,
        id: m.ExtendedPointId,
        consistency: m.ReadConsistency = None,
    ):
        """
        Retrieve full information of single point by id
        """
        path_params = {
            "collection_name": str(collection_name),
            "id": str(id),
        }

        query_params = {}
        if consistency is not None:
            query_params["consistency"] = str(consistency)

        headers = {}
        return self.api_client.request(
            type_=m.InlineResponse20011,
            method="GET",
            url="/collections/{collection_name}/points/{id}",
            headers=headers if headers else None,
            path_params=path_params,
            params=query_params,
        )

    def _build_for_get_points(
        self,
        collection_name: str,
        consistency: m.ReadConsistency = None,
        point_request: m.PointRequest = None,
    ):
        """
        Retrieve multiple points by specified IDs
        """
        path_params = {
            "collection_name": str(collection_name),
        }

        query_params = {}
        if consistency is not None:
            query_params["consistency"] = str(consistency)

        headers = {}
        body = jsonable_encoder(point_request)
        if "Content-Type" not in headers:
            headers["Content-Type"] = "application/json"
        return self.api_client.request(
            type_=m.InlineResponse20012,
            method="POST",
            url="/collections/{collection_name}/points",
            headers=headers if headers else None,
            path_params=path_params,
            params=query_params,
            data=body,
        )

    def _build_for_overwrite_payload(
        self,
        collection_name: str,
        wait: bool = None,
        ordering: WriteOrdering = None,
        set_payload: m.SetPayload = None,
    ):
        """
        Replace full payload of points with new one
        """
        path_params = {
            "collection_name": str(collection_name),
        }

        query_params = {}
        if wait is not None:
            query_params["wait"] = str(wait).lower()
        if ordering is not None:
            query_params["ordering"] = str(ordering)

        headers = {}
        body = jsonable_encoder(set_payload)
        if "Content-Type" not in headers:
            headers["Content-Type"] = "application/json"
        return self.api_client.request(
            type_=m.InlineResponse2006,
            method="PUT",
            url="/collections/{collection_name}/points/payload",
            headers=headers if headers else None,
            path_params=path_params,
            params=query_params,
            data=body,
        )

    def _build_for_recommend_batch_points(
        self,
        collection_name: str,
        consistency: m.ReadConsistency = None,
        recommend_request_batch: m.RecommendRequestBatch = None,
    ):
        """
        Look for the points which are closer to stored positive examples and at the same time further to negative examples.
        """
        path_params = {
            "collection_name": str(collection_name),
        }

        query_params = {}
        if consistency is not None:
            query_params["consistency"] = str(consistency)

        headers = {}
        body = jsonable_encoder(recommend_request_batch)
        if "Content-Type" not in headers:
            headers["Content-Type"] = "application/json"
        return self.api_client.request(
            type_=m.InlineResponse20015,
            method="POST",
            url="/collections/{collection_name}/points/recommend/batch",
            headers=headers if headers else None,
            path_params=path_params,
            params=query_params,
            data=body,
        )

    def _build_for_recommend_point_groups(
        self,
        collection_name: str,
        consistency: m.ReadConsistency = None,
        recommend_groups_request: m.RecommendGroupsRequest = None,
    ):
        """
        Look for the points which are closer to stored positive examples and at the same time further to negative examples, grouped by a given payload field.
        """
        path_params = {
            "collection_name": str(collection_name),
        }

        query_params = {}
        if consistency is not None:
            query_params["consistency"] = str(consistency)

        headers = {}
        body = jsonable_encoder(recommend_groups_request)
        if "Content-Type" not in headers:
            headers["Content-Type"] = "application/json"
        return self.api_client.request(
            type_=m.InlineResponse20016,
            method="POST",
            url="/collections/{collection_name}/points/recommend/groups",
            headers=headers if headers else None,
            path_params=path_params,
            params=query_params,
            data=body,
        )

    def _build_for_recommend_points(
        self,
        collection_name: str,
        consistency: m.ReadConsistency = None,
        recommend_request: m.RecommendRequest = None,
    ):
        """
        Look for the points which are closer to stored positive examples and at the same time further to negative examples.
        """
        path_params = {
            "collection_name": str(collection_name),
        }

        query_params = {}
        if consistency is not None:
            query_params["consistency"] = str(consistency)

        headers = {}
        body = jsonable_encoder(recommend_request)
        if "Content-Type" not in headers:
            headers["Content-Type"] = "application/json"
        return self.api_client.request(
            type_=m.InlineResponse20014,
            method="POST",
            url="/collections/{collection_name}/points/recommend",
            headers=headers if headers else None,
            path_params=path_params,
            params=query_params,
            data=body,
        )

    def _build_for_scroll_points(
        self,
        collection_name: str,
        consistency: m.ReadConsistency = None,
        scroll_request: m.ScrollRequest = None,
    ):
        """
        Scroll request - paginate over all points which matches given filtering condition
        """
        path_params = {
            "collection_name": str(collection_name),
        }

        query_params = {}
        if consistency is not None:
            query_params["consistency"] = str(consistency)

        headers = {}
        body = jsonable_encoder(scroll_request)
        if "Content-Type" not in headers:
            headers["Content-Type"] = "application/json"
        return self.api_client.request(
            type_=m.InlineResponse20013,
            method="POST",
            url="/collections/{collection_name}/points/scroll",
            headers=headers if headers else None,
            path_params=path_params,
            params=query_params,
            data=body,
        )

    def _build_for_search_batch_points(
        self,
        collection_name: str,
        consistency: m.ReadConsistency = None,
        search_request_batch: m.SearchRequestBatch = None,
    ):
        """
        Retrieve by batch the closest points based on vector similarity and given filtering conditions
        """
        path_params = {
            "collection_name": str(collection_name),
        }

        query_params = {}
        if consistency is not None:
            query_params["consistency"] = str(consistency)

        headers = {}
        body = jsonable_encoder(search_request_batch)
        if "Content-Type" not in headers:
            headers["Content-Type"] = "application/json"
        return self.api_client.request(
            type_=m.InlineResponse20015,
            method="POST",
            url="/collections/{collection_name}/points/search/batch",
            headers=headers if headers else None,
            path_params=path_params,
            params=query_params,
            data=body,
        )

    def _build_for_search_point_groups(
        self,
        collection_name: str,
        consistency: m.ReadConsistency = None,
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

        headers = {}
        body = jsonable_encoder(search_groups_request)
        if "Content-Type" not in headers:
            headers["Content-Type"] = "application/json"
        return self.api_client.request(
            type_=m.InlineResponse20016,
            method="POST",
            url="/collections/{collection_name}/points/search/groups",
            headers=headers if headers else None,
            path_params=path_params,
            params=query_params,
            data=body,
        )

    def _build_for_search_points(
        self,
        collection_name: str,
        consistency: m.ReadConsistency = None,
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

        headers = {}
        body = jsonable_encoder(search_request)
        if "Content-Type" not in headers:
            headers["Content-Type"] = "application/json"
        return self.api_client.request(
            type_=m.InlineResponse20014,
            method="POST",
            url="/collections/{collection_name}/points/search",
            headers=headers if headers else None,
            path_params=path_params,
            params=query_params,
            data=body,
        )

    def _build_for_set_payload(
        self,
        collection_name: str,
        wait: bool = None,
        ordering: WriteOrdering = None,
        set_payload: m.SetPayload = None,
    ):
        """
        Set payload values for points
        """
        path_params = {
            "collection_name": str(collection_name),
        }

        query_params = {}
        if wait is not None:
            query_params["wait"] = str(wait).lower()
        if ordering is not None:
            query_params["ordering"] = str(ordering)

        headers = {}
        body = jsonable_encoder(set_payload)
        if "Content-Type" not in headers:
            headers["Content-Type"] = "application/json"
        return self.api_client.request(
            type_=m.InlineResponse2006,
            method="POST",
            url="/collections/{collection_name}/points/payload",
            headers=headers if headers else None,
            path_params=path_params,
            params=query_params,
            data=body,
        )

    def _build_for_update_vectors(
        self,
        collection_name: str,
        wait: bool = None,
        ordering: WriteOrdering = None,
        update_vectors: m.UpdateVectors = None,
    ):
        """
        Update specified named vectors on points, keep unspecified vectors intact.
        """
        path_params = {
            "collection_name": str(collection_name),
        }

        query_params = {}
        if wait is not None:
            query_params["wait"] = str(wait).lower()
        if ordering is not None:
            query_params["ordering"] = str(ordering)

        headers = {}
        body = jsonable_encoder(update_vectors)
        if "Content-Type" not in headers:
            headers["Content-Type"] = "application/json"
        return self.api_client.request(
            type_=m.InlineResponse2006,
            method="PUT",
            url="/collections/{collection_name}/points/vectors",
            headers=headers if headers else None,
            path_params=path_params,
            params=query_params,
            data=body,
        )

    def _build_for_upsert_points(
        self,
        collection_name: str,
        wait: bool = None,
        ordering: WriteOrdering = None,
        point_insert_operations: m.PointInsertOperations = None,
    ):
        """
        Perform insert + updates on points. If point with given ID already exists - it will be overwritten.
        """
        path_params = {
            "collection_name": str(collection_name),
        }

        query_params = {}
        if wait is not None:
            query_params["wait"] = str(wait).lower()
        if ordering is not None:
            query_params["ordering"] = str(ordering)

        headers = {}
        body = jsonable_encoder(point_insert_operations)
        if "Content-Type" not in headers:
            headers["Content-Type"] = "application/json"
        return self.api_client.request(
            type_=m.InlineResponse2006,
            method="PUT",
            url="/collections/{collection_name}/points",
            headers=headers if headers else None,
            path_params=path_params,
            params=query_params,
            data=body,
        )


class AsyncPointsApi(_PointsApi):
    async def clear_payload(
        self,
        collection_name: str,
        wait: bool = None,
        ordering: WriteOrdering = None,
        points_selector: m.PointsSelector = None,
    ) -> m.InlineResponse2006:
        """
        Remove all payload for specified points
        """
        return await self._build_for_clear_payload(
            collection_name=collection_name,
            wait=wait,
            ordering=ordering,
            points_selector=points_selector,
        )

    async def count_points(
        self,
        collection_name: str,
        count_request: m.CountRequest = None,
    ) -> m.InlineResponse20017:
        """
        Count points which matches given filtering condition
        """
        return await self._build_for_count_points(
            collection_name=collection_name,
            count_request=count_request,
        )

    async def delete_payload(
        self,
        collection_name: str,
        wait: bool = None,
        ordering: WriteOrdering = None,
        delete_payload: m.DeletePayload = None,
    ) -> m.InlineResponse2006:
        """
        Delete specified key payload for points
        """
        return await self._build_for_delete_payload(
            collection_name=collection_name,
            wait=wait,
            ordering=ordering,
            delete_payload=delete_payload,
        )

    async def delete_points(
        self,
        collection_name: str,
        wait: bool = None,
        ordering: WriteOrdering = None,
        points_selector: m.PointsSelector = None,
    ) -> m.InlineResponse2006:
        """
        Delete points
        """
        return await self._build_for_delete_points(
            collection_name=collection_name,
            wait=wait,
            ordering=ordering,
            points_selector=points_selector,
        )

    async def delete_vectors(
        self,
        collection_name: str,
        wait: bool = None,
        ordering: WriteOrdering = None,
        delete_vectors: m.DeleteVectors = None,
    ) -> m.InlineResponse2006:
        """
        Delete named vectors from the given points.
        """
        return await self._build_for_delete_vectors(
            collection_name=collection_name,
            wait=wait,
            ordering=ordering,
            delete_vectors=delete_vectors,
        )

    async def get_point(
        self,
        collection_name: str,
        id: m.ExtendedPointId,
        consistency: m.ReadConsistency = None,
    ) -> m.InlineResponse20011:
        """
        Retrieve full information of single point by id
        """
        return await self._build_for_get_point(
            collection_name=collection_name,
            id=id,
            consistency=consistency,
        )

    async def get_points(
        self,
        collection_name: str,
        consistency: m.ReadConsistency = None,
        point_request: m.PointRequest = None,
    ) -> m.InlineResponse20012:
        """
        Retrieve multiple points by specified IDs
        """
        return await self._build_for_get_points(
            collection_name=collection_name,
            consistency=consistency,
            point_request=point_request,
        )

    async def overwrite_payload(
        self,
        collection_name: str,
        wait: bool = None,
        ordering: WriteOrdering = None,
        set_payload: m.SetPayload = None,
    ) -> m.InlineResponse2006:
        """
        Replace full payload of points with new one
        """
        return await self._build_for_overwrite_payload(
            collection_name=collection_name,
            wait=wait,
            ordering=ordering,
            set_payload=set_payload,
        )

    async def recommend_batch_points(
        self,
        collection_name: str,
        consistency: m.ReadConsistency = None,
        recommend_request_batch: m.RecommendRequestBatch = None,
    ) -> m.InlineResponse20015:
        """
        Look for the points which are closer to stored positive examples and at the same time further to negative examples.
        """
        return await self._build_for_recommend_batch_points(
            collection_name=collection_name,
            consistency=consistency,
            recommend_request_batch=recommend_request_batch,
        )

    async def recommend_point_groups(
        self,
        collection_name: str,
        consistency: m.ReadConsistency = None,
        recommend_groups_request: m.RecommendGroupsRequest = None,
    ) -> m.InlineResponse20016:
        """
        Look for the points which are closer to stored positive examples and at the same time further to negative examples, grouped by a given payload field.
        """
        return await self._build_for_recommend_point_groups(
            collection_name=collection_name,
            consistency=consistency,
            recommend_groups_request=recommend_groups_request,
        )

    async def recommend_points(
        self,
        collection_name: str,
        consistency: m.ReadConsistency = None,
        recommend_request: m.RecommendRequest = None,
    ) -> m.InlineResponse20014:
        """
        Look for the points which are closer to stored positive examples and at the same time further to negative examples.
        """
        return await self._build_for_recommend_points(
            collection_name=collection_name,
            consistency=consistency,
            recommend_request=recommend_request,
        )

    async def scroll_points(
        self,
        collection_name: str,
        consistency: m.ReadConsistency = None,
        scroll_request: m.ScrollRequest = None,
    ) -> m.InlineResponse20013:
        """
        Scroll request - paginate over all points which matches given filtering condition
        """
        return await self._build_for_scroll_points(
            collection_name=collection_name,
            consistency=consistency,
            scroll_request=scroll_request,
        )

    async def search_batch_points(
        self,
        collection_name: str,
        consistency: m.ReadConsistency = None,
        search_request_batch: m.SearchRequestBatch = None,
    ) -> m.InlineResponse20015:
        """
        Retrieve by batch the closest points based on vector similarity and given filtering conditions
        """
        return await self._build_for_search_batch_points(
            collection_name=collection_name,
            consistency=consistency,
            search_request_batch=search_request_batch,
        )

    async def search_point_groups(
        self,
        collection_name: str,
        consistency: m.ReadConsistency = None,
        search_groups_request: m.SearchGroupsRequest = None,
    ) -> m.InlineResponse20016:
        """
        Retrieve closest points based on vector similarity and given filtering conditions, grouped by a given payload field
        """
        return await self._build_for_search_point_groups(
            collection_name=collection_name,
            consistency=consistency,
            search_groups_request=search_groups_request,
        )

    async def search_points(
        self,
        collection_name: str,
        consistency: m.ReadConsistency = None,
        search_request: m.SearchRequest = None,
    ) -> m.InlineResponse20014:
        """
        Retrieve closest points based on vector similarity and given filtering conditions
        """
        return await self._build_for_search_points(
            collection_name=collection_name,
            consistency=consistency,
            search_request=search_request,
        )

    async def set_payload(
        self,
        collection_name: str,
        wait: bool = None,
        ordering: WriteOrdering = None,
        set_payload: m.SetPayload = None,
    ) -> m.InlineResponse2006:
        """
        Set payload values for points
        """
        return await self._build_for_set_payload(
            collection_name=collection_name,
            wait=wait,
            ordering=ordering,
            set_payload=set_payload,
        )

    async def update_vectors(
        self,
        collection_name: str,
        wait: bool = None,
        ordering: WriteOrdering = None,
        update_vectors: m.UpdateVectors = None,
    ) -> m.InlineResponse2006:
        """
        Update specified named vectors on points, keep unspecified vectors intact.
        """
        return await self._build_for_update_vectors(
            collection_name=collection_name,
            wait=wait,
            ordering=ordering,
            update_vectors=update_vectors,
        )

    async def upsert_points(
        self,
        collection_name: str,
        wait: bool = None,
        ordering: WriteOrdering = None,
        point_insert_operations: m.PointInsertOperations = None,
    ) -> m.InlineResponse2006:
        """
        Perform insert + updates on points. If point with given ID already exists - it will be overwritten.
        """
        return await self._build_for_upsert_points(
            collection_name=collection_name,
            wait=wait,
            ordering=ordering,
            point_insert_operations=point_insert_operations,
        )


class SyncPointsApi(_PointsApi):
    def clear_payload(
        self,
        collection_name: str,
        wait: bool = None,
        ordering: WriteOrdering = None,
        points_selector: m.PointsSelector = None,
    ) -> m.InlineResponse2006:
        """
        Remove all payload for specified points
        """
        return self._build_for_clear_payload(
            collection_name=collection_name,
            wait=wait,
            ordering=ordering,
            points_selector=points_selector,
        )

    def count_points(
        self,
        collection_name: str,
        count_request: m.CountRequest = None,
    ) -> m.InlineResponse20017:
        """
        Count points which matches given filtering condition
        """
        return self._build_for_count_points(
            collection_name=collection_name,
            count_request=count_request,
        )

    def delete_payload(
        self,
        collection_name: str,
        wait: bool = None,
        ordering: WriteOrdering = None,
        delete_payload: m.DeletePayload = None,
    ) -> m.InlineResponse2006:
        """
        Delete specified key payload for points
        """
        return self._build_for_delete_payload(
            collection_name=collection_name,
            wait=wait,
            ordering=ordering,
            delete_payload=delete_payload,
        )

    def delete_points(
        self,
        collection_name: str,
        wait: bool = None,
        ordering: WriteOrdering = None,
        points_selector: m.PointsSelector = None,
    ) -> m.InlineResponse2006:
        """
        Delete points
        """
        return self._build_for_delete_points(
            collection_name=collection_name,
            wait=wait,
            ordering=ordering,
            points_selector=points_selector,
        )

    def delete_vectors(
        self,
        collection_name: str,
        wait: bool = None,
        ordering: WriteOrdering = None,
        delete_vectors: m.DeleteVectors = None,
    ) -> m.InlineResponse2006:
        """
        Delete named vectors from the given points.
        """
        return self._build_for_delete_vectors(
            collection_name=collection_name,
            wait=wait,
            ordering=ordering,
            delete_vectors=delete_vectors,
        )

    def get_point(
        self,
        collection_name: str,
        id: m.ExtendedPointId,
        consistency: m.ReadConsistency = None,
    ) -> m.InlineResponse20011:
        """
        Retrieve full information of single point by id
        """
        return self._build_for_get_point(
            collection_name=collection_name,
            id=id,
            consistency=consistency,
        )

    def get_points(
        self,
        collection_name: str,
        consistency: m.ReadConsistency = None,
        point_request: m.PointRequest = None,
    ) -> m.InlineResponse20012:
        """
        Retrieve multiple points by specified IDs
        """
        return self._build_for_get_points(
            collection_name=collection_name,
            consistency=consistency,
            point_request=point_request,
        )

    def overwrite_payload(
        self,
        collection_name: str,
        wait: bool = None,
        ordering: WriteOrdering = None,
        set_payload: m.SetPayload = None,
    ) -> m.InlineResponse2006:
        """
        Replace full payload of points with new one
        """
        return self._build_for_overwrite_payload(
            collection_name=collection_name,
            wait=wait,
            ordering=ordering,
            set_payload=set_payload,
        )

    def recommend_batch_points(
        self,
        collection_name: str,
        consistency: m.ReadConsistency = None,
        recommend_request_batch: m.RecommendRequestBatch = None,
    ) -> m.InlineResponse20015:
        """
        Look for the points which are closer to stored positive examples and at the same time further to negative examples.
        """
        return self._build_for_recommend_batch_points(
            collection_name=collection_name,
            consistency=consistency,
            recommend_request_batch=recommend_request_batch,
        )

    def recommend_point_groups(
        self,
        collection_name: str,
        consistency: m.ReadConsistency = None,
        recommend_groups_request: m.RecommendGroupsRequest = None,
    ) -> m.InlineResponse20016:
        """
        Look for the points which are closer to stored positive examples and at the same time further to negative examples, grouped by a given payload field.
        """
        return self._build_for_recommend_point_groups(
            collection_name=collection_name,
            consistency=consistency,
            recommend_groups_request=recommend_groups_request,
        )

    def recommend_points(
        self,
        collection_name: str,
        consistency: m.ReadConsistency = None,
        recommend_request: m.RecommendRequest = None,
    ) -> m.InlineResponse20014:
        """
        Look for the points which are closer to stored positive examples and at the same time further to negative examples.
        """
        return self._build_for_recommend_points(
            collection_name=collection_name,
            consistency=consistency,
            recommend_request=recommend_request,
        )

    def scroll_points(
        self,
        collection_name: str,
        consistency: m.ReadConsistency = None,
        scroll_request: m.ScrollRequest = None,
    ) -> m.InlineResponse20013:
        """
        Scroll request - paginate over all points which matches given filtering condition
        """
        return self._build_for_scroll_points(
            collection_name=collection_name,
            consistency=consistency,
            scroll_request=scroll_request,
        )

    def search_batch_points(
        self,
        collection_name: str,
        consistency: m.ReadConsistency = None,
        search_request_batch: m.SearchRequestBatch = None,
    ) -> m.InlineResponse20015:
        """
        Retrieve by batch the closest points based on vector similarity and given filtering conditions
        """
        return self._build_for_search_batch_points(
            collection_name=collection_name,
            consistency=consistency,
            search_request_batch=search_request_batch,
        )

    def search_point_groups(
        self,
        collection_name: str,
        consistency: m.ReadConsistency = None,
        search_groups_request: m.SearchGroupsRequest = None,
    ) -> m.InlineResponse20016:
        """
        Retrieve closest points based on vector similarity and given filtering conditions, grouped by a given payload field
        """
        return self._build_for_search_point_groups(
            collection_name=collection_name,
            consistency=consistency,
            search_groups_request=search_groups_request,
        )

    def search_points(
        self,
        collection_name: str,
        consistency: m.ReadConsistency = None,
        search_request: m.SearchRequest = None,
    ) -> m.InlineResponse20014:
        """
        Retrieve closest points based on vector similarity and given filtering conditions
        """
        return self._build_for_search_points(
            collection_name=collection_name,
            consistency=consistency,
            search_request=search_request,
        )

    def set_payload(
        self,
        collection_name: str,
        wait: bool = None,
        ordering: WriteOrdering = None,
        set_payload: m.SetPayload = None,
    ) -> m.InlineResponse2006:
        """
        Set payload values for points
        """
        return self._build_for_set_payload(
            collection_name=collection_name,
            wait=wait,
            ordering=ordering,
            set_payload=set_payload,
        )

    def update_vectors(
        self,
        collection_name: str,
        wait: bool = None,
        ordering: WriteOrdering = None,
        update_vectors: m.UpdateVectors = None,
    ) -> m.InlineResponse2006:
        """
        Update specified named vectors on points, keep unspecified vectors intact.
        """
        return self._build_for_update_vectors(
            collection_name=collection_name,
            wait=wait,
            ordering=ordering,
            update_vectors=update_vectors,
        )

    def upsert_points(
        self,
        collection_name: str,
        wait: bool = None,
        ordering: WriteOrdering = None,
        point_insert_operations: m.PointInsertOperations = None,
    ) -> m.InlineResponse2006:
        """
        Perform insert + updates on points. If point with given ID already exists - it will be overwritten.
        """
        return self._build_for_upsert_points(
            collection_name=collection_name,
            wait=wait,
            ordering=ordering,
            point_insert_operations=point_insert_operations,
        )
