# flake8: noqa E501
from enum import Enum
from pathlib import PurePath
from types import GeneratorType
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Set, Tuple, Union

from pydantic.json import ENCODERS_BY_TYPE
from pydantic.main import BaseModel
from qdrant_client.http.models import models as m

SetIntStr = Set[Union[int, str]]
DictIntStrAny = Dict[Union[int, str], Any]


def generate_encoders_by_class_tuples(type_encoder_map: Dict[Any, Callable]) -> Dict[Callable, Tuple]:
    encoders_by_classes: Dict[Callable, List] = {}
    for type_, encoder in type_encoder_map.items():
        encoders_by_classes.setdefault(encoder, []).append(type_)
    encoders_by_class_tuples: Dict[Callable, Tuple] = {}
    for encoder, classes in encoders_by_classes.items():
        encoders_by_class_tuples[encoder] = tuple(classes)
    return encoders_by_class_tuples


encoders_by_class_tuples = generate_encoders_by_class_tuples(ENCODERS_BY_TYPE)


def jsonable_encoder(
    obj: Any,
    include: Union[SetIntStr, DictIntStrAny] = None,
    exclude=None,
    by_alias: bool = True,
    skip_defaults: bool = None,
    exclude_unset: bool = False,
    include_none: bool = True,
    custom_encoder=None,
    sqlalchemy_safe: bool = True,
) -> Any:
    if exclude is None:
        exclude = set()
    if custom_encoder is None:
        custom_encoder = {}
    if include is not None and not isinstance(include, set):
        include = set(include)
    if exclude is not None and not isinstance(exclude, set):
        exclude = set(exclude)
    if isinstance(obj, BaseModel):
        encoder = getattr(obj.Config, "json_encoders", {})
        if custom_encoder:
            encoder.update(custom_encoder)
        obj_dict = obj.dict(
            include=include,
            exclude=exclude,
            by_alias=by_alias,
            exclude_unset=bool(exclude_unset or skip_defaults),
        )

        return jsonable_encoder(
            obj_dict,
            include_none=include_none,
            custom_encoder=encoder,
            sqlalchemy_safe=sqlalchemy_safe,
        )
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, PurePath):
        return str(obj)
    if isinstance(obj, (str, int, float, type(None))):
        return obj
    if isinstance(obj, dict):
        encoded_dict = {}
        for key, value in obj.items():
            if (
                (not sqlalchemy_safe or (not isinstance(key, str)) or (not key.startswith("_sa")))
                and (value is not None or include_none)
                and ((include and key in include) or key not in exclude)
            ):
                encoded_key = jsonable_encoder(
                    key,
                    by_alias=by_alias,
                    exclude_unset=exclude_unset,
                    include_none=include_none,
                    custom_encoder=custom_encoder,
                    sqlalchemy_safe=sqlalchemy_safe,
                )
                encoded_value = jsonable_encoder(
                    value,
                    by_alias=by_alias,
                    exclude_unset=exclude_unset,
                    include_none=include_none,
                    custom_encoder=custom_encoder,
                    sqlalchemy_safe=sqlalchemy_safe,
                )
                encoded_dict[encoded_key] = encoded_value
        return encoded_dict
    if isinstance(obj, (list, set, frozenset, GeneratorType, tuple)):
        encoded_list = []
        for item in obj:
            encoded_list.append(
                jsonable_encoder(
                    item,
                    include=include,
                    exclude=exclude,
                    by_alias=by_alias,
                    exclude_unset=exclude_unset,
                    include_none=include_none,
                    custom_encoder=custom_encoder,
                    sqlalchemy_safe=sqlalchemy_safe,
                )
            )
        return encoded_list

    if custom_encoder:
        if type(obj) in custom_encoder:
            return custom_encoder[type(obj)](obj)
        else:
            for encoder_type, encoder in custom_encoder.items():
                if isinstance(obj, encoder_type):
                    return encoder(obj)

    if type(obj) in ENCODERS_BY_TYPE:
        return ENCODERS_BY_TYPE[type(obj)](obj)
    for encoder, classes_tuple in encoders_by_class_tuples.items():
        if isinstance(obj, classes_tuple):
            return encoder(obj)

    errors: List[Exception] = []
    try:
        data = dict(obj)
    except Exception as e:
        errors.append(e)
        try:
            data = vars(obj)
        except Exception as e:
            errors.append(e)
            raise ValueError(errors)
    return jsonable_encoder(
        data,
        by_alias=by_alias,
        exclude_unset=exclude_unset,
        include_none=include_none,
        custom_encoder=custom_encoder,
        sqlalchemy_safe=sqlalchemy_safe,
    )


if TYPE_CHECKING:
    from qdrant_client.http.api_client import ApiClient


class _PointsApi:
    def __init__(self, api_client: "Union[ApiClient, AsyncApiClient]"):
        self.api_client = api_client

    def _build_for_clear_payload(
        self,
        collection_name: str,
        wait: bool = None,
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

        body = jsonable_encoder(points_selector)

        return self.api_client.request(
            type_=m.InlineResponse2003,
            method="POST",
            url="/collections/{collection_name}/points/payload/clear",
            path_params=path_params,
            params=query_params,
            json=body,
        )

    def _build_for_delete_payload(
        self,
        collection_name: str,
        wait: bool = None,
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

        body = jsonable_encoder(delete_payload)

        return self.api_client.request(
            type_=m.InlineResponse2003,
            method="POST",
            url="/collections/{collection_name}/points/payload/delete",
            path_params=path_params,
            params=query_params,
            json=body,
        )

    def _build_for_delete_points(
        self,
        collection_name: str,
        wait: bool = None,
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

        body = jsonable_encoder(points_selector)

        return self.api_client.request(
            type_=m.InlineResponse2003,
            method="POST",
            url="/collections/{collection_name}/points/delete",
            path_params=path_params,
            params=query_params,
            json=body,
        )

    def _build_for_get_point(
        self,
        collection_name: str,
        id: m.ExtendedPointId,
    ):
        """
        Retrieve full information of single point by id
        """
        path_params = {
            "collection_name": str(collection_name),
            "id": str(id),
        }

        return self.api_client.request(
            type_=m.InlineResponse2007,
            method="GET",
            url="/collections/{collection_name}/points/{id}",
            path_params=path_params,
        )

    def _build_for_get_points(
        self,
        collection_name: str,
        point_request: m.PointRequest = None,
    ):
        """
        Retrieve multiple points by specified IDs
        """
        path_params = {
            "collection_name": str(collection_name),
        }

        body = jsonable_encoder(point_request)

        return self.api_client.request(
            type_=m.InlineResponse2004,
            method="POST",
            url="/collections/{collection_name}/points",
            path_params=path_params,
            json=body,
        )

    def _build_for_recommend_points(
        self,
        collection_name: str,
        recommend_request: m.RecommendRequest = None,
    ):
        """
        Look for the points which are closer to stored positive examples and at the same time further to negative examples.
        """
        path_params = {
            "collection_name": str(collection_name),
        }

        body = jsonable_encoder(recommend_request)

        return self.api_client.request(
            type_=m.InlineResponse2005,
            method="POST",
            url="/collections/{collection_name}/points/recommend",
            path_params=path_params,
            json=body,
        )

    def _build_for_scroll_points(
        self,
        collection_name: str,
        scroll_request: m.ScrollRequest = None,
    ):
        """
        Scroll request - paginate over all points which matches given filtering condition
        """
        path_params = {
            "collection_name": str(collection_name),
        }

        body = jsonable_encoder(scroll_request)

        return self.api_client.request(
            type_=m.InlineResponse2006,
            method="POST",
            url="/collections/{collection_name}/points/scroll",
            path_params=path_params,
            json=body,
        )

    def _build_for_search_points(
        self,
        collection_name: str,
        search_request: m.SearchRequest = None,
    ):
        """
        Retrieve closest points based on vector similarity and given filtering conditions
        """
        path_params = {
            "collection_name": str(collection_name),
        }

        body = jsonable_encoder(search_request)

        return self.api_client.request(
            type_=m.InlineResponse2005,
            method="POST",
            url="/collections/{collection_name}/points/search",
            path_params=path_params,
            json=body,
        )

    def _build_for_set_payload(
        self,
        collection_name: str,
        wait: bool = None,
        set_payload: m.SetPayload = None,
    ):
        """
        Set payload for points
        """
        path_params = {
            "collection_name": str(collection_name),
        }

        query_params = {}
        if wait is not None:
            query_params["wait"] = str(wait).lower()

        body = jsonable_encoder(set_payload)

        return self.api_client.request(
            type_=m.InlineResponse2003,
            method="POST",
            url="/collections/{collection_name}/points/payload",
            path_params=path_params,
            params=query_params,
            json=body,
        )

    def _build_for_update_points(
        self,
        collection_name: str,
        wait: bool = None,
        collection_update_operations: m.CollectionUpdateOperations = None,
    ):
        """
        Perform point update operation (vectors, payloads, indexes) in collection
        """
        path_params = {
            "collection_name": str(collection_name),
        }

        query_params = {}
        if wait is not None:
            query_params["wait"] = str(wait).lower()

        body = jsonable_encoder(collection_update_operations)

        return self.api_client.request(
            type_=m.InlineResponse2003,
            method="POST",
            url="/collections/{collection_name}",
            path_params=path_params,
            params=query_params,
            json=body,
        )

    def _build_for_upsert_points(
        self,
        collection_name: str,
        wait: bool = None,
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

        body = jsonable_encoder(point_insert_operations)

        return self.api_client.request(
            type_=m.InlineResponse2003,
            method="PUT",
            url="/collections/{collection_name}/points",
            path_params=path_params,
            params=query_params,
            json=body,
        )


class AsyncPointsApi(_PointsApi):
    async def clear_payload(
        self,
        collection_name: str,
        wait: bool = None,
        points_selector: m.PointsSelector = None,
    ) -> m.InlineResponse2003:
        """
        Remove all payload for specified points
        """
        return await self._build_for_clear_payload(
            collection_name=collection_name,
            wait=wait,
            points_selector=points_selector,
        )

    async def delete_payload(
        self,
        collection_name: str,
        wait: bool = None,
        delete_payload: m.DeletePayload = None,
    ) -> m.InlineResponse2003:
        """
        Delete specified key payload for points
        """
        return await self._build_for_delete_payload(
            collection_name=collection_name,
            wait=wait,
            delete_payload=delete_payload,
        )

    async def delete_points(
        self,
        collection_name: str,
        wait: bool = None,
        points_selector: m.PointsSelector = None,
    ) -> m.InlineResponse2003:
        """
        Delete points
        """
        return await self._build_for_delete_points(
            collection_name=collection_name,
            wait=wait,
            points_selector=points_selector,
        )

    async def get_point(
        self,
        collection_name: str,
        id: m.ExtendedPointId,
    ) -> m.InlineResponse2007:
        """
        Retrieve full information of single point by id
        """
        return await self._build_for_get_point(
            collection_name=collection_name,
            id=id,
        )

    async def get_points(
        self,
        collection_name: str,
        point_request: m.PointRequest = None,
    ) -> m.InlineResponse2004:
        """
        Retrieve multiple points by specified IDs
        """
        return await self._build_for_get_points(
            collection_name=collection_name,
            point_request=point_request,
        )

    async def recommend_points(
        self,
        collection_name: str,
        recommend_request: m.RecommendRequest = None,
    ) -> m.InlineResponse2005:
        """
        Look for the points which are closer to stored positive examples and at the same time further to negative examples.
        """
        return await self._build_for_recommend_points(
            collection_name=collection_name,
            recommend_request=recommend_request,
        )

    async def scroll_points(
        self,
        collection_name: str,
        scroll_request: m.ScrollRequest = None,
    ) -> m.InlineResponse2006:
        """
        Scroll request - paginate over all points which matches given filtering condition
        """
        return await self._build_for_scroll_points(
            collection_name=collection_name,
            scroll_request=scroll_request,
        )

    async def search_points(
        self,
        collection_name: str,
        search_request: m.SearchRequest = None,
    ) -> m.InlineResponse2005:
        """
        Retrieve closest points based on vector similarity and given filtering conditions
        """
        return await self._build_for_search_points(
            collection_name=collection_name,
            search_request=search_request,
        )

    async def set_payload(
        self,
        collection_name: str,
        wait: bool = None,
        set_payload: m.SetPayload = None,
    ) -> m.InlineResponse2003:
        """
        Set payload for points
        """
        return await self._build_for_set_payload(
            collection_name=collection_name,
            wait=wait,
            set_payload=set_payload,
        )

    async def update_points(
        self,
        collection_name: str,
        wait: bool = None,
        collection_update_operations: m.CollectionUpdateOperations = None,
    ) -> m.InlineResponse2003:
        """
        Perform point update operation (vectors, payloads, indexes) in collection
        """
        return await self._build_for_update_points(
            collection_name=collection_name,
            wait=wait,
            collection_update_operations=collection_update_operations,
        )

    async def upsert_points(
        self,
        collection_name: str,
        wait: bool = None,
        point_insert_operations: m.PointInsertOperations = None,
    ) -> m.InlineResponse2003:
        """
        Perform insert + updates on points. If point with given ID already exists - it will be overwritten.
        """
        return await self._build_for_upsert_points(
            collection_name=collection_name,
            wait=wait,
            point_insert_operations=point_insert_operations,
        )


class SyncPointsApi(_PointsApi):
    def clear_payload(
        self,
        collection_name: str,
        wait: bool = None,
        points_selector: m.PointsSelector = None,
    ) -> m.InlineResponse2003:
        """
        Remove all payload for specified points
        """
        return self._build_for_clear_payload(
            collection_name=collection_name,
            wait=wait,
            points_selector=points_selector,
        )

    def delete_payload(
        self,
        collection_name: str,
        wait: bool = None,
        delete_payload: m.DeletePayload = None,
    ) -> m.InlineResponse2003:
        """
        Delete specified key payload for points
        """
        return self._build_for_delete_payload(
            collection_name=collection_name,
            wait=wait,
            delete_payload=delete_payload,
        )

    def delete_points(
        self,
        collection_name: str,
        wait: bool = None,
        points_selector: m.PointsSelector = None,
    ) -> m.InlineResponse2003:
        """
        Delete points
        """
        return self._build_for_delete_points(
            collection_name=collection_name,
            wait=wait,
            points_selector=points_selector,
        )

    def get_point(
        self,
        collection_name: str,
        id: m.ExtendedPointId,
    ) -> m.InlineResponse2007:
        """
        Retrieve full information of single point by id
        """
        return self._build_for_get_point(
            collection_name=collection_name,
            id=id,
        )

    def get_points(
        self,
        collection_name: str,
        point_request: m.PointRequest = None,
    ) -> m.InlineResponse2004:
        """
        Retrieve multiple points by specified IDs
        """
        return self._build_for_get_points(
            collection_name=collection_name,
            point_request=point_request,
        )

    def recommend_points(
        self,
        collection_name: str,
        recommend_request: m.RecommendRequest = None,
    ) -> m.InlineResponse2005:
        """
        Look for the points which are closer to stored positive examples and at the same time further to negative examples.
        """
        return self._build_for_recommend_points(
            collection_name=collection_name,
            recommend_request=recommend_request,
        )

    def scroll_points(
        self,
        collection_name: str,
        scroll_request: m.ScrollRequest = None,
    ) -> m.InlineResponse2006:
        """
        Scroll request - paginate over all points which matches given filtering condition
        """
        return self._build_for_scroll_points(
            collection_name=collection_name,
            scroll_request=scroll_request,
        )

    def search_points(
        self,
        collection_name: str,
        search_request: m.SearchRequest = None,
    ) -> m.InlineResponse2005:
        """
        Retrieve closest points based on vector similarity and given filtering conditions
        """
        return self._build_for_search_points(
            collection_name=collection_name,
            search_request=search_request,
        )

    def set_payload(
        self,
        collection_name: str,
        wait: bool = None,
        set_payload: m.SetPayload = None,
    ) -> m.InlineResponse2003:
        """
        Set payload for points
        """
        return self._build_for_set_payload(
            collection_name=collection_name,
            wait=wait,
            set_payload=set_payload,
        )

    def update_points(
        self,
        collection_name: str,
        wait: bool = None,
        collection_update_operations: m.CollectionUpdateOperations = None,
    ) -> m.InlineResponse2003:
        """
        Perform point update operation (vectors, payloads, indexes) in collection
        """
        return self._build_for_update_points(
            collection_name=collection_name,
            wait=wait,
            collection_update_operations=collection_update_operations,
        )

    def upsert_points(
        self,
        collection_name: str,
        wait: bool = None,
        point_insert_operations: m.PointInsertOperations = None,
    ) -> m.InlineResponse2003:
        """
        Perform insert + updates on points. If point with given ID already exists - it will be overwritten.
        """
        return self._build_for_upsert_points(
            collection_name=collection_name,
            wait=wait,
            point_insert_operations=point_insert_operations,
        )
