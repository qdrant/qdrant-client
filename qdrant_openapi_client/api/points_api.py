# flake8: noqa E501
from enum import Enum
from pathlib import PurePath
from types import GeneratorType
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Set, Tuple, Union

from pydantic.json import ENCODERS_BY_TYPE
from pydantic.main import BaseModel
from qdrant_openapi_client.models import models as m

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
    from qdrant_openapi_client.api_client import ApiClient


class _PointsApi:
    def __init__(self, api_client: "Union[ApiClient, AsyncApiClient]"):
        self.api_client = api_client

    def _build_for_get_point(
        self,
        name: str,
        id: int,
    ):
        path_params = {
            "name": str(name),
            "id": str(id),
        }

        return self.api_client.request(
            type_=m.InlineResponse2004,
            method="GET",
            url="/collections/{name}/points/{id}",
            path_params=path_params,
        )

    def _build_for_get_points(
        self,
        name: str,
        point_request: m.PointRequest = None,
    ):
        path_params = {
            "name": str(name),
        }

        body = jsonable_encoder(point_request)

        return self.api_client.request(
            type_=m.InlineResponse2005,
            method="POST",
            url="/collections/{name}/points",
            path_params=path_params,
            json=body,
        )

    def _build_for_recommend_points(
        self,
        name: str,
        recommend_request: m.RecommendRequest = None,
    ):
        path_params = {
            "name": str(name),
        }

        body = jsonable_encoder(recommend_request)

        return self.api_client.request(
            type_=m.InlineResponse2006,
            method="POST",
            url="/collections/{name}/points/recommend",
            path_params=path_params,
            json=body,
        )

    def _build_for_search_points(
        self,
        name: str,
        search_request: m.SearchRequest = None,
    ):
        path_params = {
            "name": str(name),
        }

        body = jsonable_encoder(search_request)

        return self.api_client.request(
            type_=m.InlineResponse2006,
            method="POST",
            url="/collections/{name}/points/search",
            path_params=path_params,
            json=body,
        )

    def _build_for_update_points(
        self,
        name: str,
        wait: bool = None,
        collection_update_operations: m.CollectionUpdateOperations = None,
    ):
        path_params = {
            "name": str(name),
        }

        query_params = {}
        if wait is not None:
            query_params["wait"] = str(wait)

        body = jsonable_encoder(collection_update_operations)

        return self.api_client.request(
            type_=m.InlineResponse2003,
            method="POST",
            url="/collections/{name}",
            path_params=path_params,
            params=query_params,
            json=body,
        )


class AsyncPointsApi(_PointsApi):
    async def get_point(
        self,
        name: str,
        id: int,
    ) -> m.InlineResponse2004:
        return await self._build_for_get_point(
            name=name,
            id=id,
        )

    async def get_points(
        self,
        name: str,
        point_request: m.PointRequest = None,
    ) -> m.InlineResponse2005:
        return await self._build_for_get_points(
            name=name,
            point_request=point_request,
        )

    async def recommend_points(
        self,
        name: str,
        recommend_request: m.RecommendRequest = None,
    ) -> m.InlineResponse2006:
        return await self._build_for_recommend_points(
            name=name,
            recommend_request=recommend_request,
        )

    async def search_points(
        self,
        name: str,
        search_request: m.SearchRequest = None,
    ) -> m.InlineResponse2006:
        return await self._build_for_search_points(
            name=name,
            search_request=search_request,
        )

    async def update_points(
        self,
        name: str,
        wait: bool = None,
        collection_update_operations: m.CollectionUpdateOperations = None,
    ) -> m.InlineResponse2003:
        return await self._build_for_update_points(
            name=name,
            wait=wait,
            collection_update_operations=collection_update_operations,
        )


class SyncPointsApi(_PointsApi):
    def get_point(
        self,
        name: str,
        id: int,
    ) -> m.InlineResponse2004:
        return self._build_for_get_point(
            name=name,
            id=id,
        )

    def get_points(
        self,
        name: str,
        point_request: m.PointRequest = None,
    ) -> m.InlineResponse2005:
        return self._build_for_get_points(
            name=name,
            point_request=point_request,
        )

    def recommend_points(
        self,
        name: str,
        recommend_request: m.RecommendRequest = None,
    ) -> m.InlineResponse2006:
        return self._build_for_recommend_points(
            name=name,
            recommend_request=recommend_request,
        )

    def search_points(
        self,
        name: str,
        search_request: m.SearchRequest = None,
    ) -> m.InlineResponse2006:
        return self._build_for_search_points(
            name=name,
            search_request=search_request,
        )

    def update_points(
        self,
        name: str,
        wait: bool = None,
        collection_update_operations: m.CollectionUpdateOperations = None,
    ) -> m.InlineResponse2003:
        return self._build_for_update_points(
            name=name,
            wait=wait,
            collection_update_operations=collection_update_operations,
        )
