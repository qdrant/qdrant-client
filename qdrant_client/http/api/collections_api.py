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


class _CollectionsApi:
    def __init__(self, api_client: "Union[ApiClient, AsyncApiClient]"):
        self.api_client = api_client

    def _build_for_create_collection(
        self,
        collection_name: str,
        create_collection: m.CreateCollection = None,
    ):
        """
        Create new collection with given parameters
        """
        path_params = {
            "collection_name": str(collection_name),
        }

        body = jsonable_encoder(create_collection)

        return self.api_client.request(
            type_=m.InlineResponse2001,
            method="PUT",
            url="/collections/{collection_name}",
            path_params=path_params,
            json=body,
        )

    def _build_for_create_field_index(
        self,
        collection_name: str,
        wait: bool = None,
        create_field_index: m.CreateFieldIndex = None,
    ):
        """
        Create index for field in collection
        """
        path_params = {
            "collection_name": str(collection_name),
        }

        query_params = {}
        if wait is not None:
            query_params["wait"] = str(wait).lower()

        body = jsonable_encoder(create_field_index)

        return self.api_client.request(
            type_=m.InlineResponse2003,
            method="PUT",
            url="/collections/{collection_name}/index",
            path_params=path_params,
            params=query_params,
            json=body,
        )

    def _build_for_delete_collection(
        self,
        collection_name: str,
    ):
        """
        Drop collection and all associated data
        """
        path_params = {
            "collection_name": str(collection_name),
        }

        return self.api_client.request(
            type_=m.InlineResponse2001,
            method="DELETE",
            url="/collections/{collection_name}",
            path_params=path_params,
        )

    def _build_for_delete_field_index(
        self,
        collection_name: str,
        field_name: str,
        wait: bool = None,
    ):
        """
        Delete field index for collection
        """
        path_params = {
            "collection_name": str(collection_name),
            "field_name": str(field_name),
        }

        query_params = {}
        if wait is not None:
            query_params["wait"] = str(wait).lower()

        return self.api_client.request(
            type_=m.InlineResponse2003,
            method="DELETE",
            url="/collections/{collection_name}/index/{field_name}",
            path_params=path_params,
            params=query_params,
        )

    def _build_for_get_collection(
        self,
        collection_name: str,
    ):
        """
        Get detailed information about specified existing collection
        """
        path_params = {
            "collection_name": str(collection_name),
        }

        return self.api_client.request(
            type_=m.InlineResponse2002,
            method="GET",
            url="/collections/{collection_name}",
            path_params=path_params,
        )

    def _build_for_get_collections(
        self,
    ):
        """
        Get list name of all existing collections
        """
        return self.api_client.request(
            type_=m.InlineResponse200,
            method="GET",
            url="/collections",
        )

    def _build_for_update_aliases(
        self,
        change_aliases_operation: m.ChangeAliasesOperation = None,
    ):
        body = jsonable_encoder(change_aliases_operation)

        return self.api_client.request(type_=m.InlineResponse2001, method="POST", url="/collections/aliases", json=body)

    def _build_for_update_collection(
        self,
        collection_name: str,
        update_collection: m.UpdateCollection = None,
    ):
        """
        Update parameters of the existing collection
        """
        path_params = {
            "collection_name": str(collection_name),
        }

        body = jsonable_encoder(update_collection)

        return self.api_client.request(
            type_=m.InlineResponse2001,
            method="PATCH",
            url="/collections/{collection_name}",
            path_params=path_params,
            json=body,
        )

    def _build_for_update_collections(
        self,
        collection_meta_operations: m.CollectionMetaOperations = None,
    ):
        """
        Perform update, create, remove or alias change operations on collections
        """
        body = jsonable_encoder(collection_meta_operations)

        return self.api_client.request(type_=m.InlineResponse2001, method="POST", url="/collections", json=body)


class AsyncCollectionsApi(_CollectionsApi):
    async def create_collection(
        self,
        collection_name: str,
        create_collection: m.CreateCollection = None,
    ) -> m.InlineResponse2001:
        """
        Create new collection with given parameters
        """
        return await self._build_for_create_collection(
            collection_name=collection_name,
            create_collection=create_collection,
        )

    async def create_field_index(
        self,
        collection_name: str,
        wait: bool = None,
        create_field_index: m.CreateFieldIndex = None,
    ) -> m.InlineResponse2003:
        """
        Create index for field in collection
        """
        return await self._build_for_create_field_index(
            collection_name=collection_name,
            wait=wait,
            create_field_index=create_field_index,
        )

    async def delete_collection(
        self,
        collection_name: str,
    ) -> m.InlineResponse2001:
        """
        Drop collection and all associated data
        """
        return await self._build_for_delete_collection(
            collection_name=collection_name,
        )

    async def delete_field_index(
        self,
        collection_name: str,
        field_name: str,
        wait: bool = None,
    ) -> m.InlineResponse2003:
        """
        Delete field index for collection
        """
        return await self._build_for_delete_field_index(
            collection_name=collection_name,
            field_name=field_name,
            wait=wait,
        )

    async def get_collection(
        self,
        collection_name: str,
    ) -> m.InlineResponse2002:
        """
        Get detailed information about specified existing collection
        """
        return await self._build_for_get_collection(
            collection_name=collection_name,
        )

    async def get_collections(
        self,
    ) -> m.InlineResponse200:
        """
        Get list name of all existing collections
        """
        return await self._build_for_get_collections()

    async def update_aliases(
        self,
        change_aliases_operation: m.ChangeAliasesOperation = None,
    ) -> m.InlineResponse2001:
        return await self._build_for_update_aliases(
            change_aliases_operation=change_aliases_operation,
        )

    async def update_collection(
        self,
        collection_name: str,
        update_collection: m.UpdateCollection = None,
    ) -> m.InlineResponse2001:
        """
        Update parameters of the existing collection
        """
        return await self._build_for_update_collection(
            collection_name=collection_name,
            update_collection=update_collection,
        )

    async def update_collections(
        self,
        collection_meta_operations: m.CollectionMetaOperations = None,
    ) -> m.InlineResponse2001:
        """
        Perform update, create, remove or alias change operations on collections
        """
        return await self._build_for_update_collections(
            collection_meta_operations=collection_meta_operations,
        )


class SyncCollectionsApi(_CollectionsApi):
    def create_collection(
        self,
        collection_name: str,
        create_collection: m.CreateCollection = None,
    ) -> m.InlineResponse2001:
        """
        Create new collection with given parameters
        """
        return self._build_for_create_collection(
            collection_name=collection_name,
            create_collection=create_collection,
        )

    def create_field_index(
        self,
        collection_name: str,
        wait: bool = None,
        create_field_index: m.CreateFieldIndex = None,
    ) -> m.InlineResponse2003:
        """
        Create index for field in collection
        """
        return self._build_for_create_field_index(
            collection_name=collection_name,
            wait=wait,
            create_field_index=create_field_index,
        )

    def delete_collection(
        self,
        collection_name: str,
    ) -> m.InlineResponse2001:
        """
        Drop collection and all associated data
        """
        return self._build_for_delete_collection(
            collection_name=collection_name,
        )

    def delete_field_index(
        self,
        collection_name: str,
        field_name: str,
        wait: bool = None,
    ) -> m.InlineResponse2003:
        """
        Delete field index for collection
        """
        return self._build_for_delete_field_index(
            collection_name=collection_name,
            field_name=field_name,
            wait=wait,
        )

    def get_collection(
        self,
        collection_name: str,
    ) -> m.InlineResponse2002:
        """
        Get detailed information about specified existing collection
        """
        return self._build_for_get_collection(
            collection_name=collection_name,
        )

    def get_collections(
        self,
    ) -> m.InlineResponse200:
        """
        Get list name of all existing collections
        """
        return self._build_for_get_collections()

    def update_aliases(
        self,
        change_aliases_operation: m.ChangeAliasesOperation = None,
    ) -> m.InlineResponse2001:
        return self._build_for_update_aliases(
            change_aliases_operation=change_aliases_operation,
        )

    def update_collection(
        self,
        collection_name: str,
        update_collection: m.UpdateCollection = None,
    ) -> m.InlineResponse2001:
        """
        Update parameters of the existing collection
        """
        return self._build_for_update_collection(
            collection_name=collection_name,
            update_collection=update_collection,
        )

    def update_collections(
        self,
        collection_meta_operations: m.CollectionMetaOperations = None,
    ) -> m.InlineResponse2001:
        """
        Perform update, create, remove or alias change operations on collections
        """
        return self._build_for_update_collections(
            collection_meta_operations=collection_meta_operations,
        )
