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
file = None


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


class _SnapshotsApi:
    def __init__(self, api_client: "Union[ApiClient, AsyncApiClient]"):
        self.api_client = api_client

    def _build_for_create_full_snapshot(
        self,
    ):
        """
        Create new snapshot of the whole storage
        """
        return self.api_client.request(
            type_=m.InlineResponse2008,
            method="POST",
            url="/snapshots",
        )

    def _build_for_create_snapshot(
        self,
        collection_name: str,
    ):
        """
        Create new snapshot for a collection
        """
        path_params = {
            "collection_name": str(collection_name),
        }

        return self.api_client.request(
            type_=m.InlineResponse2008,
            method="POST",
            url="/collections/{collection_name}/snapshots",
            path_params=path_params,
        )

    def _build_for_get_full_snapshot(
        self,
        snapshot_name: str,
    ):
        """
        Download specified snapshot of the whole storage as a file
        """
        path_params = {
            "snapshot_name": str(snapshot_name),
        }

        return self.api_client.request(
            type_=file,
            method="GET",
            url="/snapshots/{snapshot_name}",
            path_params=path_params,
        )

    def _build_for_get_snapshot(
        self,
        collection_name: str,
        snapshot_name: str,
    ):
        """
        Download specified snapshot from a collection as a file
        """
        path_params = {
            "collection_name": str(collection_name),
            "snapshot_name": str(snapshot_name),
        }

        return self.api_client.request(
            type_=file,
            method="GET",
            url="/collections/{collection_name}/snapshots/{snapshot_name}",
            path_params=path_params,
        )

    def _build_for_list_full_snapshots(
        self,
    ):
        """
        Get list of snapshots of the whole storage
        """
        return self.api_client.request(
            type_=m.InlineResponse2007,
            method="GET",
            url="/snapshots",
        )

    def _build_for_list_snapshots(
        self,
        collection_name: str,
    ):
        """
        Get list of snapshots for a collection
        """
        path_params = {
            "collection_name": str(collection_name),
        }

        return self.api_client.request(
            type_=m.InlineResponse2007,
            method="GET",
            url="/collections/{collection_name}/snapshots",
            path_params=path_params,
        )


class AsyncSnapshotsApi(_SnapshotsApi):
    async def create_full_snapshot(
        self,
    ) -> m.InlineResponse2008:
        """
        Create new snapshot of the whole storage
        """
        return await self._build_for_create_full_snapshot()

    async def create_snapshot(
        self,
        collection_name: str,
    ) -> m.InlineResponse2008:
        """
        Create new snapshot for a collection
        """
        return await self._build_for_create_snapshot(
            collection_name=collection_name,
        )

    async def get_full_snapshot(
        self,
        snapshot_name: str,
    ) -> file:
        """
        Download specified snapshot of the whole storage as a file
        """
        return await self._build_for_get_full_snapshot(
            snapshot_name=snapshot_name,
        )

    async def get_snapshot(
        self,
        collection_name: str,
        snapshot_name: str,
    ) -> file:
        """
        Download specified snapshot from a collection as a file
        """
        return await self._build_for_get_snapshot(
            collection_name=collection_name,
            snapshot_name=snapshot_name,
        )

    async def list_full_snapshots(
        self,
    ) -> m.InlineResponse2007:
        """
        Get list of snapshots of the whole storage
        """
        return await self._build_for_list_full_snapshots()

    async def list_snapshots(
        self,
        collection_name: str,
    ) -> m.InlineResponse2007:
        """
        Get list of snapshots for a collection
        """
        return await self._build_for_list_snapshots(
            collection_name=collection_name,
        )


class SyncSnapshotsApi(_SnapshotsApi):
    def create_full_snapshot(
        self,
    ) -> m.InlineResponse2008:
        """
        Create new snapshot of the whole storage
        """
        return self._build_for_create_full_snapshot()

    def create_snapshot(
        self,
        collection_name: str,
    ) -> m.InlineResponse2008:
        """
        Create new snapshot for a collection
        """
        return self._build_for_create_snapshot(
            collection_name=collection_name,
        )

    def get_full_snapshot(
        self,
        snapshot_name: str,
    ) -> file:
        """
        Download specified snapshot of the whole storage as a file
        """
        return self._build_for_get_full_snapshot(
            snapshot_name=snapshot_name,
        )

    def get_snapshot(
        self,
        collection_name: str,
        snapshot_name: str,
    ) -> file:
        """
        Download specified snapshot from a collection as a file
        """
        return self._build_for_get_snapshot(
            collection_name=collection_name,
            snapshot_name=snapshot_name,
        )

    def list_full_snapshots(
        self,
    ) -> m.InlineResponse2007:
        """
        Get list of snapshots of the whole storage
        """
        return self._build_for_list_full_snapshots()

    def list_snapshots(
        self,
        collection_name: str,
    ) -> m.InlineResponse2007:
        """
        Get list of snapshots for a collection
        """
        return self._build_for_list_snapshots(
            collection_name=collection_name,
        )
