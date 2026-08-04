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


class _QuotasApi:
    def __init__(self, api_client: "Union[ApiClient, AsyncApiClient]"):
        self.api_client = api_client

    def _build_for_get_quotas(
        self,
    ):
        """
        Get the cluster-wide resource quota configuration, together with the current utilization it is measured against. The configuration is the same on every peer, but the reported utilization is for the node serving this request only - memory and disk are node-local, so query each peer to see where the whole cluster stands.
        """
        headers = {}
        return self.api_client.request(
            type_=m.InlineResponse2005,
            method="GET",
            url="/quotas",
            headers=headers if headers else None,
        )

    def _build_for_update_quotas(
        self,
        wait: bool = None,
        quota_config: m.QuotaConfig = None,
    ):
        """
        Replace the cluster-wide resource quota configuration. The new configuration is propagated to every peer through consensus and persisted, so it survives restarts
        """
        query_params = {}
        if wait is not None:
            query_params["wait"] = str(wait).lower()

        headers = {}
        body = jsonable_encoder(quota_config)
        if "Content-Type" not in headers:
            headers["Content-Type"] = "application/json"
        return self.api_client.request(
            type_=m.InlineResponse2001,
            method="PUT",
            url="/quotas",
            headers=headers if headers else None,
            params=query_params,
            content=body,
        )


class AsyncQuotasApi(_QuotasApi):
    async def get_quotas(
        self,
    ) -> m.InlineResponse2005:
        """
        Get the cluster-wide resource quota configuration, together with the current utilization it is measured against. The configuration is the same on every peer, but the reported utilization is for the node serving this request only - memory and disk are node-local, so query each peer to see where the whole cluster stands.
        """
        return await self._build_for_get_quotas()

    async def update_quotas(
        self,
        wait: bool = None,
        quota_config: m.QuotaConfig = None,
    ) -> m.InlineResponse2001:
        """
        Replace the cluster-wide resource quota configuration. The new configuration is propagated to every peer through consensus and persisted, so it survives restarts
        """
        return await self._build_for_update_quotas(
            wait=wait,
            quota_config=quota_config,
        )


class SyncQuotasApi(_QuotasApi):
    def get_quotas(
        self,
    ) -> m.InlineResponse2005:
        """
        Get the cluster-wide resource quota configuration, together with the current utilization it is measured against. The configuration is the same on every peer, but the reported utilization is for the node serving this request only - memory and disk are node-local, so query each peer to see where the whole cluster stands.
        """
        return self._build_for_get_quotas()

    def update_quotas(
        self,
        wait: bool = None,
        quota_config: m.QuotaConfig = None,
    ) -> m.InlineResponse2001:
        """
        Replace the cluster-wide resource quota configuration. The new configuration is propagated to every peer through consensus and persisted, so it survives restarts
        """
        return self._build_for_update_quotas(
            wait=wait,
            quota_config=quota_config,
        )
