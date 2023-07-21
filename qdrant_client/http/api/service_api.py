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


class _ServiceApi:
    def __init__(self, api_client: "Union[ApiClient, AsyncApiClient]"):
        self.api_client = api_client

    def _build_for_get_locks(
        self,
    ):
        """
        Get lock options. If write is locked, all write operations and collection creation are forbidden
        """
        return self.api_client.request(
            type_=m.InlineResponse2001,
            method="GET",
            url="/locks",
        )

    def _build_for_metrics(
        self,
        anonymize: bool = None,
    ):
        """
        Collect metrics data including app info, collections info, cluster info and statistics
        """
        query_params = {}
        if anonymize is not None:
            query_params["anonymize"] = str(anonymize).lower()

        return self.api_client.request(
            type_=str,
            method="GET",
            url="/metrics",
            params=query_params,
        )

    def _build_for_post_locks(
        self,
        locks_option: m.LocksOption = None,
    ):
        """
        Set lock options. If write is locked, all write operations and collection creation are forbidden. Returns previous lock options
        """
        headers = {}
        body = jsonable_encoder(locks_option)
        if "Content-Type" not in headers:
            headers["Content-Type"] = "application/json"
        return self.api_client.request(type_=m.InlineResponse2001, method="POST", url="/locks", data=body)

    def _build_for_telemetry(
        self,
        anonymize: bool = None,
    ):
        """
        Collect telemetry data including app info, system info, collections info, cluster info, configs and statistics
        """
        query_params = {}
        if anonymize is not None:
            query_params["anonymize"] = str(anonymize).lower()

        return self.api_client.request(
            type_=m.InlineResponse200,
            method="GET",
            url="/telemetry",
            params=query_params,
        )


class AsyncServiceApi(_ServiceApi):
    async def get_locks(
        self,
    ) -> m.InlineResponse2001:
        """
        Get lock options. If write is locked, all write operations and collection creation are forbidden
        """
        return await self._build_for_get_locks()

    async def metrics(
        self,
        anonymize: bool = None,
    ) -> str:
        """
        Collect metrics data including app info, collections info, cluster info and statistics
        """
        return await self._build_for_metrics(
            anonymize=anonymize,
        )

    async def post_locks(
        self,
        locks_option: m.LocksOption = None,
    ) -> m.InlineResponse2001:
        """
        Set lock options. If write is locked, all write operations and collection creation are forbidden. Returns previous lock options
        """
        return await self._build_for_post_locks(
            locks_option=locks_option,
        )

    async def telemetry(
        self,
        anonymize: bool = None,
    ) -> m.InlineResponse200:
        """
        Collect telemetry data including app info, system info, collections info, cluster info, configs and statistics
        """
        return await self._build_for_telemetry(
            anonymize=anonymize,
        )


class SyncServiceApi(_ServiceApi):
    def get_locks(
        self,
    ) -> m.InlineResponse2001:
        """
        Get lock options. If write is locked, all write operations and collection creation are forbidden
        """
        return self._build_for_get_locks()

    def metrics(
        self,
        anonymize: bool = None,
    ) -> str:
        """
        Collect metrics data including app info, collections info, cluster info and statistics
        """
        return self._build_for_metrics(
            anonymize=anonymize,
        )

    def post_locks(
        self,
        locks_option: m.LocksOption = None,
    ) -> m.InlineResponse2001:
        """
        Set lock options. If write is locked, all write operations and collection creation are forbidden. Returns previous lock options
        """
        return self._build_for_post_locks(
            locks_option=locks_option,
        )

    def telemetry(
        self,
        anonymize: bool = None,
    ) -> m.InlineResponse200:
        """
        Collect telemetry data including app info, system info, collections info, cluster info, configs and statistics
        """
        return self._build_for_telemetry(
            anonymize=anonymize,
        )
