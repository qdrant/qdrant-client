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


class _ClusterApi:
    def __init__(self, api_client: "Union[ApiClient, AsyncApiClient]"):
        self.api_client = api_client

    def _build_for_cluster_status(
        self,
    ):
        """
        Get information about the current state and composition of the cluster
        """
        headers = {}
        return self.api_client.request(
            type_=m.InlineResponse2002,
            method="GET",
            url="/cluster",
            headers=headers if headers else None,
        )

    def _build_for_collection_cluster_info(
        self,
        collection_name: str,
    ):
        """
        Get cluster information for a collection
        """
        path_params = {
            "collection_name": str(collection_name),
        }

        headers = {}
        return self.api_client.request(
            type_=m.InlineResponse2007,
            method="GET",
            url="/collections/{collection_name}/cluster",
            headers=headers if headers else None,
            path_params=path_params,
        )

    def _build_for_recover_current_peer(
        self,
    ):
        headers = {}
        return self.api_client.request(
            type_=m.InlineResponse2003,
            method="POST",
            url="/cluster/recover",
            headers=headers if headers else None,
        )

    def _build_for_remove_peer(
        self,
        peer_id: int,
        force: bool = None,
    ):
        """
        Tries to remove peer from the cluster. Will return an error if peer has shards on it.
        """
        path_params = {
            "peer_id": str(peer_id),
        }

        query_params = {}
        if force is not None:
            query_params["force"] = str(force).lower()

        headers = {}
        return self.api_client.request(
            type_=m.InlineResponse2003,
            method="DELETE",
            url="/cluster/peer/{peer_id}",
            headers=headers if headers else None,
            path_params=path_params,
            params=query_params,
        )

    def _build_for_update_collection_cluster(
        self,
        collection_name: str,
        timeout: int = None,
        cluster_operations: m.ClusterOperations = None,
    ):
        path_params = {
            "collection_name": str(collection_name),
        }

        query_params = {}
        if timeout is not None:
            query_params["timeout"] = str(timeout)

        headers = {}
        body = jsonable_encoder(cluster_operations)
        if "Content-Type" not in headers:
            headers["Content-Type"] = "application/json"
        return self.api_client.request(
            type_=m.InlineResponse2003,
            method="POST",
            url="/collections/{collection_name}/cluster",
            headers=headers if headers else None,
            path_params=path_params,
            params=query_params,
            data=body,
        )


class AsyncClusterApi(_ClusterApi):
    async def cluster_status(
        self,
    ) -> m.InlineResponse2002:
        """
        Get information about the current state and composition of the cluster
        """
        return await self._build_for_cluster_status()

    async def collection_cluster_info(
        self,
        collection_name: str,
    ) -> m.InlineResponse2007:
        """
        Get cluster information for a collection
        """
        return await self._build_for_collection_cluster_info(
            collection_name=collection_name,
        )

    async def recover_current_peer(
        self,
    ) -> m.InlineResponse2003:
        return await self._build_for_recover_current_peer()

    async def remove_peer(
        self,
        peer_id: int,
        force: bool = None,
    ) -> m.InlineResponse2003:
        """
        Tries to remove peer from the cluster. Will return an error if peer has shards on it.
        """
        return await self._build_for_remove_peer(
            peer_id=peer_id,
            force=force,
        )

    async def update_collection_cluster(
        self,
        collection_name: str,
        timeout: int = None,
        cluster_operations: m.ClusterOperations = None,
    ) -> m.InlineResponse2003:
        return await self._build_for_update_collection_cluster(
            collection_name=collection_name,
            timeout=timeout,
            cluster_operations=cluster_operations,
        )


class SyncClusterApi(_ClusterApi):
    def cluster_status(
        self,
    ) -> m.InlineResponse2002:
        """
        Get information about the current state and composition of the cluster
        """
        return self._build_for_cluster_status()

    def collection_cluster_info(
        self,
        collection_name: str,
    ) -> m.InlineResponse2007:
        """
        Get cluster information for a collection
        """
        return self._build_for_collection_cluster_info(
            collection_name=collection_name,
        )

    def recover_current_peer(
        self,
    ) -> m.InlineResponse2003:
        return self._build_for_recover_current_peer()

    def remove_peer(
        self,
        peer_id: int,
        force: bool = None,
    ) -> m.InlineResponse2003:
        """
        Tries to remove peer from the cluster. Will return an error if peer has shards on it.
        """
        return self._build_for_remove_peer(
            peer_id=peer_id,
            force=force,
        )

    def update_collection_cluster(
        self,
        collection_name: str,
        timeout: int = None,
        cluster_operations: m.ClusterOperations = None,
    ) -> m.InlineResponse2003:
        return self._build_for_update_collection_cluster(
            collection_name=collection_name,
            timeout=timeout,
            cluster_operations=cluster_operations,
        )
