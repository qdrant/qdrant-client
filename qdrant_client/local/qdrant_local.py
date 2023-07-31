import itertools
import json
import logging
import os
import shutil
from io import TextIOWrapper
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import portalocker

from qdrant_client._pydantic_compat import to_dict
from qdrant_client.client_base import QdrantBase
from qdrant_client.conversions import common_types as types
from qdrant_client.http import models as rest_models
from qdrant_client.local.local_collection import LocalCollection

META_INFO_FILENAME = "meta.json"


class QdrantLocal(QdrantBase):
    """
    Everything Qdrant server can do, but locally.

    Use this implementation to run vector search without running a Qdrant server.
    Everything that works with local Qdrant will work with server Qdrant as well.

    Use for small-scale data, demos, and tests.
    If you need more speed or size, use Qdrant server.
    """

    def __init__(self, location: str) -> None:
        """
        Initialize local Qdrant.

        Args:
            location: Where to store data. Can be a path to a directory or `:memory:` for in-memory storage.
        """
        self.location = location
        self.persistent = location != ":memory:"
        self.collections: Dict[str, LocalCollection] = {}
        self.aliases: Dict[str, str] = {}
        self._flock_file: Optional[TextIOWrapper] = None
        self._load()
        self._closed: bool = False

    def closed(self) -> bool:
        return self._closed

    def close(self, **kwargs: Any) -> None:
        self._closed = True
        for collection in self.collections.values():
            collection.close()
        if self._flock_file is not None:
            portalocker.unlock(self._flock_file)

    def _load(self) -> None:
        if not self.persistent:
            return
        meta_path = os.path.join(self.location, META_INFO_FILENAME)
        if not os.path.exists(meta_path):
            os.makedirs(self.location, exist_ok=True)
            with open(meta_path, "w") as f:
                f.write(json.dumps({"collections": {}, "aliases": {}}))
        else:
            with open(meta_path, "r") as f:
                meta = json.load(f)
                for collection_name, config_json in meta["collections"].items():
                    config = rest_models.CreateCollection(**config_json)
                    collection_path = self._collection_path(collection_name)
                    self.collections[collection_name] = LocalCollection(config, collection_path)
                self.aliases = meta["aliases"]

        lock_file_path = os.path.join(self.location, ".lock")
        if not os.path.exists(lock_file_path):
            os.makedirs(self.location, exist_ok=True)
            with open(lock_file_path, "w") as f:
                f.write("tmp lock file")
        self._flock_file = open(lock_file_path, "r+")
        try:
            portalocker.lock(
                self._flock_file,
                portalocker.LockFlags.EXCLUSIVE | portalocker.LockFlags.NON_BLOCKING,
            )
        except portalocker.exceptions.LockException:
            raise RuntimeError(
                f"Storage folder {self.location} is already accessed by another instance of Qdrant client."
                f" If you require concurrent access, use Qdrant server instead."
            )

    def _save(self) -> None:
        if not self.persistent:
            return
        meta_path = os.path.join(self.location, META_INFO_FILENAME)
        with open(meta_path, "w") as f:
            f.write(
                json.dumps(
                    {
                        "collections": {
                            collection_name: to_dict(collection.config)
                            for collection_name, collection in self.collections.items()
                        },
                        "aliases": self.aliases,
                    }
                )
            )

    def _get_collection(self, collection_name: str) -> LocalCollection:
        if collection_name in self.collections:
            return self.collections[collection_name]
        if collection_name in self.aliases:
            return self.collections[self.aliases[collection_name]]
        raise ValueError(f"Collection {collection_name} not found")

    def search_batch(
        self,
        collection_name: str,
        requests: Sequence[types.SearchRequest],
        **kwargs: Any,
    ) -> List[List[types.ScoredPoint]]:
        collection = self._get_collection(collection_name)

        return [
            collection.search(
                query_vector=request.vector,
                query_filter=request.filter,
                limit=request.limit,
                offset=request.offset,
                with_payload=request.with_payload,
                with_vectors=request.with_vector,
                score_threshold=request.score_threshold,
            )
            for request in requests
        ]

    def search(
        self,
        collection_name: str,
        query_vector: Union[
            types.NumpyArray,
            Sequence[float],
            Tuple[str, List[float]],
            types.NamedVector,
        ],
        query_filter: Optional[types.Filter] = None,
        search_params: Optional[types.SearchParams] = None,
        limit: int = 10,
        offset: int = 0,
        with_payload: Union[bool, Sequence[str], types.PayloadSelector] = True,
        with_vectors: Union[bool, Sequence[str]] = False,
        score_threshold: Optional[float] = None,
        **kwargs: Any,
    ) -> List[types.ScoredPoint]:
        collection = self._get_collection(collection_name)
        return collection.search(
            query_vector=query_vector,
            query_filter=query_filter,
            limit=limit,
            offset=offset,
            with_payload=with_payload,
            with_vectors=with_vectors,
            score_threshold=score_threshold,
        )

    def search_groups(
        self,
        collection_name: str,
        query_vector: Union[
            types.NumpyArray,
            Sequence[float],
            Tuple[str, List[float]],
            types.NamedVector,
        ],
        group_by: str,
        query_filter: Optional[rest_models.Filter] = None,
        search_params: Optional[rest_models.SearchParams] = None,
        limit: int = 10,
        group_size: int = 1,
        with_payload: Union[bool, Sequence[str], rest_models.PayloadSelector] = True,
        with_vectors: Union[bool, Sequence[str]] = False,
        score_threshold: Optional[float] = None,
        with_lookup: Optional[types.WithLookupInterface] = None,
        **kwargs: Any,
    ) -> types.GroupsResult:
        collection = self._get_collection(collection_name)
        with_lookup_collection = None
        if with_lookup is not None:
            if isinstance(with_lookup, str):
                with_lookup_collection = self._get_collection(with_lookup)
            else:
                with_lookup_collection = self._get_collection(with_lookup.collection)

        return collection.search_groups(
            query_vector=query_vector,
            query_filter=query_filter,
            limit=limit,
            group_by=group_by,
            group_size=group_size,
            with_payload=with_payload,
            with_vectors=with_vectors,
            score_threshold=score_threshold,
            with_lookup=with_lookup,
            with_lookup_collection=with_lookup_collection,
        )

    def recommend_batch(
        self,
        collection_name: str,
        requests: Sequence[types.RecommendRequest],
        **kwargs: Any,
    ) -> List[List[types.ScoredPoint]]:
        collection = self._get_collection(collection_name)

        return [
            collection.recommend(
                positive=request.positive,
                negative=request.negative,
                query_filter=request.filter,
                limit=request.limit,
                offset=request.offset,
                with_payload=request.with_payload,
                with_vectors=request.with_vector,
                score_threshold=request.score_threshold,
            )
            for request in requests
        ]

    def recommend(
        self,
        collection_name: str,
        positive: Sequence[types.PointId],
        negative: Optional[Sequence[types.PointId]] = None,
        query_filter: Optional[types.Filter] = None,
        search_params: Optional[types.SearchParams] = None,
        limit: int = 10,
        offset: int = 0,
        with_payload: Union[bool, List[str], types.PayloadSelector] = True,
        with_vectors: Union[bool, List[str]] = False,
        score_threshold: Optional[float] = None,
        using: Optional[str] = None,
        lookup_from: Optional[types.LookupLocation] = None,
        **kwargs: Any,
    ) -> List[types.ScoredPoint]:
        collection = self._get_collection(collection_name)
        return collection.recommend(
            positive=positive,
            negative=negative,
            query_filter=query_filter,
            limit=limit,
            offset=offset,
            with_payload=with_payload,
            with_vectors=with_vectors,
            score_threshold=score_threshold,
            using=using,
            lookup_from_collection=self._get_collection(lookup_from.collection)
            if lookup_from
            else None,
            lookup_from_vector_name=lookup_from.vector if lookup_from else None,
        )

    def recommend_groups(
        self,
        collection_name: str,
        group_by: str,
        positive: Sequence[types.PointId],
        negative: Optional[Sequence[types.PointId]] = None,
        query_filter: Optional[types.Filter] = None,
        search_params: Optional[types.SearchParams] = None,
        limit: int = 10,
        group_size: int = 1,
        score_threshold: Optional[float] = None,
        with_payload: Union[bool, Sequence[str], types.PayloadSelector] = True,
        with_vectors: Union[bool, Sequence[str]] = False,
        using: Optional[str] = None,
        lookup_from: Optional[types.LookupLocation] = None,
        with_lookup: Optional[types.WithLookupInterface] = None,
        **kwargs: Any,
    ) -> types.GroupsResult:
        collection = self._get_collection(collection_name)
        with_lookup_collection = None
        if with_lookup is not None:
            if isinstance(with_lookup, str):
                with_lookup_collection = self._get_collection(with_lookup)
            else:
                with_lookup_collection = self._get_collection(with_lookup.collection)

        return collection.recommend_groups(
            positive=positive,
            negative=negative,
            group_by=group_by,
            group_size=group_size,
            query_filter=query_filter,
            limit=limit,
            with_payload=with_payload,
            with_vectors=with_vectors,
            score_threshold=score_threshold,
            using=using,
            lookup_from_collection=self._get_collection(lookup_from.collection)
            if lookup_from
            else None,
            lookup_from_vector_name=lookup_from.vector if lookup_from else None,
            with_lookup=with_lookup,
            with_lookup_collection=with_lookup_collection,
        )

    def scroll(
        self,
        collection_name: str,
        scroll_filter: Optional[types.Filter] = None,
        limit: int = 10,
        offset: Optional[types.PointId] = None,
        with_payload: Union[bool, Sequence[str], types.PayloadSelector] = True,
        with_vectors: Union[bool, Sequence[str]] = False,
        **kwargs: Any,
    ) -> Tuple[List[types.Record], Optional[types.PointId]]:
        collection = self._get_collection(collection_name)
        return collection.scroll(
            scroll_filter=scroll_filter,
            limit=limit,
            offset=offset,
            with_payload=with_payload,
            with_vectors=with_vectors,
        )

    def count(
        self,
        collection_name: str,
        count_filter: Optional[types.Filter] = None,
        exact: bool = True,
        **kwargs: Any,
    ) -> types.CountResult:
        collection = self._get_collection(collection_name)
        return collection.count(count_filter=count_filter)

    def upsert(
        self, collection_name: str, points: types.Points, **kwargs: Any
    ) -> types.UpdateResult:
        collection = self._get_collection(collection_name)
        collection.upsert(points)
        return self._default_update_result()

    def update_vectors(
        self,
        collection_name: str,
        vectors: Sequence[types.PointVectors],
        **kwargs: Any,
    ) -> types.UpdateResult:
        collection = self._get_collection(collection_name)
        collection.update_vectors(vectors)
        return self._default_update_result()

    def delete_vectors(
        self,
        collection_name: str,
        vectors: Sequence[str],
        points: types.PointsSelector,
        **kwargs: Any,
    ) -> types.UpdateResult:
        collection = self._get_collection(collection_name)
        collection.delete_vectors(vectors, points)
        return self._default_update_result()

    def retrieve(
        self,
        collection_name: str,
        ids: Sequence[types.PointId],
        with_payload: Union[bool, Sequence[str], types.PayloadSelector] = True,
        with_vectors: Union[bool, Sequence[str]] = False,
        **kwargs: Any,
    ) -> List[types.Record]:
        collection = self._get_collection(collection_name)
        return collection.retrieve(ids, with_payload, with_vectors)

    @classmethod
    def _default_update_result(cls, operation_id: int = 0) -> types.UpdateResult:
        return types.UpdateResult(
            operation_id=operation_id,
            status=rest_models.UpdateStatus.COMPLETED,
        )

    def delete(
        self, collection_name: str, points_selector: types.PointsSelector, **kwargs: Any
    ) -> types.UpdateResult:
        collection = self._get_collection(collection_name)
        collection.delete(points_selector)
        return self._default_update_result()

    def set_payload(
        self,
        collection_name: str,
        payload: types.Payload,
        points: types.PointsSelector,
        **kwargs: Any,
    ) -> types.UpdateResult:
        collection = self._get_collection(collection_name)
        collection.set_payload(payload=payload, selector=points)
        return self._default_update_result()

    def overwrite_payload(
        self,
        collection_name: str,
        payload: types.Payload,
        points: types.PointsSelector,
        **kwargs: Any,
    ) -> types.UpdateResult:
        collection = self._get_collection(collection_name)
        collection.overwrite_payload(payload=payload, selector=points)
        return self._default_update_result()

    def delete_payload(
        self,
        collection_name: str,
        keys: Sequence[str],
        points: types.PointsSelector,
        **kwargs: Any,
    ) -> types.UpdateResult:
        collection = self._get_collection(collection_name)
        collection.delete_payload(keys=keys, selector=points)
        return self._default_update_result()

    def clear_payload(
        self, collection_name: str, points_selector: types.PointsSelector, **kwargs: Any
    ) -> types.UpdateResult:
        collection = self._get_collection(collection_name)
        collection.clear_payload(selector=points_selector)
        return self._default_update_result()

    def update_collection_aliases(
        self, change_aliases_operations: Sequence[types.AliasOperations], **kwargs: Any
    ) -> bool:
        for operation in change_aliases_operations:
            if isinstance(operation, rest_models.CreateAliasOperation):
                self._get_collection(operation.create_alias.collection_name)
                self.aliases[
                    operation.create_alias.alias_name
                ] = operation.create_alias.collection_name
            elif isinstance(operation, rest_models.DeleteAliasOperation):
                self.aliases.pop(operation.delete_alias.alias_name, None)
            elif isinstance(operation, rest_models.RenameAliasOperation):
                new_name = operation.rename_alias.new_alias_name
                old_name = operation.rename_alias.old_alias_name
                self.aliases[new_name] = self.aliases.pop(old_name)
            else:
                raise ValueError(f"Unknown operation: {operation}")
        self._save()
        return True

    def get_collection_aliases(
        self, collection_name: str, **kwargs: Any
    ) -> types.CollectionsAliasesResponse:
        return types.CollectionsAliasesResponse(
            aliases=[
                rest_models.AliasDescription(
                    alias_name=alias_name,
                    collection_name=name,
                )
                for alias_name, name in self.aliases.items()
                if name == collection_name
            ]
        )

    def get_aliases(self, **kwargs: Any) -> types.CollectionsAliasesResponse:
        return types.CollectionsAliasesResponse(
            aliases=[
                rest_models.AliasDescription(
                    alias_name=alias_name,
                    collection_name=name,
                )
                for alias_name, name in self.aliases.items()
            ]
        )

    def get_collections(self, **kwargs: Any) -> types.CollectionsResponse:
        return types.CollectionsResponse(
            collections=[
                rest_models.CollectionDescription(name=name)
                for name, _ in self.collections.items()
            ]
        )

    def get_collection(self, collection_name: str, **kwargs: Any) -> types.CollectionInfo:
        collection = self._get_collection(collection_name)
        return collection.info()

    def update_collection(self, collection_name: str, **kwargs: Any) -> bool:
        _collection = self._get_collection(collection_name)
        return False

    def _collection_path(self, collection_name: str) -> Optional[str]:
        if self.persistent:
            return os.path.join(self.location, "collection", collection_name)
        else:
            return None

    def delete_collection(self, collection_name: str, **kwargs: Any) -> bool:
        _collection = self.collections.pop(collection_name, None)
        del _collection
        self.aliases = {
            alias_name: name
            for alias_name, name in self.aliases.items()
            if name != collection_name
        }
        collection_path = self._collection_path(collection_name)
        if collection_path is not None:
            shutil.rmtree(collection_path, ignore_errors=True)
        self._save()
        return True

    def create_collection(
        self,
        collection_name: str,
        vectors_config: Union[types.VectorParams, Mapping[str, types.VectorParams]],
        init_from: Optional[types.InitFrom] = None,
        **kwargs: Any,
    ) -> bool:
        if collection_name in self.collections:
            raise ValueError(f"Collection {collection_name} already exists")
        collection_path = self._collection_path(collection_name)
        if collection_path is not None:
            os.makedirs(collection_path, exist_ok=True)

        collection = LocalCollection(
            rest_models.CreateCollection(
                vectors=vectors_config,
            ),
            location=collection_path,
        )

        self.collections[collection_name] = collection
        self._save()
        return True

    def recreate_collection(
        self,
        collection_name: str,
        vectors_config: Union[types.VectorParams, Mapping[str, types.VectorParams]],
        init_from: Optional[types.InitFrom] = None,
        **kwargs: Any,
    ) -> bool:
        self.delete_collection(collection_name)
        return self.create_collection(collection_name, vectors_config, init_from)

    def upload_records(
        self, collection_name: str, records: Iterable[types.Record], **kwargs: Any
    ) -> None:
        collection = self._get_collection(collection_name)
        collection.upsert(
            [
                rest_models.PointStruct(
                    id=record.id,
                    vector=record.vector or {},
                    payload=record.payload or {},
                )
                for record in records
            ]
        )

    def upload_collection(
        self,
        collection_name: str,
        vectors: Union[
            Dict[str, types.NumpyArray], types.NumpyArray, Iterable[types.VectorStruct]
        ],
        payload: Optional[Iterable[Dict[Any, Any]]] = None,
        ids: Optional[Iterable[types.PointId]] = None,
        **kwargs: Any,
    ) -> None:
        collection = self._get_collection(collection_name)
        if isinstance(vectors, dict) and any(isinstance(v, np.ndarray) for v in vectors.values()):
            assert (
                len(set([arr.shape[0] for arr in vectors.values()])) == 1
            ), "Each named vector should have the same number of vectors"

            num_vectors = next(iter(vectors.values())).shape[0]
            # convert Dict[str, np.ndarray] to List[Dict[str, List[float]]]
            vectors = [
                {name: vectors[name][i].tolist() for name in vectors.keys()}
                for i in range(num_vectors)
            ]

        collection.upsert(
            [
                rest_models.PointStruct(
                    id=point_id,
                    vector=(vector.tolist() if isinstance(vector, np.ndarray) else vector) or {},
                    payload=payload or {},
                )
                for (point_id, vector, payload) in zip(
                    ids or itertools.count(),
                    iter(vectors),
                    payload or itertools.cycle([{}]),
                )
            ]
        )

    def create_payload_index(
        self,
        collection_name: str,
        field_name: str,
        field_schema: Optional[types.PayloadSchemaType] = None,
        field_type: Optional[types.PayloadSchemaType] = None,
        **kwargs: Any,
    ) -> types.UpdateResult:
        logging.warning(
            "Payload indexes have no effect in the local Qdrant. Please use server Qdrant if you need payload indexes."
        )
        return self._default_update_result()

    def delete_payload_index(
        self, collection_name: str, field_name: str, **kwargs: Any
    ) -> types.UpdateResult:
        logging.warning(
            "Payload indexes have no effect in the local Qdrant. Please use server Qdrant if you need payload indexes."
        )
        return self._default_update_result()

    def list_snapshots(
        self, collection_name: str, **kwargs: Any
    ) -> List[types.SnapshotDescription]:
        return []

    def create_snapshot(
        self, collection_name: str, **kwargs: Any
    ) -> Optional[types.SnapshotDescription]:
        raise NotImplementedError(
            "Snapshots are not supported in the local Qdrant. Please use server Qdrant if you need full snapshots."
        )

    def delete_snapshot(self, collection_name: str, snapshot_name: str, **kwargs: Any) -> bool:
        raise NotImplementedError(
            "Snapshots are not supported in the local Qdrant. Please use server Qdrant if you need full snapshots."
        )

    def list_full_snapshots(self, **kwargs: Any) -> List[types.SnapshotDescription]:
        return []

    def create_full_snapshot(self, **kwargs: Any) -> types.SnapshotDescription:
        raise NotImplementedError(
            "Snapshots are not supported in the local Qdrant. Please use server Qdrant if you need full snapshots."
        )

    def delete_full_snapshot(self, snapshot_name: str, **kwargs: Any) -> bool:
        raise NotImplementedError(
            "Snapshots are not supported in the local Qdrant. Please use server Qdrant if you need full snapshots."
        )

    def recover_snapshot(self, collection_name: str, location: str, **kwargs: Any) -> bool:
        raise NotImplementedError(
            "Snapshots are not supported in the local Qdrant. Please use server Qdrant if you need full snapshots."
        )

    def lock_storage(self, reason: str, **kwargs: Any) -> types.LocksOption:
        raise NotImplementedError(
            "Locks are not supported in the local Qdrant. Please use server Qdrant if you need full snapshots."
        )

    def unlock_storage(self, **kwargs: Any) -> types.LocksOption:
        raise NotImplementedError(
            "Locks are not supported in the local Qdrant. Please use server Qdrant if you need full snapshots."
        )

    def get_locks(self, **kwargs: Any) -> types.LocksOption:
        return types.LocksOption(
            error_message=None,
            write=False,
        )
