from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

from qdrant_client.conversions import common_types as types
from qdrant_client.http import models


class QdrantBase:
    def __init__(self, **kwargs: Any):
        pass

    def search_batch(
        self,
        collection_name: str,
        requests: Sequence[types.SearchRequest],
        **kwargs: Any,
    ) -> List[List[types.ScoredPoint]]:
        raise NotImplementedError()

    def search(
        self,
        collection_name: str,
        query_vector: Union[
            types.NumpyArray,
            Sequence[float],
            Tuple[str, List[float]],
            types.NamedVector,
        ],
        query_filter: Optional[models.Filter] = None,
        search_params: Optional[models.SearchParams] = None,
        limit: int = 10,
        offset: int = 0,
        with_payload: Union[bool, Sequence[str], models.PayloadSelector] = True,
        with_vectors: Union[bool, Sequence[str]] = False,
        score_threshold: Optional[float] = None,
        **kwargs: Any,
    ) -> List[types.ScoredPoint]:
        raise NotImplementedError()

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
        query_filter: Optional[models.Filter] = None,
        search_params: Optional[models.SearchParams] = None,
        limit: int = 10,
        group_size: int = 1,
        with_payload: Union[bool, Sequence[str], models.PayloadSelector] = True,
        with_vectors: Union[bool, Sequence[str]] = False,
        score_threshold: Optional[float] = None,
        with_lookup: Optional[types.WithLookupInterface] = None,
        **kwargs: Any,
    ) -> types.GroupsResult:
        raise NotImplementedError()

    def recommend_batch(
        self,
        collection_name: str,
        requests: Sequence[types.RecommendRequest],
        **kwargs: Any,
    ) -> List[List[types.ScoredPoint]]:
        raise NotImplementedError()

    def recommend(
        self,
        collection_name: str,
        positive: Optional[Sequence[types.RecommendExample]] = None,
        negative: Optional[Sequence[types.RecommendExample]] = None,
        query_filter: Optional[types.Filter] = None,
        search_params: Optional[types.SearchParams] = None,
        limit: int = 10,
        offset: int = 0,
        with_payload: Union[bool, List[str], types.PayloadSelector] = True,
        with_vectors: Union[bool, List[str]] = False,
        score_threshold: Optional[float] = None,
        using: Optional[str] = None,
        lookup_from: Optional[types.LookupLocation] = None,
        strategy: Optional[types.RecommendStrategy] = None,
        **kwargs: Any,
    ) -> List[types.ScoredPoint]:
        raise NotImplementedError()

    def recommend_groups(
        self,
        collection_name: str,
        group_by: str,
        positive: Optional[Sequence[types.RecommendExample]] = None,
        negative: Optional[Sequence[types.RecommendExample]] = None,
        query_filter: Optional[models.Filter] = None,
        search_params: Optional[models.SearchParams] = None,
        limit: int = 10,
        group_size: int = 1,
        score_threshold: Optional[float] = None,
        with_payload: Union[bool, Sequence[str], models.PayloadSelector] = True,
        with_vectors: Union[bool, Sequence[str]] = False,
        using: Optional[str] = None,
        lookup_from: Optional[models.LookupLocation] = None,
        with_lookup: Optional[types.WithLookupInterface] = None,
        strategy: Optional[types.RecommendStrategy] = None,
        **kwargs: Any,
    ) -> types.GroupsResult:
        raise NotImplementedError()

    def discover(
        self,
        collection_name: str,
        target: Optional[types.RecommendExample] = None,
        context: Optional[Sequence[types.ContextExamplePair]] = None,
        query_filter: Optional[types.Filter] = None,
        search_params: Optional[types.SearchParams] = None,
        limit: int = 10,
        offset: int = 0,
        with_payload: Union[bool, List[str], types.PayloadSelector] = True,
        with_vectors: Union[bool, List[str]] = False,
        score_threshold: Optional[float] = None,
        using: Optional[str] = None,
        lookup_from: Optional[types.LookupLocation] = None,
        consistency: Optional[types.ReadConsistency] = None,
        timeout: Optional[int] = None,
        **kwargs: Any,
    ) -> List[types.ScoredPoint]:
        """
        Use context and a target to find the most similar points, constrained by the context.

        Args:
            target:
                Look for vectors closest to this.

                When using the target (with or without context), the integer part of the score represents the rank with respect to the context, while the decimal part of the score relates to the distance to the target.

            context:
                Pairs of { positive, negative } examples to constrain the search.

                When using only the context (without a target), a special search –called context search– is performed where pairs of points are used to generate a loss that guides the search towards the zone where most positive examples overlap. This means that the score minimizes the scenario of finding a point closer to a negative than to a positive part of a pair.

                Since the score of a context relates to loss, the maximum score a point can get is 0.0, and it becomes normal that many points can have a score of 0.0.

                For discovery search (when including a target), the context part of the score for each pair is calculated +1 if the point is closer to a positive than to a negative part of a pair, and -1 otherwise.

            filter:
                Look only for points which satisfies this conditions

            params:
                Additional search params

            limit:
                Max number of result to return

            offset:
                Offset of the first result to return. May be used to paginate results. Note: large offset values may cause performance issues.

            with_payload:
                Select which payload to return with the response. Default: None

            with_vector:
                Whether to return the point vector with the result?

            using:
                Define which vector to use for recommendation, if not specified - try to use default vector.

            lookup_from:
                The location used to lookup vectors. If not specified - use current collection. Note: the other collection should have the same vector size as the current collection.

            consistency:
                Read consistency of the search. Defines how many replicas should be queried before returning the result. Values:

                - int - number of replicas to query, values should present in all queried replicas
                - 'majority' - query all replicas, but return values present in the majority of replicas
                - 'quorum' - query the majority of replicas, return values present in all of them
                - 'all' - query all replicas, and return values present in all replicas

            timeout:
                Overrides global timeout for this search. Unit is seconds.

        Returns:
            List of discovered points with discovery or context scores, accordingly.
        """
        raise NotImplementedError()

    def discover_batch(
        self,
        collection_name: str,
        requests: Sequence[types.DiscoverRequest],
        **kwargs: Any,
    ) -> List[List[types.ScoredPoint]]:
        raise NotImplementedError()

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
        raise NotImplementedError()

    def count(
        self,
        collection_name: str,
        count_filter: Optional[types.Filter] = None,
        exact: bool = True,
        **kwargs: Any,
    ) -> types.CountResult:
        raise NotImplementedError()

    def upsert(
        self,
        collection_name: str,
        points: types.Points,
        **kwargs: Any,
    ) -> types.UpdateResult:
        raise NotImplementedError()

    def update_vectors(
        self,
        collection_name: str,
        points: Sequence[types.PointVectors],
        **kwargs: Any,
    ) -> types.UpdateResult:
        raise NotImplementedError()

    def delete_vectors(
        self,
        collection_name: str,
        vectors: Sequence[str],
        points: types.PointsSelector,
        **kwargs: Any,
    ) -> types.UpdateResult:
        raise NotImplementedError()

    def retrieve(
        self,
        collection_name: str,
        ids: Sequence[types.PointId],
        with_payload: Union[bool, Sequence[str], types.PayloadSelector] = True,
        with_vectors: Union[bool, Sequence[str]] = False,
        **kwargs: Any,
    ) -> List[types.Record]:
        raise NotImplementedError()

    def delete(
        self,
        collection_name: str,
        points_selector: types.PointsSelector,
        **kwargs: Any,
    ) -> types.UpdateResult:
        raise NotImplementedError()

    def set_payload(
        self,
        collection_name: str,
        payload: types.Payload,
        points: types.PointsSelector,
        **kwargs: Any,
    ) -> types.UpdateResult:
        raise NotImplementedError()

    def overwrite_payload(
        self,
        collection_name: str,
        payload: types.Payload,
        points: types.PointsSelector,
        **kwargs: Any,
    ) -> types.UpdateResult:
        raise NotImplementedError()

    def delete_payload(
        self,
        collection_name: str,
        keys: Sequence[str],
        points: types.PointsSelector,
        **kwargs: Any,
    ) -> types.UpdateResult:
        raise NotImplementedError()

    def clear_payload(
        self,
        collection_name: str,
        points_selector: types.PointsSelector,
        **kwargs: Any,
    ) -> types.UpdateResult:
        raise NotImplementedError()

    def batch_update_points(
        self,
        collection_name: str,
        update_operations: Sequence[types.UpdateOperation],
        **kwargs: Any,
    ) -> List[types.UpdateResult]:
        raise NotImplementedError()

    def update_collection_aliases(
        self,
        change_aliases_operations: Sequence[types.AliasOperations],
        **kwargs: Any,
    ) -> bool:
        raise NotImplementedError()

    def get_collection_aliases(
        self, collection_name: str, **kwargs: Any
    ) -> types.CollectionsAliasesResponse:
        raise NotImplementedError()

    def get_aliases(self, **kwargs: Any) -> types.CollectionsAliasesResponse:
        raise NotImplementedError()

    def get_collections(self, **kwargs: Any) -> types.CollectionsResponse:
        raise NotImplementedError()

    def get_collection(self, collection_name: str, **kwargs: Any) -> types.CollectionInfo:
        raise NotImplementedError()

    def update_collection(
        self,
        collection_name: str,
        **kwargs: Any,
    ) -> bool:
        raise NotImplementedError()

    def delete_collection(self, collection_name: str, **kwargs: Any) -> bool:
        raise NotImplementedError()

    def create_collection(
        self,
        collection_name: str,
        vectors_config: Union[types.VectorParams, Mapping[str, types.VectorParams]],
        **kwargs: Any,
    ) -> bool:
        raise NotImplementedError()

    def recreate_collection(
        self,
        collection_name: str,
        vectors_config: Union[types.VectorParams, Mapping[str, types.VectorParams]],
        **kwargs: Any,
    ) -> bool:
        raise NotImplementedError()

    def upload_records(
        self,
        collection_name: str,
        records: Iterable[types.Record],
        **kwargs: Any,
    ) -> None:
        raise NotImplementedError()

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
        raise NotImplementedError()

    def create_payload_index(
        self,
        collection_name: str,
        field_name: str,
        field_schema: Optional[types.PayloadSchemaType] = None,
        field_type: Optional[types.PayloadSchemaType] = None,
        **kwargs: Any,
    ) -> types.UpdateResult:
        raise NotImplementedError()

    def delete_payload_index(
        self,
        collection_name: str,
        field_name: str,
        **kwargs: Any,
    ) -> types.UpdateResult:
        raise NotImplementedError()

    def list_snapshots(
        self, collection_name: str, **kwargs: Any
    ) -> List[types.SnapshotDescription]:
        raise NotImplementedError()

    def create_snapshot(
        self, collection_name: str, **kwargs: Any
    ) -> Optional[types.SnapshotDescription]:
        raise NotImplementedError()

    def delete_snapshot(self, collection_name: str, snapshot_name: str, **kwargs: Any) -> bool:
        raise NotImplementedError()

    def list_full_snapshots(self, **kwargs: Any) -> List[types.SnapshotDescription]:
        raise NotImplementedError()

    def create_full_snapshot(self, **kwargs: Any) -> types.SnapshotDescription:
        raise NotImplementedError()

    def delete_full_snapshot(self, snapshot_name: str, **kwargs: Any) -> bool:
        raise NotImplementedError()

    def recover_snapshot(
        self,
        collection_name: str,
        location: str,
        **kwargs: Any,
    ) -> bool:
        raise NotImplementedError()

    def list_shard_snapshots(
        self, collection_name: str, shard_id: int, **kwargs: Any
    ) -> List[types.SnapshotDescription]:
        raise NotImplementedError()

    def create_shard_snapshot(
        self, collection_name: str, shard_id: int, **kwargs: Any
    ) -> Optional[types.SnapshotDescription]:
        raise NotImplementedError()

    def delete_shard_snapshot(
        self, collection_name: str, shard_id: int, snapshot_name: str, **kwargs: Any
    ) -> bool:
        raise NotImplementedError()

    def recover_shard_snapshot(
        self,
        collection_name: str,
        shard_id: int,
        location: str,
        **kwargs: Any,
    ) -> bool:
        raise NotImplementedError()

    def lock_storage(self, reason: str, **kwargs: Any) -> types.LocksOption:
        raise NotImplementedError()

    def unlock_storage(self, **kwargs: Any) -> types.LocksOption:
        raise NotImplementedError()

    def get_locks(self, **kwargs: Any) -> types.LocksOption:
        raise NotImplementedError()

    def close(self, **kwargs: Any) -> None:
        pass

    def migrate(
        self,
        dest_client: "QdrantBase",
        collection_names: Optional[List[str]] = None,
        batch_size: int = 100,
        recreate_on_collision: bool = False,
    ) -> None:
        raise NotImplementedError()
