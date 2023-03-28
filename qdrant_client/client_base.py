from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

from qdrant_client.conversions import common_types as types
from qdrant_client.http import models


class QdrantBase:
    def search_batch(
        self,
        collection_name: str,
        requests: Sequence[types.SearchRequest],
        **kwargs: Any,
    ) -> List[List[types.ScoredPoint]]:
        """Search for points in multiple collections

        Args:
            collection_name: Name of the collection
            requests: List of search requests

        Returns:
            List of search responses
        """
        raise NotImplementedError()

    def search(
        self,
        collection_name: str,
        query_vector: Union[
            types.NumpyArray, Sequence[float], Tuple[str, List[float]], types.NamedVector
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
        """Search for closest vectors in collection taking into account filtering conditions

        Args:
            collection_name: Collection to search in
            query_vector:
                Search for vectors closest to this.
                Can be either a vector itself, or a named vector, or a tuple of vector name and vector itself
            query_filter:
                - Exclude vectors which doesn't fit given conditions.
                - If `None` - search among all vectors
            search_params: Additional search params
            limit: How many results return
            offset:
                Offset of the first result to return.
                May be used to paginate results.
                Note: large offset values may cause performance issues.
            with_payload:
                - Specify which stored payload should be attached to the result.
                - If `True` - attach all payload
                - If `False` - do not attach any payload
                - If List of string - include only specified fields
                - If `PayloadSelector` - use explicit rules
            with_vectors:
                - If `True` - Attach stored vector to the search result.
                - If `False` - Do not attach vector.
                - If List of string - include only specified fields
                - Default: `False`
            score_threshold:
                Define a minimal score threshold for the result.
                If defined, less similar results will not be returned.
                Score of the returned result might be higher or smaller than the threshold depending
                on the Distance function used.
                E.g. for cosine similarity only higher scores will be returned.

        Examples:

        `Search with filter`::

            qdrant.search(
                collection_name="test_collection",
                query_vector=[1.0, 0.1, 0.2, 0.7],
                query_filter=Filter(
                    must=[
                        FieldCondition(
                            key='color',
                            range=Match(
                                value="red"
                            )
                        )
                    ]
                )
            )

        Returns:
            List of found close points with similarity scores.
        """

        raise NotImplementedError()

    def recommend_batch(
        self,
        collection_name: str,
        requests: Sequence[types.RecommendRequest],
        **kwargs: Any,
    ) -> List[List[types.ScoredPoint]]:
        """Perform multiple recommend requests in batch mode

        Args:
            collection_name: Name of the collection
            requests: List of recommend requests

        Returns:
            List of recommend responses
        """
        raise NotImplementedError()

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
        """Recommend points: search for similar points based on already stored in Qdrant examples.

        Provide IDs of the stored points, and Qdrant will perform search based on already existing vectors.
        This functionality is especially useful for recommendation over existing collection of points.

        Args:
            collection_name: Collection to search in
            positive:
                List of stored point IDs, which should be used as reference for similarity search.
                If there is only one ID provided - this request is equivalent to the regular search with vector of that point.
                If there are more than one IDs, Qdrant will attempt to search for similar to all of them.
                Recommendation for multiple vectors is experimental. Its behaviour may change in the future.
            negative:
                List of stored point IDs, which should be dissimilar to the search result.
                Negative examples is an experimental functionality. Its behaviour may change in the future.
            query_filter:
                - Exclude vectors which doesn't fit given conditions.
                - If `None` - search among all vectors
            search_params: Additional search params
            limit: How many results return
            offset:
                Offset of the first result to return.
                May be used to paginate results.
                Note: large offset values may cause performance issues.
            with_payload:
                - Specify which stored payload should be attached to the result.
                - If `True` - attach all payload
                - If `False` - do not attach any payload
                - If List of string - include only specified fields
                - If `PayloadSelector` - use explicit rules
            with_vectors:
                - If `True` - Attach stored vector to the search result.
                - If `False` - Do not attach vector.
                - If List of string - include only specified fields
                - Default: `False`
            score_threshold:
                Define a minimal score threshold for the result.
                If defined, less similar results will not be returned.
                Score of the returned result might be higher or smaller than the threshold depending
                on the Distance function used.
                E.g. for cosine similarity only higher scores will be returned.
            using:
                Name of the vectors to use for recommendations.
                If `None` - use default vectors.
            lookup_from:
                Defines a location (collection and vector field name), used to lookup vectors for recommendations.
                If `None` - use current collection will be used.

        Returns:
            List of recommended points with similarity scores.
        """
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
        """Scroll over all (matching) points in the collection.

        This method provides a way to iterate over all stored points with some optional filtering condition.
        Scroll does not apply any similarity estimations, it will return points sorted by id in ascending order.

        Args:
            collection_name: Name of the collection
            scroll_filter: If provided - only returns points matching filtering conditions
            limit: How many points to return
            offset: If provided - skip points with ids less than given `offset`
            with_payload:
                - Specify which stored payload should be attached to the result.
                - If `True` - attach all payload
                - If `False` - do not attach any payload
                - If List of string - include only specified fields
                - If `PayloadSelector` - use explicit rules
            with_vectors:
                - If `True` - Attach stored vector to the search result.
                - If `False` - Do not attach vector.
                - If List of string - include only specified fields
                - Default: `False`

        Returns:
            A pair of (List of points) and (optional offset for the next scroll request).
            If next page offset is `None` - there is no more points in the collection to scroll.
        """
        raise NotImplementedError()

    def count(
        self,
        collection_name: str,
        count_filter: Optional[types.Filter] = None,
        exact: bool = True,
        **kwargs: Any,
    ) -> types.CountResult:
        """Count points in the collection.

        Count points in the collection matching the given filter.

        Args:
            collection_name: name of the collection to count points in
            count_filter: filtering conditions
            exact:
                If `True` - provide the exact count of points matching the filter.
                If `False` - provide the approximate count of points matching the filter. Works faster.

        Returns:
            Amount of points in the collection matching the filter.
        """
        raise NotImplementedError()

    def upsert(
        self,
        collection_name: str,
        points: types.Points,
        **kwargs: Any,
    ) -> types.UpdateResult:
        """Update or insert a new point into the collection.

        If point with given ID already exists - it will be overwritten.

        Args:
            collection_name: To which collection to insert
            points: Batch or list of points to insert

        Returns:
            Operation result
        """
        raise NotImplementedError()

    def retrieve(
        self,
        collection_name: str,
        ids: Sequence[types.PointId],
        with_payload: Union[bool, Sequence[str], types.PayloadSelector] = True,
        with_vectors: Union[bool, Sequence[str]] = False,
        **kwargs: Any,
    ) -> List[types.Record]:
        """Retrieve stored points by IDs

        Args:
            collection_name: Name of the collection to lookup in
            ids: list of IDs to lookup
            with_payload:
                - Specify which stored payload should be attached to the result.
                - If `True` - attach all payload
                - If `False` - do not attach any payload
                - If List of string - include only specified fields
                - If `PayloadSelector` - use explicit rules
            with_vectors:
                - If `True` - Attach stored vector to the search result.
                - If `False` - Do not attach vector.
                - If List of string - Attach only specified vectors.
                - Default: `False`

        Returns:
            List of points
        """
        raise NotImplementedError()

    def delete(
        self,
        collection_name: str,
        points_selector: types.PointsSelector,
        **kwargs: Any,
    ) -> types.UpdateResult:
        """Deletes selected points from collection

        Args:
            collection_name: Name of the collection
            points_selector: Selects points based on list of IDs or filter
                 Example:
                    - `points=[1, 2, 3, "cd3b53f0-11a7-449f-bc50-d06310e7ed90"]`
                    - `points=Filter(must=[FieldCondition(key='rand_number', range=Range(gte=0.7))])`

        Returns:
            Operation result
        """
        raise NotImplementedError()

    def set_payload(
        self,
        collection_name: str,
        payload: types.Payload,
        points: types.PointsSelector,
        **kwargs: Any,
    ) -> types.UpdateResult:
        """Modifies payload of the specified points

        Examples:

        `Set payload`::

            # Assign payload value with key `"key"` to points 1, 2, 3.
            # If payload value with specified key already exists - it will be overwritten
            qdrant_client.set_payload(
                collection_name="test_collection",
                wait=True,
                payload={
                    "key": "value"
                },
                points=[1,2,3]
            )

        Args:
            collection_name: Name of the collection

            payload: Key-value pairs of payload to assign
            points: List of affected points, filter or points selector.
             Example:
                - `points=[1, 2, 3, "cd3b53f0-11a7-449f-bc50-d06310e7ed90"]`
                - `points=Filter(must=[FieldCondition(key='rand_number', range=Range(gte=0.7))])`

        Returns:
            Operation result
        """
        raise NotImplementedError()

    def overwrite_payload(
        self,
        collection_name: str,
        payload: types.Payload,
        points: types.PointsSelector,
        **kwargs: Any,
    ) -> types.UpdateResult:
        """Overwrites payload of the specified points
        After this operation is applied, only the specified payload will be present in the point.
        The existing payload, even if the key is not specified in the payload, will be deleted.

        Examples:

        `Set payload`::

            # Overwrite payload value with key `"key"` to points 1, 2, 3.
            # If any other valid payload value exists - it will be deleted
            qdrant_client.overwrite_payload(
                collection_name="test_collection",
                wait=True,
                payload={
                    "key": "value"
                },
                points=[1,2,3]
            )

        Args:
            collection_name: Name of the collection
            payload: Key-value pairs of payload to assign
            points: List of affected points, filter or points selector.
             Example:
                - `points=[1, 2, 3, "cd3b53f0-11a7-449f-bc50-d06310e7ed90"]`
                - `points=Filter(must=[FieldCondition(key='rand_number', range=Range(gte=0.7))])`

        Returns:
            Operation result
        """
        raise NotImplementedError()

    def delete_payload(
        self,
        collection_name: str,
        keys: Sequence[str],
        points: types.PointsSelector,
        **kwargs: Any,
    ) -> types.UpdateResult:
        """Remove values from point's payload

        Args:
            collection_name: Name of the collection
            keys: List of payload keys to remove
            points: List of affected points, filter or points selector.
                Example:
                   - `points=[1, 2, 3, "cd3b53f0-11a7-449f-bc50-d06310e7ed90"]`
                   - `points=Filter(must=[FieldCondition(key='rand_number', range=Range(gte=0.7))])`

        Returns:
            Operation result
        """
        raise NotImplementedError()

    def clear_payload(
        self,
        collection_name: str,
        points_selector: types.PointsSelector,
        **kwargs: Any,
    ) -> types.UpdateResult:
        """Delete all payload for selected points

        Args:
            collection_name: Name of the collection
            points_selector: List of affected points, filter or points selector.
                Example:
                   - `points=[1, 2, 3, "cd3b53f0-11a7-449f-bc50-d06310e7ed90"]`
                   - `points=Filter(must=[FieldCondition(key='rand_number', range=Range(gte=0.7))])`
        Returns:
            Operation result
        """
        raise NotImplementedError()

    def update_collection_aliases(
        self,
        change_aliases_operations: Sequence[types.AliasOperations],
        **kwargs: Any,
    ) -> bool:
        """Operation for performing changes of collection aliases.

        Alias changes are atomic, meaning that no collection modifications can happen between alias operations.

        Args:
            change_aliases_operations: List of operations to perform
        Returns:
            Operation result
        """
        raise NotImplementedError()

    def get_collection_aliases(
        self, collection_name: str, **kwargs: Any
    ) -> types.CollectionsAliasesResponse:
        """Get collection aliases

        Args:
            collection_name: Name of the collection

        Returns:
            Collection aliases
        """
        raise NotImplementedError()

    def get_aliases(self, **kwargs: Any) -> types.CollectionsAliasesResponse:
        """Get all aliases

        Returns:
            All aliases of all collections
        """
        raise NotImplementedError()

    def get_collections(self, **kwargs: Any) -> types.CollectionsResponse:
        """Get list name of all existing collections

        Returns:
            List of the collections
        """
        raise NotImplementedError()

    def get_collection(self, collection_name: str, **kwargs: Any) -> types.CollectionInfo:
        """Get detailed information about specified existing collection

        Args:
            collection_name: Name of the collection

        Returns:
            Detailed information about the collection
        """
        raise NotImplementedError()

    def update_collection(
        self,
        collection_name: str,
        **kwargs: Any,
    ) -> bool:
        """Update parameters of the collection

        Args:
            collection_name: Name of the collection

        Returns:
            Operation result
        """
        raise NotImplementedError()

    def delete_collection(self, collection_name: str, **kwargs: Any) -> bool:
        """Removes collection and all it's data

        Args:
            collection_name: Name of the collection to delete

        Returns:
            Operation result
        """
        raise NotImplementedError()

    def create_collection(
        self,
        collection_name: str,
        vectors_config: Union[types.VectorParams, Mapping[str, types.VectorParams]],
        **kwargs: Any,
    ) -> bool:
        """Create empty collection with given parameters

        Args:
            collection_name: Name of the collection to recreate
            vectors_config:
                Configuration of the vector storage. Vector params contains size and distance for the vector storage.
                If dict is passed, service will create a vector storage for each key in the dict.
                If single VectorParams is passed, service will create a single anonymous vector storage.

        Returns:
            Operation result
        """
        raise NotImplementedError()

    def recreate_collection(
        self,
        collection_name: str,
        vectors_config: Union[types.VectorParams, Mapping[str, types.VectorParams]],
        **kwargs: Any,
    ) -> bool:
        """Delete and create empty collection with given parameters

        Args:
            collection_name: Name of the collection to recreate
            vectors_config:
                Configuration of the vector storage. Vector params contains size and distance for the vector storage.
                If dict is passed, service will create a vector storage for each key in the dict.
                If single VectorParams is passed, service will create a single anonymous vector storage.

        Returns:
            Operation result
        """
        raise NotImplementedError()

    def upload_records(
        self,
        collection_name: str,
        records: Iterable[types.Record],
        **kwargs: Any,
    ) -> None:
        """Upload records to the collection

        Similar to `upload_collection` method, but operates with records, rather than vector and payload individually.

        Args:
            collection_name:  Name of the collection to upload to
            records: Iterator over records to upload

        """
        raise NotImplementedError()

    def upload_collection(
        self,
        collection_name: str,
        vectors: Union[types.NumpyArray, Dict[str, types.NumpyArray], Iterable[List[float]]],
        payload: Optional[Iterable[Dict[Any, Any]]] = None,
        ids: Optional[Iterable[types.PointId]] = None,
        **kwargs: Any,
    ) -> None:
        """Upload vectors and payload to the collection.
        This method will perform automatic batching of the data.
        If you need to perform a single update, use `upsert` method.
        Note: use `upload_records` method if you want to upload multiple vectors with single payload.

        Args:
            collection_name:  Name of the collection to upload to
            vectors: np.ndarray or an iterable over vectors to upload. Might be mmaped
            payload: Iterable of vectors payload, Optional, Default: None
            ids: Iterable of custom vectors ids, Optional, Default: None
        """
        raise NotImplementedError()

    def create_payload_index(
        self,
        collection_name: str,
        field_name: str,
        field_schema: Optional[types.PayloadSchemaType] = None,
        field_type: Optional[types.PayloadSchemaType] = None,
        **kwargs: Any,
    ) -> types.UpdateResult:
        """Creates index for a given payload field.
        Indexed fields allow to perform filtered search operations faster.

        Args:
            collection_name: Name of the collection
            field_name: Name of the payload field
            field_schema: Type of data to index
            field_type: Same as field_schema, but deprecated

        Returns:
            Operation Result
        """
        raise NotImplementedError()

    def delete_payload_index(
        self,
        collection_name: str,
        field_name: str,
        **kwargs: Any,
    ) -> types.UpdateResult:
        """Removes index for a given payload field.

        Args:
            collection_name: Name of the collection
            field_name: Name of the payload field

        Returns:
            Operation Result
        """
        raise NotImplementedError()

    def list_snapshots(
        self, collection_name: str, **kwargs: Any
    ) -> List[types.SnapshotDescription]:
        """List all snapshots for a given collection.

        Args:
            collection_name: Name of the collection

        Returns:
            List of snapshots
        """
        raise NotImplementedError()

    def create_snapshot(
        self, collection_name: str, **kwargs: Any
    ) -> Optional[types.SnapshotDescription]:
        """Create snapshot for a given collection.

        Args:
            collection_name: Name of the collection

        Returns:
            Snapshot description
        """
        raise NotImplementedError()

    def delete_snapshot(self, collection_name: str, snapshot_name: str, **kwargs: Any) -> bool:
        """Delete snapshot for a given collection.

        Args:
            collection_name: Name of the collection
            snapshot_name: Snapshot id

        Returns:
            True if snapshot was deleted
        """
        raise NotImplementedError()

    def list_full_snapshots(self, **kwargs: Any) -> List[types.SnapshotDescription]:
        """List all snapshots for a whole storage

        Returns:
            List of snapshots
        """
        raise NotImplementedError()

    def create_full_snapshot(self, **kwargs: Any) -> types.SnapshotDescription:
        """Create snapshot for a whole storage.

        Returns:
            Snapshot description
        """
        raise NotImplementedError()

    def delete_full_snapshot(self, snapshot_name: str, **kwargs: Any) -> bool:
        """Delete snapshot for a whole storage.

        Args:
            snapshot_name: Snapshot name

        Returns:
            True if snapshot was deleted
        """
        raise NotImplementedError()

    def recover_snapshot(
        self,
        collection_name: str,
        location: str,
        **kwargs: Any,
    ) -> bool:
        """Recover collection from snapshot.

        Args:
            collection_name: Name of the collection
            location:
                URL of the snapshot.
                Example:
                    - URL `http://localhost:8080/collections/my_collection/snapshots/my_snapshot`
                    - Local path `file:///qdrant/snapshots/test_collection-2022-08-04-10-49-10.snapshot`

        """
        raise NotImplementedError()

    def lock_storage(self, reason: str, **kwargs: Any) -> types.LocksOption:
        """Lock storage for writing."""
        raise NotImplementedError()

    def unlock_storage(self, **kwargs: Any) -> types.LocksOption:
        """Unlock storage for writing."""
        raise NotImplementedError()

    def get_locks(self, **kwargs: Any) -> types.LocksOption:
        """Get current locks state."""
        raise NotImplementedError()
