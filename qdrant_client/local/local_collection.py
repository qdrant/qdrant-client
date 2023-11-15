import uuid
from collections import OrderedDict, defaultdict
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union, get_args

import numpy as np

from qdrant_client._pydantic_compat import construct
from qdrant_client.conversions import common_types as types
from qdrant_client.http import models
from qdrant_client.http.models.models import ExtendedPointId
from qdrant_client.local.distances import (
    DistanceOrder,
    QueryVector,
    RecoQuery,
    calculate_best_scores,
    calculate_distance,
    distance_to_order,
)
from qdrant_client.local.payload_filters import calculate_payload_mask
from qdrant_client.local.payload_value_extractor import value_by_key
from qdrant_client.local.persistence import CollectionPersistence

DEFAULT_VECTOR_NAME = ""
EPSILON = 1.1920929e-7  # https://doc.rust-lang.org/std/f32/constant.EPSILON.html
# https://github.com/qdrant/qdrant/blob/7164ac4a5987d28f1c93f5712aef8e09e7d93555/lib/segment/src/spaces/simple_avx.rs#L99C10-L99C10


class LocalCollection:
    """
    LocalCollection is a class that represents a collection of vectors in the local storage.
    """

    def __init__(
        self,
        config: models.CreateCollection,
        location: Optional[str] = None,
        force_disable_check_same_thread: bool = False,
    ) -> None:
        """
        Create or load a collection from the local storage.
        Args:
            location: path to the collection directory. If None, the collection will be created in memory.
            force_disable_check_same_thread: force disable check_same_thread for sqlite3 connection. default: False
        """
        vectors_config = config.vectors
        if isinstance(vectors_config, models.VectorParams):
            vectors_config = {DEFAULT_VECTOR_NAME: vectors_config}

        self.vectors: Dict[str, np.ndarray] = {
            name: np.zeros((0, params.size), dtype=np.float32)
            for name, params in vectors_config.items()
        }
        self.payload: List[models.Payload] = []
        self.deleted = np.zeros(0, dtype=bool)
        self.deleted_per_vector = {name: np.zeros(0, dtype=bool) for name in self.vectors.keys()}
        self.ids: Dict[models.ExtendedPointId, int] = {}  # Mapping from external id to internal id
        self.ids_inv: List[models.ExtendedPointId] = []  # Mapping from internal id to external id
        self.persistent = location is not None
        self.storage = None
        self.config = config
        if location is not None:
            self.storage = CollectionPersistence(location, force_disable_check_same_thread)
        self.load()

    def close(self) -> None:
        if self.storage is not None:
            self.storage.close()

    def load(self) -> None:
        if self.storage is not None:
            vectors = defaultdict(list)
            deleted_ids = []

            for idx, point in enumerate(self.storage.load()):
                self.ids[point.id] = idx
                self.ids_inv.append(point.id)

                vector = point.vector
                if isinstance(point.vector, list):
                    vector = {DEFAULT_VECTOR_NAME: point.vector}

                all_vector_names = list(self.vectors.keys())

                for name in all_vector_names:
                    v = vector.get(name)
                    if v is not None:
                        vectors[name].append(v)
                    else:
                        vectors[name].append(
                            np.ones(self.config.vectors[name].size, dtype=np.float32)
                        )
                        deleted_ids.append((idx, name))

                self.payload.append(point.payload or {})

            for name, named_vectors in vectors.items():
                self.vectors[name] = np.array(named_vectors)
                self.deleted_per_vector[name] = np.zeros(len(self.payload), dtype=bool)

            for idx, name in deleted_ids:
                self.deleted_per_vector[name][idx] = 1

            self.deleted = np.zeros(len(self.payload), dtype=bool)

    @classmethod
    def _resolve_query_vector_name(
        cls,
        query_vector: Union[
            types.NumpyArray,
            Sequence[float],
            Tuple[str, List[float]],
            types.NamedVector,
            RecoQuery,
            Tuple[str, RecoQuery],
        ],
    ) -> Tuple[str, QueryVector]:
        if isinstance(query_vector, tuple):
            name, vector = query_vector
            if isinstance(vector, RecoQuery):
                return name, vector
        elif isinstance(query_vector, types.NamedVector):
            name = query_vector.name
            vector = query_vector.vector
        elif isinstance(query_vector, np.ndarray):
            name = DEFAULT_VECTOR_NAME
            vector = query_vector
        elif isinstance(query_vector, list):
            name = DEFAULT_VECTOR_NAME
            vector = query_vector
        elif isinstance(query_vector, RecoQuery):
            name = DEFAULT_VECTOR_NAME
            vector = query_vector
            return name, vector
        else:
            raise ValueError(f"Unsupported vector type {type(query_vector)}")

        return name, np.array(vector)

    def get_vector_params(self, name: str) -> models.VectorParams:
        if isinstance(self.config.vectors, dict):
            if name in self.config.vectors:
                return self.config.vectors[name]
            else:
                raise ValueError(f"Vector {name} is not found in the collection")

        if isinstance(self.config.vectors, models.VectorParams):
            if name != DEFAULT_VECTOR_NAME:
                raise ValueError(f"Vector {name} is not found in the collection")

            return self.config.vectors

        raise ValueError(f"Malformed config.vectors: {self.config.vectors}")

    @classmethod
    def _check_include_pattern(cls, pattern: str, key: str) -> bool:
        """
        >>> LocalCollection._check_include_pattern('a', 'a')
        True
        >>> LocalCollection._check_include_pattern('a.b', 'b')
        False
        >>> LocalCollection._check_include_pattern('a.b', 'a.b')
        True
        >>> LocalCollection._check_include_pattern('a.b', 'a.b.c')
        True
        >>> LocalCollection._check_include_pattern('a.b[]', 'a.b[].c')
        True
        >>> LocalCollection._check_include_pattern('a.b[]', 'a.b.c')
        False
        >>> LocalCollection._check_include_pattern('a', 'a.b')
        True
        >>> LocalCollection._check_include_pattern('a.b', 'a')
        True
        >>> LocalCollection._check_include_pattern('a', 'aa.b.c')
        False
        """
        if pattern.startswith(key) or key.startswith(pattern):
            return True
        return False

    @classmethod
    def _check_exclude_pattern(cls, pattern: str, key: str) -> bool:
        if key.startswith(pattern):
            return True
        return False

    @classmethod
    def _filter_payload(
        cls, payload: Any, predicate: Callable[[str], bool], path: str = ""
    ) -> Any:
        if isinstance(payload, dict):
            res = {}
            if path != "":
                new_path = path + "."
            else:
                new_path = path

            for key, value in payload.items():
                if predicate(new_path + key):
                    res[key] = cls._filter_payload(value, predicate, new_path + key)
            return res
        elif isinstance(payload, list):
            res_array = []
            path = path + "[]"
            for idx, value in enumerate(payload):
                if predicate(path):
                    res_array.append(cls._filter_payload(value, predicate, path))
            return res_array
        else:
            return payload

    @classmethod
    def _process_payload(
        cls,
        payload: dict,
        with_payload: Union[bool, Sequence[str], types.PayloadSelector] = True,
    ) -> Optional[dict]:
        if not with_payload:
            return None

        if isinstance(with_payload, bool):
            return payload

        if isinstance(with_payload, list):
            return cls._filter_payload(
                payload,
                lambda key: any(
                    map(lambda pattern: cls._check_include_pattern(pattern, key), with_payload)  # type: ignore
                ),
            )

        if isinstance(with_payload, models.PayloadSelectorInclude):
            return cls._filter_payload(
                payload,
                lambda key: any(
                    map(
                        lambda pattern: cls._check_include_pattern(pattern, key),
                        with_payload.include,  # type: ignore
                    )
                ),
            )

        if isinstance(with_payload, models.PayloadSelectorExclude):
            return cls._filter_payload(
                payload,
                lambda key: all(
                    map(
                        lambda pattern: not cls._check_exclude_pattern(pattern, key),
                        with_payload.exclude,  # type: ignore
                    )
                ),
            )

        return payload

    def _get_payload(
        self,
        idx: int,
        with_payload: Union[bool, Sequence[str], types.PayloadSelector] = True,
    ) -> Optional[models.Payload]:
        payload = self.payload[idx]
        return self._process_payload(payload, with_payload)

    def _get_vectors(
        self, idx: int, with_vectors: Union[bool, Sequence[str]] = False
    ) -> Optional[models.VectorStruct]:
        if not with_vectors:
            return None

        vectors = {
            name: self.vectors[name][idx].tolist()
            for name in self.vectors
            if not self.deleted_per_vector[name][idx]
        }

        if isinstance(with_vectors, list):
            vectors = {name: vectors[name] for name in with_vectors if name in vectors}

        if len(vectors) == 1 and DEFAULT_VECTOR_NAME in vectors:
            return vectors[DEFAULT_VECTOR_NAME]

        return vectors

    def search(
        self,
        query_vector: Union[
            types.NumpyArray,
            Sequence[float],
            types.NamedVector,
            RecoQuery,
            Tuple[str, Union[RecoQuery, types.NumpyArray, List[float]]],
        ],
        query_filter: Optional[types.Filter] = None,
        limit: int = 10,
        offset: int = 0,
        with_payload: Union[bool, Sequence[str], types.PayloadSelector] = True,
        with_vectors: Union[bool, Sequence[str]] = False,
        score_threshold: Optional[float] = None,
    ) -> List[models.ScoredPoint]:
        payload_mask = calculate_payload_mask(
            payloads=self.payload,
            payload_filter=query_filter,
            ids_inv=self.ids_inv,
        )
        name, query_vector = self._resolve_query_vector_name(query_vector)

        result: List[models.ScoredPoint] = []

        if name not in self.vectors:
            raise ValueError(f"Vector {name} is not found in the collection")

        vectors = self.vectors[name]
        params = self.get_vector_params(name)

        if isinstance(query_vector, np.ndarray):
            scores = calculate_distance(
                query_vector, vectors[: len(self.payload)], params.distance
            )
        elif isinstance(query_vector, RecoQuery):
            scores = calculate_best_scores(
                query_vector, vectors[: len(self.payload)], params.distance
            )
        else:
            raise (ValueError(f"Unsupported query vector type {type(query_vector)}"))

        # in deleted: 1 - deleted, 0 - not deleted
        # in payload_mask: 1 - accepted, 0 - rejected
        # in mask: 1 - ok, 0 - rejected
        mask = payload_mask & ~self.deleted & ~self.deleted_per_vector[name]

        required_order = distance_to_order(params.distance)

        if required_order == DistanceOrder.BIGGER_IS_BETTER:
            order = np.argsort(scores)[::-1]
        else:
            order = np.argsort(scores)

        for idx in order:
            if len(result) >= limit + offset:
                break

            if not mask[idx]:
                continue

            score = scores[idx]
            point_id = self.ids_inv[idx]

            if score_threshold is not None:
                if required_order == DistanceOrder.BIGGER_IS_BETTER:
                    if score < score_threshold:
                        break
                else:
                    if score > score_threshold:
                        break

            scored_point = construct(
                models.ScoredPoint,
                id=point_id,
                score=float(score),
                version=0,
                payload=self._get_payload(idx, with_payload),
                vector=self._get_vectors(idx, with_vectors),
            )

            result.append(scored_point)

        return result[offset:]

    def search_groups(
        self,
        query_vector: Union[
            types.NumpyArray,
            Sequence[float],
            Tuple[str, Union[List[float], RecoQuery, types.NumpyArray]],
            types.NamedVector,
            RecoQuery,
        ],
        group_by: str,
        query_filter: Optional[models.Filter] = None,
        limit: int = 10,
        group_size: int = 1,
        with_payload: Union[bool, Sequence[str], models.PayloadSelector] = True,
        with_vectors: Union[bool, Sequence[str]] = False,
        score_threshold: Optional[float] = None,
        with_lookup: Optional[types.WithLookupInterface] = None,
        with_lookup_collection: Optional["LocalCollection"] = None,
    ) -> models.GroupsResult:
        points = self.search(
            query_vector=query_vector,
            query_filter=query_filter,
            limit=len(self.ids_inv),
            with_payload=True,
            with_vectors=with_vectors,
            score_threshold=score_threshold,
        )

        groups = OrderedDict()

        for point in points:
            if not isinstance(point.payload, dict):
                continue

            group_values = value_by_key(point.payload, group_by)
            if group_values is None:
                continue

            group_values = list(set(v for v in group_values if isinstance(v, (str, int))))

            point.payload = self._process_payload(point.payload, with_payload)

            for group_value in group_values:
                if group_value not in groups:
                    groups[group_value] = models.PointGroup(id=group_value, hits=[])

                if len(groups[group_value].hits) >= group_size:
                    continue

                groups[group_value].hits.append(point)

        groups_result: List[models.PointGroup] = list(groups.values())[:limit]

        if isinstance(with_lookup, str):
            with_lookup = models.WithLookup(
                collection=with_lookup,
                with_payload=None,
                with_vectors=None,
            )

        if with_lookup is not None and with_lookup_collection is not None:
            for group in groups_result:
                lookup = with_lookup_collection.retrieve(
                    ids=[group.id],
                    with_payload=with_lookup.with_payload,
                    with_vectors=with_lookup.with_vectors,
                )
                group.lookup = next(iter(lookup), None)

        return models.GroupsResult(groups=groups_result)

    def retrieve(
        self,
        ids: Sequence[types.PointId],
        with_payload: Union[bool, Sequence[str], types.PayloadSelector] = True,
        with_vectors: Union[bool, Sequence[str]] = False,
    ) -> List[models.Record]:
        result = []

        for point_id in ids:
            if point_id not in self.ids:
                continue

            idx = self.ids[point_id]
            if self.deleted[idx] == 1:
                continue

            result.append(
                models.Record(
                    id=point_id,
                    payload=self._get_payload(idx, with_payload),
                    vector=self._get_vectors(idx, with_vectors),
                )
            )

        return result

    def _preprocess_recommend(
        self,
        positive: Optional[Sequence[types.RecommendExample]] = None,
        negative: Optional[Sequence[types.RecommendExample]] = None,
        strategy: Optional[types.RecommendStrategy] = None,
        query_filter: Optional[types.Filter] = None,
        using: Optional[str] = None,
        lookup_from_collection: Optional["LocalCollection"] = None,
        lookup_from_vector_name: Optional[str] = None,
    ) -> Tuple[List[List[float]], List[List[float]], types.Filter]:
        collection = lookup_from_collection if lookup_from_collection is not None else self
        search_in_vector_name = using if using is not None else DEFAULT_VECTOR_NAME
        vector_name = (
            lookup_from_vector_name
            if lookup_from_vector_name is not None
            else search_in_vector_name
        )

        positive = positive if positive is not None else []
        negative = negative if negative is not None else []

        # Validate input depending on strategy
        if strategy == types.RecommendStrategy.AVERAGE_VECTOR:
            if len(positive) == 0:
                raise ValueError("Positive list is empty")
        elif strategy == types.RecommendStrategy.BEST_SCORE:
            if len(positive) == 0 and len(negative) == 0:
                raise ValueError("No positive or negative examples given")

        # Turn every example into vectors
        positive_vectors = []
        negative_vectors = []
        mentioned_ids: List[ExtendedPointId] = []

        for example in positive:
            if isinstance(example, get_args(types.PointId)):
                if example not in collection.ids:
                    raise ValueError(f"Point {example} is not found in the collection")

                idx = collection.ids[example]
                positive_vectors.append(collection.vectors[vector_name][idx])  # type: ignore
                mentioned_ids.append(example)
            else:
                positive_vectors.append(example)

        for example in negative:
            if isinstance(example, get_args(types.PointId)):
                if example not in collection.ids:
                    raise ValueError(f"Point {example} is not found in the collection")

                idx = collection.ids[example]
                negative_vectors.append(collection.vectors[vector_name][idx])  # type: ignore
                mentioned_ids.append(example)
            else:
                negative_vectors.append(example)

        # Edit query filter
        ignore_mentioned_ids = models.HasIdCondition(has_id=mentioned_ids)

        if query_filter is None:
            query_filter = models.Filter(must_not=[ignore_mentioned_ids])
        else:
            if query_filter.must_not is None:  # type: ignore
                query_filter.must_not = [ignore_mentioned_ids]
            else:
                query_filter.must_not.append(ignore_mentioned_ids)

        return positive_vectors, negative_vectors, query_filter  # type: ignore

    def _recommend_average(
        self,
        positive_vectors: Optional[Sequence[List[float]]] = None,
        negative_vectors: Optional[Sequence[List[float]]] = None,
    ) -> types.NumpyArray:
        positive_vectors = positive_vectors if positive_vectors is not None else []
        negative_vectors = negative_vectors if negative_vectors is not None else []

        # Validate input
        if len(positive_vectors) == 0:
            raise ValueError("Positive list is empty")

        positive_vectors_np = np.stack(positive_vectors)
        negative_vectors_np = np.stack(negative_vectors) if len(negative_vectors) > 0 else None

        mean_positive_vector = np.mean(positive_vectors_np, axis=0)

        if negative_vectors_np is not None:
            vector = (
                mean_positive_vector + mean_positive_vector - np.mean(negative_vectors_np, axis=0)
            )
        else:
            vector = mean_positive_vector

        return vector

    def recommend(
        self,
        positive: Optional[Sequence[types.RecommendExample]] = None,
        negative: Optional[Sequence[types.RecommendExample]] = None,
        query_filter: Optional[types.Filter] = None,
        limit: int = 10,
        offset: int = 0,
        with_payload: Union[bool, List[str], types.PayloadSelector] = True,
        with_vectors: Union[bool, List[str]] = False,
        score_threshold: Optional[float] = None,
        using: Optional[str] = None,
        lookup_from_collection: Optional["LocalCollection"] = None,
        lookup_from_vector_name: Optional[str] = None,
        strategy: Optional[types.RecommendStrategy] = None,
    ) -> List[models.ScoredPoint]:
        strategy = strategy if strategy is not None else types.RecommendStrategy.AVERAGE_VECTOR

        positive_vectors, negative_vectors, edited_query_filter = self._preprocess_recommend(
            positive,
            negative,
            strategy,
            query_filter,
            using,
            lookup_from_collection,
            lookup_from_vector_name,
        )

        if strategy == types.RecommendStrategy.AVERAGE_VECTOR:
            query_vector = self._recommend_average(
                positive_vectors,
                negative_vectors,
            )
        elif strategy == types.RecommendStrategy.BEST_SCORE:
            query_vector = RecoQuery(
                positive=positive_vectors,
                negative=negative_vectors,
            )
        else:
            raise ValueError(
                f"strategy `{strategy}` is not a valid strategy, choose one from {types.RecommendStrategy}"
            )

        search_in_vector_name = using if using is not None else DEFAULT_VECTOR_NAME

        return self.search(
            query_vector=(search_in_vector_name, query_vector),
            query_filter=edited_query_filter,
            limit=limit,
            offset=offset,
            with_payload=with_payload,
            with_vectors=with_vectors,
            score_threshold=score_threshold,
        )

    def recommend_groups(
        self,
        group_by: str,
        positive: Optional[Sequence[types.RecommendExample]] = None,
        negative: Optional[Sequence[types.RecommendExample]] = None,
        query_filter: Optional[models.Filter] = None,
        limit: int = 10,
        group_size: int = 1,
        score_threshold: Optional[float] = None,
        with_payload: Union[bool, Sequence[str], models.PayloadSelector] = True,
        with_vectors: Union[bool, Sequence[str]] = False,
        using: Optional[str] = None,
        lookup_from_collection: Optional["LocalCollection"] = None,
        lookup_from_vector_name: Optional[str] = None,
        with_lookup: Optional[types.WithLookupInterface] = None,
        with_lookup_collection: Optional["LocalCollection"] = None,
        strategy: Optional[types.RecommendStrategy] = None,
    ) -> types.GroupsResult:
        strategy = strategy if strategy is not None else types.RecommendStrategy.AVERAGE_VECTOR

        positive_vectors, negative_vectors, edited_query_filter = self._preprocess_recommend(
            positive,
            negative,
            strategy,
            query_filter,
            using,
            lookup_from_collection,
            lookup_from_vector_name,
        )

        if strategy == types.RecommendStrategy.AVERAGE_VECTOR:
            query_vector = self._recommend_average(
                positive_vectors,
                negative_vectors,
            )
        elif strategy == types.RecommendStrategy.BEST_SCORE:
            query_vector = RecoQuery(
                positive=positive_vectors,
                negative=negative_vectors,
            )
        else:
            raise ValueError(
                f"strategy `{strategy}` is not a valid strategy, choose one from {types.RecommendStrategy}"
            )

        search_in_vector_name = using if using is not None else DEFAULT_VECTOR_NAME

        return self.search_groups(
            query_vector=(search_in_vector_name, query_vector),
            query_filter=edited_query_filter,
            group_by=group_by,
            group_size=group_size,
            limit=limit,
            with_payload=with_payload,
            with_vectors=with_vectors,
            score_threshold=score_threshold,
            with_lookup=with_lookup,
            with_lookup_collection=with_lookup_collection,
        )

    @classmethod
    def _universal_id(cls, point_id: models.ExtendedPointId) -> Tuple[str, int]:
        if isinstance(point_id, str):
            return point_id, 0
        elif isinstance(point_id, int):
            return "", point_id
        raise TypeError(f"Incompatible point id type: {type(point_id)}")

    def scroll(
        self,
        scroll_filter: Optional[types.Filter] = None,
        limit: int = 10,
        offset: Optional[types.PointId] = None,
        with_payload: Union[bool, Sequence[str], types.PayloadSelector] = True,
        with_vectors: Union[bool, Sequence[str]] = False,
    ) -> Tuple[List[types.Record], Optional[types.PointId]]:
        if len(self.ids) == 0:
            return [], None

        sorted_ids = sorted(self.ids.items(), key=lambda x: self._universal_id(x[0]))

        result: List[types.Record] = []

        payload_mask = calculate_payload_mask(
            payloads=self.payload,
            payload_filter=scroll_filter,
            ids_inv=self.ids_inv,
        )

        mask = payload_mask & ~self.deleted

        for point_id, idx in sorted_ids:
            if offset is not None and self._universal_id(point_id) < self._universal_id(offset):
                continue

            if len(result) >= limit + 1:
                break

            if not mask[idx]:
                continue

            result.append(
                models.Record(
                    id=point_id,
                    payload=self._get_payload(idx, with_payload),
                    vector=self._get_vectors(idx, with_vectors),
                )
            )

        if len(result) > limit:
            return result[:limit], result[limit].id
        else:
            return result, None

    def count(self, count_filter: Optional[types.Filter] = None) -> models.CountResult:
        payload_mask = calculate_payload_mask(
            payloads=self.payload,
            payload_filter=count_filter,
            ids_inv=self.ids_inv,
        )
        mask = payload_mask & ~self.deleted
        return models.CountResult(count=np.count_nonzero(mask))

    def _update_point(self, point: models.PointStruct) -> None:
        idx = self.ids[point.id]
        self.payload[idx] = point.payload or {}

        if isinstance(point.vector, list):
            vectors = {DEFAULT_VECTOR_NAME: point.vector}
        else:
            vectors = point.vector

        for vector_name, named_vectors in self.vectors.items():
            vector = vectors.get(vector_name)
            if vector is not None:
                params = self.get_vector_params(vector_name)
                if params.distance == models.Distance.COSINE:
                    norm = np.linalg.norm(vector)
                    vector = np.array(vector) / norm if norm > EPSILON else vector
                self.vectors[vector_name][idx] = vector
                self.deleted_per_vector[vector_name][idx] = 0
            else:
                self.deleted_per_vector[vector_name][idx] = 1

        self.deleted[idx] = 0

    def _add_point(self, point: models.PointStruct) -> None:
        idx = len(self.ids)
        self.ids[point.id] = idx
        self.ids_inv.append(point.id)
        self.payload.append(point.payload or {})
        self.deleted = np.append(self.deleted, 0)

        if isinstance(point.vector, list):
            vectors = {DEFAULT_VECTOR_NAME: point.vector}
        else:
            vectors = point.vector

        for vector_name, named_vectors in self.vectors.items():
            vector = vectors.get(vector_name)
            if named_vectors.shape[0] <= idx:
                named_vectors = np.resize(named_vectors, (idx * 2 + 1, named_vectors.shape[1]))

            if vector is None:
                # Add fake vector and mark as removed
                fake_vector = np.ones(named_vectors.shape[1])
                named_vectors[idx] = fake_vector
                self.deleted_per_vector[vector_name] = np.append(
                    self.deleted_per_vector[vector_name], 1
                )
            else:
                vector_np = np.array(vector)
                params = self.get_vector_params(vector_name)
                if params.distance == models.Distance.COSINE:
                    norm = np.linalg.norm(vector_np)
                    vector_np = vector_np / norm if norm > EPSILON else vector_np
                named_vectors[idx] = vector_np
                self.vectors[vector_name] = named_vectors
                self.deleted_per_vector[vector_name] = np.append(
                    self.deleted_per_vector[vector_name], 0
                )

    def _upsert_point(self, point: models.PointStruct) -> None:
        if isinstance(point.id, str):
            # try to parse as UUID
            try:
                _uuid = uuid.UUID(point.id)
            except ValueError as e:
                raise ValueError(f"Point id {point.id} is not a valid UUID") from e

        if point.id in self.ids:
            self._update_point(point)
        else:
            self._add_point(point)

        if self.storage is not None:
            self.storage.persist(point)

    def upsert(self, points: Union[List[models.PointStruct], models.Batch]) -> None:
        if isinstance(points, list):
            for point in points:
                self._upsert_point(point)
        elif isinstance(points, models.Batch):
            batch = points
            if isinstance(batch.vectors, list):
                vectors = {DEFAULT_VECTOR_NAME: batch.vectors}
            else:
                vectors = batch.vectors

            for idx, point_id in enumerate(batch.ids):
                payload = None
                if batch.payloads is not None:
                    payload = batch.payloads[idx]

                vector = {name: v[idx] for name, v in vectors.items()}

                self._upsert_point(
                    models.PointStruct(
                        id=point_id,
                        payload=payload,
                        vector=vector,
                    )
                )
        else:
            raise ValueError(f"Unsupported type: {type(points)}")

    def _update_named_vectors(self, idx: int, vectors: Dict[str, List[float]]) -> None:
        for vector_name, vector in vectors.items():
            self.vectors[vector_name][idx] = np.array(vector)

    def update_vectors(self, points: Sequence[types.PointVectors]) -> None:
        for point in points:
            point_id = point.id
            idx = self.ids[point_id]
            vector_struct = point.vector
            if isinstance(vector_struct, list):
                fixed_vectors = {DEFAULT_VECTOR_NAME: vector_struct}
            else:
                fixed_vectors = vector_struct
            self._update_named_vectors(idx, fixed_vectors)
            self._persist_by_id(point_id)

    def delete_vectors(
        self,
        vectors: Sequence[str],
        selector: Union[
            models.Filter,
            List[models.ExtendedPointId],
            models.FilterSelector,
            models.PointIdsList,
        ],
    ) -> None:
        ids = self._selector_to_ids(selector)
        for point_id in ids:
            idx = self.ids[point_id]
            for vector_name in vectors:
                self.deleted_per_vector[vector_name][idx] = 1
            self._persist_by_id(point_id)

    def _delete_ids(self, ids: List[types.PointId]) -> None:
        for point_id in ids:
            idx = self.ids[point_id]
            self.deleted[idx] = 1

        if self.storage is not None:
            for point_id in ids:
                self.storage.delete(point_id)

    def _filter_to_ids(self, delete_filter: types.Filter) -> List[models.ExtendedPointId]:
        mask = calculate_payload_mask(
            payloads=self.payload,
            payload_filter=delete_filter,
            ids_inv=self.ids_inv,
        )
        mask = mask & ~self.deleted
        ids = [point_id for point_id, idx in self.ids.items() if mask[idx]]
        return ids

    def _selector_to_ids(
        self,
        selector: Union[
            models.Filter,
            List[models.ExtendedPointId],
            models.FilterSelector,
            models.PointIdsList,
        ],
    ) -> List[models.ExtendedPointId]:
        if isinstance(selector, list):
            return selector
        elif isinstance(selector, models.Filter):
            return self._filter_to_ids(selector)
        elif isinstance(selector, models.PointIdsList):
            return selector.points
        elif isinstance(selector, models.FilterSelector):
            return self._filter_to_ids(selector.filter)
        else:
            raise ValueError(f"Unsupported selector type: {type(selector)}")

    def delete(
        self,
        selector: Union[
            models.Filter,
            List[models.ExtendedPointId],
            models.FilterSelector,
            models.PointIdsList,
        ],
    ) -> None:
        ids = self._selector_to_ids(selector)
        self._delete_ids(ids)

    def _persist_by_id(self, point_id: models.ExtendedPointId) -> None:
        if self.storage is not None:
            idx = self.ids[point_id]
            point = models.PointStruct(
                id=point_id,
                payload=self._get_payload(idx, with_payload=True),
                vector=self._get_vectors(idx, with_vectors=True),
            )
            self.storage.persist(point)

    def set_payload(
        self,
        payload: models.Payload,
        selector: Union[
            models.Filter,
            List[models.ExtendedPointId],
            models.FilterSelector,
            models.PointIdsList,
        ],
    ) -> None:
        ids = self._selector_to_ids(selector)
        for point_id in ids:
            idx = self.ids[point_id]
            self.payload[idx] = {
                **(self.payload[idx] or {}),
                **payload,
            }
            self._persist_by_id(point_id)

    def overwrite_payload(
        self,
        payload: models.Payload,
        selector: Union[
            models.Filter,
            List[models.ExtendedPointId],
            models.FilterSelector,
            models.PointIdsList,
        ],
    ) -> None:
        ids = self._selector_to_ids(selector)
        for point_id in ids:
            idx = self.ids[point_id]
            self.payload[idx] = payload or {}
            self._persist_by_id(point_id)

    def delete_payload(
        self,
        keys: Sequence[str],
        selector: Union[
            models.Filter,
            List[models.ExtendedPointId],
            models.FilterSelector,
            models.PointIdsList,
        ],
    ) -> None:
        ids = self._selector_to_ids(selector)
        for point_id in ids:
            idx = self.ids[point_id]
            for key in keys:
                if key in self.payload[idx]:
                    self.payload[idx].pop(key)
            self._persist_by_id(point_id)

    def clear_payload(
        self,
        selector: Union[
            models.Filter,
            List[models.ExtendedPointId],
            models.FilterSelector,
            models.PointIdsList,
        ],
    ) -> None:
        ids = self._selector_to_ids(selector)
        for point_id in ids:
            idx = self.ids[point_id]
            self.payload[idx] = {}
            self._persist_by_id(point_id)

    def batch_update_points(
        self,
        update_operations: Sequence[types.UpdateOperation],
    ) -> None:
        for update_op in update_operations:
            if isinstance(update_op, models.UpsertOperation):
                if isinstance(update_op.upsert, models.PointsBatch):
                    self.upsert(update_op.upsert.batch)
                elif isinstance(update_op.upsert, models.PointsList):
                    self.upsert(update_op.upsert.points)
                else:
                    raise ValueError(f"Unsupported upsert type: {type(update_op.upsert)}")
            elif isinstance(update_op, models.DeleteOperation):
                self.delete(update_op.delete)
            elif isinstance(update_op, models.SetPayloadOperation):
                points_selector = update_op.set_payload.points or update_op.set_payload.filter
                self.set_payload(update_op.set_payload.payload, points_selector)
            elif isinstance(update_op, models.OverwritePayloadOperation):
                points_selector = (
                    update_op.overwrite_payload.points or update_op.overwrite_payload.filter
                )
                self.overwrite_payload(update_op.overwrite_payload.payload, points_selector)
            elif isinstance(update_op, models.DeletePayloadOperation):
                points_selector = (
                    update_op.delete_payload.points or update_op.delete_payload.filter
                )
                self.delete_payload(update_op.delete_payload.keys, points_selector)
            elif isinstance(update_op, models.ClearPayloadOperation):
                self.clear_payload(update_op.clear_payload)
            elif isinstance(update_op, models.UpdateVectorsOperation):
                self.update_vectors(update_op.update_vectors.points)
            elif isinstance(update_op, models.DeleteVectorsOperation):
                points_selector = (
                    update_op.delete_vectors.points or update_op.delete_vectors.filter
                )
                self.delete_vectors(update_op.delete_vectors.vector, points_selector)
            else:
                raise ValueError(f"Unsupported update operation: {type(update_op)}")

    def info(self) -> models.CollectionInfo:
        return models.CollectionInfo(
            status=models.CollectionStatus.GREEN,
            optimizer_status=models.OptimizersStatusOneOf.OK,
            vectors_count=self.count().count * len(self.vectors),
            indexed_vectors_count=0,  # LocalCollection does not do indexing
            points_count=self.count().count,
            segments_count=1,
            payload_schema={},
            config=models.CollectionConfig(
                params=models.CollectionParams(
                    vectors=self.config.vectors,
                    shard_number=self.config.shard_number,
                    replication_factor=self.config.replication_factor,
                    write_consistency_factor=self.config.write_consistency_factor,
                    on_disk_payload=self.config.on_disk_payload,
                ),
                hnsw_config=models.HnswConfig(
                    m=16,
                    ef_construct=100,
                    full_scan_threshold=10000,
                ),
                wal_config=models.WalConfig(
                    wal_capacity_mb=32,
                    wal_segments_ahead=0,
                ),
                optimizer_config=models.OptimizersConfig(
                    deleted_threshold=0.2,
                    vacuum_min_vector_number=1000,
                    default_segment_number=0,
                    indexing_threshold=20000,
                    flush_interval_sec=5,
                    max_optimization_threads=1,
                ),
                quantization_config=None,
            ),
        )
