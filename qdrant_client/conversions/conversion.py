from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional

from google.protobuf.timestamp_pb2 import Timestamp
from google.protobuf.json_format import MessageToDict

try:
    from google.protobuf.pyext._message import MessageMapContainer  # type: ignore
except ImportError:
    pass

from qdrant_client import grpc as grpc
from qdrant_client.http.models import models as rest
from qdrant_client.grpc import Value, ListValue, Struct, NullValue


def json_to_value(payload: Any) -> Value:
    if payload is None:
        return Value(null_value=NullValue.NULL_VALUE)
    if isinstance(payload, bool):
        return Value(bool_value=payload)
    if isinstance(payload, int):
        return Value(integer_value=payload)
    if isinstance(payload, float):
        return Value(double_value=payload)
    if isinstance(payload, str):
        return Value(string_value=payload)
    if isinstance(payload, list):
        return Value(list_value=ListValue(values=[json_to_value(v) for v in payload]))
    if isinstance(payload, dict):
        return Value(struct_value=Struct(fields=dict((k, json_to_value(v)) for k, v in payload.items())))
    raise ValueError(f"Not supported json value: {payload}")  # pragma: no cover


def value_to_json(value: Value) -> Any:
    if isinstance(value, Value):
        value_ = MessageToDict(value, preserving_proto_field_name=False)
    else:
        value_ = value

    if "integerValue" in value_:
        # by default int are represented as string for precision
        # But in python it is OK to just use `int`
        return int(value_["integerValue"])
    if "doubleValue" in value_:
        return value_["doubleValue"]
    if "stringValue" in value_:
        return value_["stringValue"]
    if "boolValue" in value_:
        return value_["boolValue"]
    if "structValue" in value_:
        if 'fields' not in value_['structValue']:
            return {}
        return dict((key, value_to_json(val)) for key, val in value_["structValue"]['fields'].items())
    if "listValue" in value_:
        if 'values' in value_["listValue"]:
            return list(value_to_json(val) for val in value_["listValue"]['values'])
        else:
            return []
    if "nullValue" in value_:
        return None
    raise ValueError(f"Not supported value: {value_}")  # pragma: no cover


def payload_to_grpc(payload: Dict[str, Any]) -> Dict[str, Value]:
    return dict(
        (key, json_to_value(val))
        for key, val in payload.items()
    )


def grpc_to_payload(grpc: Dict[str, Value]) -> Dict[str, Any]:
    return dict(
        (key, value_to_json(val))
        for key, val in grpc.items()
    )


class GrpcToRest:

    @classmethod
    def convert_condition(cls, model: grpc.Condition) -> rest.Condition:
        name = model.WhichOneof("condition_one_of")
        val = getattr(model, name)

        if name == "field":
            return cls.convert_field_condition(val)
        if name == "filter":
            return cls.convert_filter(val)
        if name == "has_id":
            return cls.convert_has_id_condition(val)
        if name == "is_empty":
            return cls.convert_is_empty_condition(val)

        raise ValueError(f"invalid Condition model: {model}")  # pragma: no cover

    @classmethod
    def convert_filter(cls, model: grpc.Filter) -> rest.Filter:
        return rest.Filter(
            must=[cls.convert_condition(condition) for condition in model.must],
            should=[cls.convert_condition(condition) for condition in model.should],
            must_not=[cls.convert_condition(condition) for condition in model.must_not]
        )

    @classmethod
    def convert_range(cls, model: grpc.Range) -> rest.Range:
        return rest.Range(
            gt=model.gt if model.HasField("gt") else None,
            gte=model.gte if model.HasField("gte") else None,
            lt=model.lt if model.HasField("lt") else None,
            lte=model.lte if model.HasField("lte") else None,
        )

    @classmethod
    def convert_geo_radius(cls, model: grpc.GeoRadius) -> rest.GeoRadius:
        return rest.GeoRadius(
            center=cls.convert_geo_point(model.center),
            radius=model.radius
        )

    @classmethod
    def convert_collection_description(cls, model: grpc.CollectionDescription) -> rest.CollectionDescription:
        return rest.CollectionDescription(name=model.name)

    @classmethod
    def convert_collection_info(cls, model: grpc.CollectionInfo) -> rest.CollectionInfo:
        return rest.CollectionInfo(
            config=cls.convert_collection_config(model.config),
            optimizer_status=cls.convert_optimizer_status(model.optimizer_status),
            payload_schema=cls.convert_payload_schema(model.payload_schema),
            segments_count=model.segments_count,
            status=cls.convert_collection_status(model.status),
            vectors_count=model.vectors_count,
            points_count=model.points_count,
            indexed_vectors_count=model.indexed_vectors_count or 0,
        )

    @classmethod
    def convert_optimizer_status(cls, model: grpc.OptimizerStatus) -> rest.OptimizersStatus:
        if model.ok:
            return rest.OptimizersStatusOneOf.OK
        else:
            return rest.OptimizersStatusOneOf1(error=model.error)

    @classmethod
    def convert_collection_config(cls, model: grpc.CollectionConfig) -> rest.CollectionConfig:
        return rest.CollectionConfig(
            hnsw_config=cls.convert_hnsw_config_diff(model.hnsw_config),
            optimizer_config=cls.convert_optimizer_config(model.optimizer_config),
            params=cls.convert_collection_params(model.params),
            wal_config=cls.convert_wal_config(model.wal_config)
        )

    @classmethod
    def convert_hnsw_config_diff(cls, model: grpc.HnswConfigDiff) -> rest.HnswConfigDiff:
        return rest.HnswConfigDiff(
            ef_construct=model.ef_construct if model.HasField("ef_construct") else None,
            m=model.m if model.HasField("m") else None,
            full_scan_threshold=model.full_scan_threshold if model.HasField("full_scan_threshold") else None,
            max_indexing_threads=model.max_indexing_threads if model.HasField("max_indexing_threads") else None,
            on_disk=model.on_disk if model.HasField("on_disk") else None,
            payload_m=model.payload_m if model.HasField("payload_m") else None,
        )

    @classmethod
    def convert_hnsw_config(cls, model: grpc.HnswConfigDiff) -> rest.HnswConfig:
        return rest.HnswConfig(
            ef_construct=model.ef_construct if model.HasField("ef_construct") else None,
            m=model.m if model.HasField("m") else None,
            full_scan_threshold=model.full_scan_threshold if model.HasField("full_scan_threshold") else None,
            max_indexing_threads=model.max_indexing_threads if model.HasField("max_indexing_threads") else None,
            on_disk=model.on_disk if model.HasField("on_disk") else None,
            payload_m=model.payload_m if model.HasField("payload_m") else None,
        )

    @classmethod
    def convert_optimizer_config(cls, model: grpc.OptimizersConfigDiff) -> rest.OptimizersConfig:
        return rest.OptimizersConfig(
            default_segment_number=model.default_segment_number if model.HasField("default_segment_number") else None,
            deleted_threshold=model.deleted_threshold if model.HasField("deleted_threshold") else None,
            flush_interval_sec=model.flush_interval_sec if model.HasField("flush_interval_sec") else None,
            indexing_threshold=model.indexing_threshold if model.HasField("indexing_threshold") else None,
            max_optimization_threads=model.max_optimization_threads if model.HasField(
                "max_optimization_threads") else None,
            max_segment_size=model.max_segment_size if model.HasField("max_segment_size") else None,
            memmap_threshold=model.memmap_threshold if model.HasField("memmap_threshold") else None,
            vacuum_min_vector_number=model.vacuum_min_vector_number if model.HasField(
                "vacuum_min_vector_number") else None
        )

    @classmethod
    def convert_distance(cls, model: grpc.Distance) -> rest.Distance:
        if model == grpc.Distance.Cosine:
            return rest.Distance.COSINE
        elif model == grpc.Distance.Euclid:
            return rest.Distance.EUCLID
        elif model == grpc.Distance.Dot:
            return rest.Distance.DOT
        else:
            raise ValueError(f"invalid Distance model: {model}")  # pragma: no cover

    @classmethod
    def convert_wal_config(cls, model: grpc.WalConfigDiff) -> rest.WalConfig:
        return rest.WalConfig(wal_capacity_mb=model.wal_capacity_mb if model.HasField("wal_capacity_mb") else None,
                              wal_segments_ahead=model.wal_segments_ahead if model.HasField(
                                  "wal_segments_ahead") else None)

    @classmethod
    def convert_payload_schema(cls, model: Dict[str, grpc.PayloadSchemaInfo]) -> Dict[str, rest.PayloadIndexInfo]:
        return {key: cls.convert_payload_schema_info(info) for key, info in model.items()}

    @classmethod
    def convert_payload_schema_info(cls, model: grpc.PayloadSchemaInfo) -> rest.PayloadIndexInfo:
        return rest.PayloadIndexInfo(
            data_type=cls.convert_payload_schema_type(model.data_type),
            points=model.points,
        )

    @classmethod
    def convert_payload_schema_type(cls, model: grpc.PayloadSchemaType) -> rest.PayloadSchemaType:
        if model == grpc.PayloadSchemaType.Float:
            return rest.PayloadSchemaType.FLOAT
        elif model == grpc.PayloadSchemaType.Geo:
            return rest.PayloadSchemaType.GEO
        elif model == grpc.PayloadSchemaType.Integer:
            return rest.PayloadSchemaType.INTEGER
        elif model == grpc.PayloadSchemaType.Keyword:
            return rest.PayloadSchemaType.KEYWORD
        elif model == grpc.PayloadSchemaType.Text:
            return rest.PayloadSchemaType.TEXT
        else:
            raise ValueError(f"invalid PayloadSchemaType model: {model}")  # pragma: no cover

    @classmethod
    def convert_collection_status(cls, model: grpc.CollectionStatus) -> rest.CollectionStatus:
        if model == grpc.CollectionStatus.Green:
            return rest.CollectionStatus.GREEN
        elif model == grpc.CollectionStatus.Yellow:
            return rest.CollectionStatus.YELLOW
        elif model == grpc.CollectionStatus.Red:
            return rest.CollectionStatus.RED
        else:
            raise ValueError(f"invalid CollectionStatus model: {model}")  # pragma: no cover

    @classmethod
    def convert_update_result(cls, model: grpc.UpdateResult) -> rest.UpdateResult:
        return rest.UpdateResult(operation_id=model.operation_id, status=cls.convert_update_status(model.status))

    @classmethod
    def convert_update_status(cls, model: grpc.UpdateStatus) -> rest.UpdateStatus:
        if model == grpc.UpdateStatus.Acknowledged:
            return rest.UpdateStatus.ACKNOWLEDGED
        elif model == grpc.UpdateStatus.Completed:
            return rest.UpdateStatus.COMPLETED
        else:
            raise ValueError(f"invalid UpdateStatus model: {model}")  # pragma: no cover

    @classmethod
    def convert_has_id_condition(cls, model: grpc.HasIdCondition) -> rest.HasIdCondition:
        return rest.HasIdCondition(
            has_id=[cls.convert_point_id(idx) for idx in model.has_id]
        )

    @classmethod
    def convert_point_id(cls, model: grpc.PointId) -> rest.ExtendedPointId:
        name = model.WhichOneof("point_id_options")

        if name == "num":
            return model.num
        if name == "uuid":
            return model.uuid
        raise ValueError(f"invalid PointId model: {model}")  # pragma: no cover

    @classmethod
    def convert_delete_alias(cls, model: grpc.DeleteAlias) -> rest.DeleteAlias:
        return rest.DeleteAlias(alias_name=model.alias_name)

    @classmethod
    def convert_rename_alias(cls, model: grpc.RenameAlias) -> rest.RenameAlias:
        return rest.RenameAlias(old_alias_name=model.old_alias_name, new_alias_name=model.new_alias_name)

    @classmethod
    def convert_is_empty_condition(cls, model: grpc.IsEmptyCondition) -> rest.IsEmptyCondition:
        return rest.IsEmptyCondition(is_empty=rest.PayloadField(key=model.key))

    @classmethod
    def convert_search_params(cls, model: grpc.SearchParams) -> rest.SearchParams:
        return rest.SearchParams(
            hnsw_ef=model.hnsw_ef if model.HasField("hnsw_ef") else None,
            exact=model.exact if model.HasField("exact") else None,
        )

    @classmethod
    def convert_create_alias(cls, model: grpc.CreateAlias) -> rest.CreateAlias:
        return rest.CreateAlias(
            collection_name=model.collection_name,
            alias_name=model.alias_name
        )

    @classmethod
    def convert_create_collection(cls, model: grpc.CreateCollection) -> rest.CreateCollection:
        return rest.CreateCollection(
            vectors=cls.convert_vectors_config(model.vectors_config) if model.HasField("vectors_config") else None,
            collection_name=model.collection_name,
            hnsw_config=cls.convert_hnsw_config(model.hnsw_config),
            wal_config=cls.convert_wal_config(model.wal_config),
            optimizers_config=cls.convert_optimizer_config(model.optimizers_config),
            shard_number=model.shard_number
        )

    @classmethod
    def convert_scored_point(cls, model: grpc.ScoredPoint) -> rest.ScoredPoint:
        return rest.ScoredPoint(
            id=cls.convert_point_id(model.id),
            payload=cls.convert_payload(model.payload),
            score=model.score,
            vector=cls.convert_vectors(model.vectors) if model.HasField("vectors") else None,
            version=model.version,
        )

    @classmethod
    def convert_payload(cls, model: "MessageMapContainer") -> rest.Payload:
        return dict(
            (key, value_to_json(model[key]))
            for key in model
        )

    @classmethod
    def convert_values_count(cls, model: grpc.ValuesCount) -> rest.ValuesCount:
        return rest.ValuesCount(
            gt=model.gt if model.HasField("gt") else None,
            gte=model.gte if model.HasField("gte") else None,
            lt=model.lt if model.HasField("lt") else None,
            lte=model.lte if model.HasField("lte") else None,
        )

    @classmethod
    def convert_geo_bounding_box(cls, model: grpc.GeoBoundingBox) -> rest.GeoBoundingBox:
        return rest.GeoBoundingBox(
            bottom_right=cls.convert_geo_point(model.bottom_right),
            top_left=cls.convert_geo_point(model.top_left)
        )

    @classmethod
    def convert_point_struct(cls, model: grpc.PointStruct) -> rest.PointStruct:
        return rest.PointStruct(
            id=cls.convert_point_id(model.id),
            payload=cls.convert_payload(model.payload),
            vector=cls.convert_vectors(model.vectors) if model.HasField("vectors") else None,
        )

    @classmethod
    def convert_field_condition(cls, model: grpc.FieldCondition) -> rest.FieldCondition:
        geo_bounding_box = cls.convert_geo_bounding_box(model.geo_bounding_box) if model.HasField(
            'geo_bounding_box') else None
        geo_radius = cls.convert_geo_radius(model.geo_radius) if model.HasField('geo_radius') else None
        match = cls.convert_match(model.match) if model.HasField('match') else None
        range_ = cls.convert_range(model.range) if model.HasField('range') else None
        values_count = cls.convert_values_count(model.values_count) if model.HasField('values_count') else None
        return rest.FieldCondition(
            key=model.key,
            geo_bounding_box=geo_bounding_box,
            geo_radius=geo_radius,
            match=match,
            range=range_,
            values_count=values_count,
        )

    @classmethod
    def convert_match(cls, model: grpc.Match) -> rest.Match:
        name = model.WhichOneof("match_value")
        val = getattr(model, name)

        if name == "integer":
            return rest.MatchValue(value=val)
        if name == "boolean":
            return rest.MatchValue(value=val)
        if name == "keyword":
            return rest.MatchValue(value=val)
        if name == "text":
            return rest.MatchText(text=val)
        raise ValueError(f"invalid Match model: {model}")  # pragma: no cover

    @classmethod
    def convert_wal_config_diff(cls, model: grpc.WalConfigDiff) -> rest.WalConfigDiff:
        return rest.WalConfigDiff(
            wal_capacity_mb=model.wal_capacity_mb if model.HasField("wal_capacity_mb") else None,
            wal_segments_ahead=model.wal_segments_ahead if model.HasField("wal_segments_ahead") else None
        )

    @classmethod
    def convert_collection_params(cls, model: grpc.CollectionParams) -> rest.CollectionParams:
        return rest.CollectionParams(
            vectors=cls.convert_vectors_config(model.vectors_config) if model.HasField("vectors_config") else None,
            shard_number=model.shard_number,
            on_disk_payload=model.on_disk_payload,
            replication_factor=model.replication_factor if model.HasField("replication_factor") else None,
            write_consistency_factor=model.write_consistency_factor if model.HasField(
                "write_consistency_factor") else None,
        )

    @classmethod
    def convert_optimizers_config_diff(cls, model: grpc.OptimizersConfigDiff) -> rest.OptimizersConfigDiff:
        return rest.OptimizersConfigDiff(
            default_segment_number=model.default_segment_number if model.HasField("default_segment_number") else None,
            deleted_threshold=model.deleted_threshold if model.HasField("deleted_threshold") else None,
            flush_interval_sec=model.flush_interval_sec if model.HasField("flush_interval_sec") else None,
            indexing_threshold=model.indexing_threshold if model.HasField("indexing_threshold") else None,
            max_optimization_threads=model.max_optimization_threads if model.HasField(
                "max_optimization_threads") else None,
            max_segment_size=model.max_segment_size if model.HasField("max_segment_size") else None,
            memmap_threshold=model.memmap_threshold if model.HasField("memmap_threshold") else None,
            vacuum_min_vector_number=model.vacuum_min_vector_number if model.HasField(
                "vacuum_min_vector_number") else None,
        )

    @classmethod
    def convert_update_collection(cls, model: grpc.UpdateCollection) -> rest.UpdateCollection:
        return rest.UpdateCollection(
            optimizers_config=cls.convert_optimizers_config_diff(
                model.optimizers_config) if model.HasField('optimizers_config') else None
        )

    @classmethod
    def convert_geo_point(cls, model: grpc.GeoPoint) -> rest.GeoPoint:
        return rest.GeoPoint(
            lat=model.lat,
            lon=model.lon,
        )

    @classmethod
    def convert_alias_operations(cls, model: grpc.AliasOperations) -> rest.AliasOperations:
        name = model.WhichOneof("action")
        val = getattr(model, name)

        if name == "rename_alias":
            return rest.RenameAliasOperation(rename_alias=cls.convert_rename_alias(val))
        if name == "create_alias":
            return rest.CreateAliasOperation(create_alias=cls.convert_create_alias(val))
        if name == "delete_alias":
            return rest.DeleteAliasOperation(delete_alias=cls.convert_delete_alias(val))

        raise ValueError(f"invalid AliasOperations model: {model}")  # pragma: no cover

    @classmethod
    def convert_points_selector(cls, model: grpc.PointsSelector) -> rest.PointsSelector:
        name = model.WhichOneof("points_selector_one_of")
        val = getattr(model, name)

        if name == "points":
            return rest.PointIdsList(points=[
                cls.convert_point_id(point)
                for point in val.ids
            ])
        if name == "filter":
            return rest.FilterSelector(
                filter=cls.convert_filter(val)
            )
        raise ValueError(f"invalid PointsSelector model: {model}")  # pragma: no cover

    @classmethod
    def convert_with_payload_selector(cls, model: grpc.WithPayloadSelector) -> rest.WithPayloadInterface:
        name = model.WhichOneof("selector_options")
        val = getattr(model, name)

        if name == "enable":
            return val
        if name == "include":
            return list(val.fields)
        if name == "exclude":
            return rest.PayloadSelectorExclude(exclude=list(val.fields))

        raise ValueError(f"invalid WithPayloadSelector model: {model}")  # pragma: no cover

    @classmethod
    def convert_with_payload_interface(cls, model: grpc.WithPayloadSelector) -> rest.WithPayloadInterface:
        return cls.convert_with_payload_selector(model)

    @classmethod
    def convert_retrieved_point(cls, model: grpc.RetrievedPoint) -> rest.Record:
        return rest.Record(
            id=cls.convert_point_id(model.id),
            payload=cls.convert_payload(model.payload),
            vector=cls.convert_vectors(model.vectors) if model.HasField("vectors") else None,
        )

    @classmethod
    def convert_record(cls, model: grpc.RetrievedPoint) -> rest.Record:
        return cls.convert_retrieved_point(model)

    @classmethod
    def convert_count_result(cls, model: grpc.CountResult) -> rest.CountResult:
        return rest.CountResult(
            count=model.count
        )

    @classmethod
    def convert_snapshot_description(cls, model: grpc.SnapshotDescription) -> rest.SnapshotDescription:
        return rest.SnapshotDescription(
            name=model.name,
            creation_time=model.creation_time.ToDatetime().isoformat() if model.HasField("creation_time") else None,
            size=model.size,
        )

    @classmethod
    def convert_vector_params(cls, model: grpc.VectorParams) -> rest.VectorParams:
        return rest.VectorParams(
            size=model.size,
            distance=cls.convert_distance(model.distance)
        )

    @classmethod
    def convert_vectors_config(cls, model: grpc.VectorsConfig) -> rest.VectorsConfig:
        name = model.WhichOneof("config")
        val = getattr(model, name)

        if name == "params":
            return cls.convert_vector_params(val)
        if name == "params_map":
            return dict(
                (key, cls.convert_vector_params(vec_params))
                for key, vec_params in val.map.items()
            )
        raise ValueError(f"invalid VectorsConfig model: {model}")  # pragma: no cover

    @classmethod
    def convert_vector(cls, model: grpc.Vector) -> List[float]:
        return model.data[:]

    @classmethod
    def convert_named_vectors(cls, model: grpc.NamedVectors) -> Dict[str, List[float]]:
        return {
            name: cls.convert_vector(vector)
            for name, vector in model.vectors.items()
        }

    @classmethod
    def convert_vectors(cls, model: grpc.Vectors) -> rest.VectorStruct:
        name = model.WhichOneof("vectors_options")
        val = getattr(model, name)
        if name == "vector":
            return cls.convert_vector(val)
        if name == "vectors":
            return cls.convert_named_vectors(val)
        raise ValueError(f"invalid Vectors model: {model}")  # pragma: no cover

    @classmethod
    def convert_vectors_selector(cls, model: grpc.VectorsSelector) -> List[str]:
        return model.names[:]

    @classmethod
    def convert_with_vectors_selector(cls, model: grpc.WithVectorsSelector) -> rest.WithVector:
        name = model.WhichOneof("selector_options")
        val = getattr(model, name)
        if name == "enable":
            return val
        if name == "include":
            return cls.convert_vectors_selector(val)
        raise ValueError(f"invalid WithVectorsSelector model: {model}")

    @classmethod
    def convert_search_points(cls, model: grpc.SearchPoints) -> rest.SearchRequest:
        return rest.SearchRequest(
            vector=rest.NamedVector(
                name=model.vector_name,
                vector=model.vector[:]
            ),
            filter=cls.convert_filter(model.filter) if model.HasField("filter") else None,
            limit=model.limit,
            with_payload=cls.convert_with_payload_interface(
                model.with_payload) if model.HasField("with_payload") else None,
            params=cls.convert_search_params(model.params) if model.HasField("params") else None,
            score_threshold=model.score_threshold if model.HasField("score_threshold") else None,
            offset=model.offset if model.HasField("offset") else None,
            with_vector=cls.convert_with_vectors_selector(
                model.with_vectors) if model.HasField("with_vectors") else None,
        )

    @classmethod
    def convert_recommend_points(cls, model: grpc.RecommendPoints) -> rest.RecommendRequest:
        return rest.RecommendRequest(
            positive=[cls.convert_point_id(point_id) for point_id in model.positive],
            negative=[cls.convert_point_id(point_id) for point_id in model.negative],
            filter=cls.convert_filter(model.filter) if model.HasField("filter") else None,
            limit=model.limit,
            with_payload=cls.convert_with_payload_interface(model.with_payload) if model.HasField(
                "with_payload") else None,
            params=cls.convert_search_params(model.params) if model.HasField("params") else None,
            score_threshold=model.score_threshold if model.HasField("score_threshold") else None,
            offset=model.offset if model.HasField("offset") else None,
            with_vector=cls.convert_with_vectors_selector(
                model.with_vectors) if model.HasField("with_vectors") else None,
            using=model.using,
            lookup_from=cls.convert_lookup_location(model.lookup_from) if model.HasField("lookup_from") else None,
        )

    @classmethod
    def convert_tokenizer_type(cls, model: grpc.TokenizerType) -> rest.TokenizerType:
        if model == grpc.Prefix:
            return rest.TokenizerType.PREFIX
        if model == grpc.Whitespace:
            return rest.TokenizerType.WHITESPACE
        if model == grpc.Word:
            return rest.TokenizerType.WORD
        raise ValueError(f"invalid TokenizerType model: {model}")  # pragma: no cover

    @classmethod
    def convert_text_index_params(cls, model: grpc.TextIndexParams) -> rest.TextIndexParams:
        return rest.TextIndexParams(
            type="text",
            tokenizer=cls.convert_tokenizer_type(model.tokenizer),
            min_token_len=model.min_token_len if model.HasField("min_token_len") else None,
            max_token_len=model.max_token_len if model.HasField("max_token_len") else None,
            lowercase=model.lowercase if model.HasField("lowercase") else None,
        )

    @classmethod
    def convert_collection_params_diff(cls, model: grpc.CollectionParamsDiff) -> rest.CollectionParamsDiff:
        return rest.CollectionParamsDiff(
            replication_factor=model.replication_factor if model.HasField("replication_factor") else None,
            write_consistency_factor=model.write_consistency_factor if model.HasField(
                "write_consistency_factor") else None,
        )

    @classmethod
    def convert_lookup_location(cls, model: grpc.LookupLocation) -> rest.LookupLocation:
        return rest.LookupLocation(
            collection=model.collection_name,
            vector=model.vector_name if model.HasField("vector_name") else None,
        )


# ----------------------------------------
#
# ----------- REST TO gRPC ---------------
#
# ----------------------------------------

class RestToGrpc:
    @classmethod
    def convert_filter(cls, model: rest.Filter) -> grpc.Filter:
        return grpc.Filter(
            must=[cls.convert_condition(condition) for condition in model.must] if model.must is not None else None,
            must_not=[cls.convert_condition(condition) for condition in
                      model.must_not] if model.must_not is not None else None,
            should=[cls.convert_condition(condition) for condition in
                    model.should] if model.should is not None else None,
        )

    @classmethod
    def convert_range(cls, model: rest.Range) -> grpc.Range:
        return grpc.Range(
            lt=model.lt,
            gt=model.gt,
            gte=model.gte,
            lte=model.lte,
        )

    @classmethod
    def convert_geo_radius(cls, model: rest.GeoRadius) -> grpc.GeoRadius:
        return grpc.GeoRadius(
            center=cls.convert_geo_point(model.center),
            radius=model.radius
        )

    @classmethod
    def convert_collection_description(cls, model: rest.CollectionDescription) -> grpc.CollectionDescription:
        return grpc.CollectionDescription(
            name=model.name
        )

    @classmethod
    def convert_collection_info(cls, model: rest.CollectionInfo) -> grpc.CollectionInfo:
        return grpc.CollectionInfo(
            config=cls.convert_collection_config(model.config) if model.config else None,
            optimizer_status=cls.convert_optimizer_status(model.optimizer_status),
            payload_schema=cls.convert_payload_schema(
                model.payload_schema) if model.payload_schema is not None else None,
            segments_count=model.segments_count,
            status=cls.convert_collection_status(model.status),
            vectors_count=model.vectors_count,
            points_count=model.points_count,
        )

    @classmethod
    def convert_collection_status(cls, model: rest.CollectionStatus) -> grpc.CollectionStatus:
        if model == rest.CollectionStatus.RED:
            return grpc.CollectionStatus.Red
        if model == rest.CollectionStatus.YELLOW:
            return grpc.CollectionStatus.Yellow
        if model == rest.CollectionStatus.GREEN:
            return grpc.CollectionStatus.Green

        raise ValueError(f"invalid CollectionStatus model: {model}")  # pragma: no cover

    @classmethod
    def convert_optimizer_status(cls, model: rest.OptimizersStatus) -> grpc.OptimizerStatus:
        if isinstance(model, rest.OptimizersStatusOneOf):
            return grpc.OptimizerStatus(
                ok=True,
            )
        if isinstance(model, rest.OptimizersStatusOneOf1):
            return grpc.OptimizerStatus(
                ok=False,
                error=model.error
            )
        raise ValueError(f"invalid OptimizersStatus model: {model}")  # pragma: no cover

    @classmethod
    def convert_payload_schema(cls, model: Dict[str, rest.PayloadIndexInfo]) -> Dict[str, grpc.PayloadSchemaInfo]:
        return dict(
            (key, cls.convert_payload_index_info(val))
            for key, val in model.items()
        )

    @classmethod
    def convert_payload_index_info(cls, model: rest.PayloadIndexInfo) -> grpc.PayloadSchemaInfo:
        return grpc.PayloadSchemaInfo(
            data_type=cls.convert_payload_schema_type(model.data_type)
        )

    @classmethod
    def convert_payload_schema_type(cls, model: rest.PayloadSchemaType) -> grpc.PayloadSchemaType:
        if model == rest.PayloadSchemaType.KEYWORD:
            return grpc.PayloadSchemaType.Keyword
        if model == rest.PayloadSchemaType.INTEGER:
            return grpc.PayloadSchemaType.Integer
        if model == rest.PayloadSchemaType.FLOAT:
            return grpc.PayloadSchemaType.Float
        if model == rest.PayloadSchemaType.GEO:
            return grpc.PayloadSchemaType.Geo

        raise ValueError(f"invalid PayloadSchemaType model: {model}")  # pragma: no cover

    @classmethod
    def convert_update_result(cls, model: rest.UpdateResult) -> grpc.UpdateResult:
        return grpc.UpdateResult(
            operation_id=model.operation_id,
            status=cls.convert_update_stats(model.status)
        )

    @classmethod
    def convert_update_stats(cls, model: rest.UpdateStatus) -> grpc.UpdateStatus:
        if model == rest.UpdateStatus.COMPLETED:
            return grpc.UpdateStatus.Completed
        if model == rest.UpdateStatus.ACKNOWLEDGED:
            return grpc.UpdateStatus.Acknowledged

        raise ValueError(f"invalid UpdateStatus model: {model}")  # pragma: no cover

    @classmethod
    def convert_has_id_condition(cls, model: rest.HasIdCondition) -> grpc.HasIdCondition:
        return grpc.HasIdCondition(
            has_id=[cls.convert_extended_point_id(idx) for idx in model.has_id]
        )

    @classmethod
    def convert_delete_alias(cls, model: rest.DeleteAlias) -> grpc.DeleteAlias:
        return grpc.DeleteAlias(
            alias_name=model.alias_name
        )

    @classmethod
    def convert_rename_alias(cls, model: rest.RenameAlias) -> grpc.RenameAlias:
        return grpc.RenameAlias(
            old_alias_name=model.old_alias_name,
            new_alias_name=model.new_alias_name
        )

    @classmethod
    def convert_is_empty_condition(cls, model: rest.IsEmptyCondition) -> grpc.IsEmptyCondition:
        return grpc.IsEmptyCondition(
            key=model.is_empty.key
        )

    @classmethod
    def convert_search_params(cls, model: rest.SearchParams) -> grpc.SearchParams:
        return grpc.SearchParams(
            hnsw_ef=model.hnsw_ef,
            exact=model.exact,
        )

    @classmethod
    def convert_create_alias(cls, model: rest.CreateAlias) -> grpc.CreateAlias:
        return grpc.CreateAlias(
            collection_name=model.collection_name,
            alias_name=model.alias_name
        )

    @classmethod
    def convert_create_collection(cls, model: rest.CreateCollection, collection_name: str) -> grpc.CreateCollection:
        return grpc.CreateCollection(
            vectors_config=cls.convert_vectors_config(model.vectors) if model.vectors is not None else None,
            collection_name=collection_name,
            hnsw_config=cls.convert_hnsw_config_diff(model.hnsw_config) if model.hnsw_config is not None else None,
            optimizers_config=cls.convert_optimizers_config_diff(
                model.optimizers_config) if model.optimizers_config is not None else None,
            shard_number=model.shard_number,
            wal_config=cls.convert_wal_config_diff(model.wal_config) if model.wal_config is not None else None,
        )

    @classmethod
    def convert_scored_point(cls, model: rest.ScoredPoint) -> grpc.ScoredPoint:
        return grpc.ScoredPoint(
            id=cls.convert_extended_point_id(model.id),
            payload=cls.convert_payload(model.payload) if model.payload is not None else None,
            score=model.score,
            vectors=cls.convert_vector_struct(model.vector) if model.vector is not None else None,
            version=model.version
        )

    @classmethod
    def convert_values_count(cls, model: rest.ValuesCount) -> grpc.ValuesCount:
        return grpc.ValuesCount(
            lt=model.lt,
            gt=model.gt,
            gte=model.gte,
            lte=model.lte,
        )

    @classmethod
    def convert_geo_bounding_box(cls, model: rest.GeoBoundingBox) -> grpc.GeoBoundingBox:
        return grpc.GeoBoundingBox(
            top_left=cls.convert_geo_point(model.top_left),
            bottom_right=cls.convert_geo_point(model.bottom_right),
        )

    @classmethod
    def convert_point_struct(cls, model: rest.PointStruct) -> grpc.PointStruct:
        return grpc.PointStruct(
            id=cls.convert_extended_point_id(model.id),
            vectors=cls.convert_vector_struct(model.vector),
            payload=cls.convert_payload(model.payload)
        )

    @classmethod
    def convert_payload(cls, model: rest.Payload) -> Dict[str, grpc.Value]:
        return dict((key, json_to_value(val)) for key, val in model.items())

    @classmethod
    def convert_hnsw_config_diff(cls, model: rest.HnswConfigDiff) -> grpc.HnswConfigDiff:
        return grpc.HnswConfigDiff(
            ef_construct=model.ef_construct,
            full_scan_threshold=model.full_scan_threshold,
            m=model.m,
            max_indexing_threads=model.max_indexing_threads,
            on_disk=model.on_disk,
            payload_m=model.payload_m,
        )

    @classmethod
    def convert_field_condition(cls, model: rest.FieldCondition) -> grpc.FieldCondition:
        if model.match:
            return grpc.FieldCondition(
                key=model.key,
                match=cls.convert_match(model.match)
            )
        if model.range:
            return grpc.FieldCondition(
                key=model.key,
                range=cls.convert_range(model.range)
            )
        if model.geo_bounding_box:
            return grpc.FieldCondition(
                key=model.key,
                geo_bounding_box=cls.convert_geo_bounding_box(model.geo_bounding_box)
            )
        if model.geo_radius:
            return grpc.FieldCondition(
                key=model.key,
                geo_radius=cls.convert_geo_radius(model.geo_radius)
            )
        if model.values_count:
            return grpc.FieldCondition(
                key=model.key,
                values_count=cls.convert_values_count(model.values_count)
            )
        raise ValueError(f"invalid FieldCondition model: {model}")  # pragma: no cover

    @classmethod
    def convert_wal_config_diff(cls, model: rest.WalConfigDiff) -> grpc.WalConfigDiff:
        return grpc.WalConfigDiff(
            wal_capacity_mb=model.wal_capacity_mb,
            wal_segments_ahead=model.wal_segments_ahead
        )

    @classmethod
    def convert_collection_config(cls, model: rest.CollectionConfig) -> grpc.CollectionConfig:
        return grpc.CollectionConfig(
            params=cls.convert_collection_params(model.params),
            hnsw_config=cls.convert_hnsw_config(model.hnsw_config),
            optimizer_config=cls.convert_optimizers_config(model.optimizer_config),
            wal_config=cls.convert_wal_config(model.wal_config)
        )

    @classmethod
    def convert_hnsw_config(cls, model: rest.HnswConfig) -> grpc.HnswConfigDiff:
        return grpc.HnswConfigDiff(
            ef_construct=model.ef_construct,
            full_scan_threshold=model.full_scan_threshold,
            m=model.m,
            max_indexing_threads=model.max_indexing_threads,
            on_disk=model.on_disk,
            payload_m=model.payload_m,
        )

    @classmethod
    def convert_wal_config(cls, model: rest.WalConfig) -> grpc.WalConfigDiff:
        return grpc.WalConfigDiff(
            wal_capacity_mb=model.wal_capacity_mb,
            wal_segments_ahead=model.wal_segments_ahead
        )

    @classmethod
    def convert_distance(cls, model: rest.Distance) -> grpc.Distance:
        if model == rest.Distance.DOT:
            return grpc.Distance.Dot
        if model == rest.Distance.COSINE:
            return grpc.Distance.Cosine
        if model == rest.Distance.EUCLID:
            return grpc.Distance.Euclid

        raise ValueError(f"invalid Distance model: {model}")  # pragma: no cover

    @classmethod
    def convert_collection_params(cls, model: rest.CollectionParams) -> grpc.CollectionParams:
        return grpc.CollectionParams(
            vectors_config=cls.convert_vectors_config(model.vectors) if model.vectors is not None else None,
            shard_number=model.shard_number,
            on_disk_payload=model.on_disk_payload or False,
            write_consistency_factor=model.write_consistency_factor,
            replication_factor=model.replication_factor,
        )

    @classmethod
    def convert_optimizers_config(cls, model: rest.OptimizersConfig) -> grpc.OptimizersConfigDiff:
        return grpc.OptimizersConfigDiff(
            default_segment_number=model.default_segment_number,
            deleted_threshold=model.deleted_threshold,
            flush_interval_sec=model.flush_interval_sec,
            indexing_threshold=model.indexing_threshold,
            max_optimization_threads=model.max_optimization_threads,
            max_segment_size=model.max_segment_size,
            memmap_threshold=model.memmap_threshold,
            vacuum_min_vector_number=model.vacuum_min_vector_number,
        )

    @classmethod
    def convert_optimizers_config_diff(cls, model: rest.OptimizersConfigDiff) -> grpc.OptimizersConfigDiff:
        return grpc.OptimizersConfigDiff(
            default_segment_number=model.default_segment_number,
            deleted_threshold=model.deleted_threshold,
            flush_interval_sec=model.flush_interval_sec,
            indexing_threshold=model.indexing_threshold,
            max_optimization_threads=model.max_optimization_threads,
            max_segment_size=model.max_segment_size,
            memmap_threshold=model.memmap_threshold,
            vacuum_min_vector_number=model.vacuum_min_vector_number,
        )

    @classmethod
    def convert_update_collection(cls, model: rest.UpdateCollection, collection_name: str) -> grpc.UpdateCollection:
        return grpc.UpdateCollection(
            collection_name=collection_name,
            optimizers_config=cls.convert_optimizers_config_diff(
                model.optimizers_config) if model.optimizers_config is not None else None
        )

    @classmethod
    def convert_geo_point(cls, model: rest.GeoPoint) -> grpc.GeoPoint:
        return grpc.GeoPoint(
            lon=model.lon,
            lat=model.lat
        )

    @classmethod
    def convert_match(cls, model: rest.Match) -> grpc.Match:
        if isinstance(model, rest.MatchValue):
            if isinstance(model.value, bool):
                return grpc.Match(boolean=model.value)
            if isinstance(model.value, int):
                return grpc.Match(integer=model.value)
            if isinstance(model.value, str):
                return grpc.Match(keyword=model.value)
        if isinstance(model, rest.MatchText):
            return grpc.Match(text=model.text)

        raise ValueError(f"invalid Match model: {model}")  # pragma: no cover

    @classmethod
    def convert_alias_operations(cls, model: rest.AliasOperations) -> grpc.AliasOperations:
        if isinstance(model, rest.CreateAliasOperation):
            return grpc.AliasOperations(create_alias=cls.convert_create_alias(model.create_alias))
        if isinstance(model, rest.DeleteAliasOperation):
            return grpc.AliasOperations(delete_alias=cls.convert_delete_alias(model.delete_alias))
        if isinstance(model, rest.RenameAliasOperation):
            return grpc.AliasOperations(rename_alias=cls.convert_rename_alias(model.rename_alias))

        raise ValueError(f"invalid AliasOperations model: {model}")  # pragma: no cover

    @classmethod
    def convert_extended_point_id(cls, model: rest.ExtendedPointId) -> grpc.PointId:
        if isinstance(model, int):
            return grpc.PointId(num=model)
        if isinstance(model, str):
            return grpc.PointId(uuid=model)
        raise ValueError(f"invalid ExtendedPointId model: {model}")  # pragma: no cover

    @classmethod
    def convert_points_selector(cls, model: rest.PointsSelector) -> grpc.PointsSelector:
        if isinstance(model, rest.PointIdsList):
            return grpc.PointsSelector(
                points=grpc.PointsIdsList(ids=[cls.convert_extended_point_id(point) for point in model.points])
            )
        if isinstance(model, rest.FilterSelector):
            return grpc.PointsSelector(
                filter=cls.convert_filter(model.filter)
            )
        raise ValueError(f"invalid PointsSelector model: {model}")  # pragma: no cover

    @classmethod
    def convert_condition(cls, model: rest.Condition) -> grpc.Condition:
        if isinstance(model, rest.FieldCondition):
            return grpc.Condition(field=cls.convert_field_condition(model))
        if isinstance(model, rest.IsEmptyCondition):
            return grpc.Condition(is_empty=cls.convert_is_empty_condition(model))
        if isinstance(model, rest.HasIdCondition):
            return grpc.Condition(has_id=cls.convert_has_id_condition(model))
        if isinstance(model, rest.Filter):
            return grpc.Condition(filter=cls.convert_filter(model))

        raise ValueError(f"invalid Condition model: {model}")  # pragma: no cover

    @classmethod
    def convert_payload_selector(cls, model: rest.PayloadSelector) -> grpc.WithPayloadSelector:
        if isinstance(model, rest.PayloadSelectorInclude):
            return grpc.WithPayloadSelector(
                include=grpc.PayloadIncludeSelector(fields=model.include)
            )
        if isinstance(model, rest.PayloadSelectorExclude):
            return grpc.WithPayloadSelector(
                exclude=grpc.PayloadExcludeSelector(fields=model.exclude)
            )
        raise ValueError(f"invalid PayloadSelector model: {model}")  # pragma: no cover

    @classmethod
    def convert_with_payload_selector(cls, model: rest.PayloadSelector) -> grpc.WithPayloadSelector:
        return cls.convert_with_payload_interface(model)

    @classmethod
    def convert_with_payload_interface(cls, model: rest.WithPayloadInterface) -> grpc.WithPayloadSelector:
        if isinstance(model, bool):
            return grpc.WithPayloadSelector(enable=model)
        elif isinstance(model, list):
            return grpc.WithPayloadSelector(include=grpc.PayloadIncludeSelector(fields=model))
        elif isinstance(model, (
                rest.PayloadSelectorInclude,
                rest.PayloadSelectorExclude,
        )):
            return cls.convert_payload_selector(model)

        raise ValueError(f"invalid WithPayloadInterface model: {model}")  # pragma: no cover

    @classmethod
    def convert_record(cls, model: rest.Record) -> grpc.RetrievedPoint:
        return grpc.RetrievedPoint(
            id=cls.convert_extended_point_id(model.id),
            payload=cls.convert_payload(model.payload),
            vectors=cls.convert_vector_struct(model.vector) if model.vector is not None else None,
        )

    @classmethod
    def convert_retrieved_point(cls, model: rest.Record) -> grpc.RetrievedPoint:
        return cls.convert_record(model)

    @classmethod
    def convert_count_result(cls, model: rest.CountResult) -> grpc.CountResult:
        return grpc.CountResult(count=model.count)

    @classmethod
    def convert_snapshot_description(cls, model: rest.SnapshotDescription) -> grpc.SnapshotDescription:
        timestamp = Timestamp()
        timestamp.FromDatetime(datetime.fromisoformat(model.creation_time))
        return grpc.SnapshotDescription(
            name=model.name,
            creation_time=timestamp,
            size=model.size,
        )

    @classmethod
    def convert_vector_params(cls, model: rest.VectorParams) -> grpc.VectorParams:
        return grpc.VectorParams(
            size=model.size,
            distance=cls.convert_distance(model.distance)
        )

    @classmethod
    def convert_vectors_config(cls, model: rest.VectorsConfig) -> grpc.VectorsConfig:
        if isinstance(model, rest.VectorParams):
            return grpc.VectorsConfig(params=cls.convert_vector_params(model))
        elif isinstance(model, dict):
            return grpc.VectorsConfig(params_map=grpc.VectorParamsMap(map=dict(
                (key, cls.convert_vector_params(val))
                for key, val in model.items()
            )))
        else:
            raise ValueError(f"invalid VectorsConfig model: {model}")  # pragma: no cover

    @classmethod
    def convert_vector_struct(cls, model: rest.VectorStruct) -> grpc.Vectors:
        if isinstance(model, list):
            return grpc.Vectors(
                vector=grpc.Vector(data=model)
            )
        elif isinstance(model, dict):
            return grpc.Vectors(
                vectors=grpc.NamedVectors(vectors=dict(
                    (key, grpc.Vector(data=val))
                    for key, val in model.items()
                ))
            )
        else:
            raise ValueError(f"invalid VectorStruct model: {model}")  # pragma: no cover

    @classmethod
    def convert_with_vectors(cls, model: rest.WithVector) -> grpc.WithVectorsSelector:
        if isinstance(model, bool):
            return grpc.WithVectorsSelector(enable=model)
        elif isinstance(model, list):
            return grpc.WithVectorsSelector(
                include=grpc.VectorsSelector(names=model)
            )
        else:
            raise ValueError(f"invalid WithVectors model: {model}")  # pragma: no cover

    @classmethod
    def convert_batch_vector_struct(cls, model: rest.BatchVectorStruct, num_records: int) -> List[grpc.Vectors]:
        if isinstance(model, list):
            return [cls.convert_vector_struct(item) for item in model]
        elif isinstance(model, dict):
            result: List[Dict] = [{} for _ in range(num_records)]
            for key, val in model.items():
                for i, item in enumerate(val):
                    result[i][key] = item
            return [cls.convert_vector_struct(item) for item in result]
        else:
            raise ValueError(f"invalid BatchVectorStruct model: {model}")  # pragma: no cover

    @classmethod
    def convert_named_vector_struct(cls, model: rest.NamedVectorStruct) -> Tuple[List[float], Optional[str]]:
        if isinstance(model, list):
            return model, None
        elif isinstance(model, rest.NamedVector):
            return model.vector, model.name
        else:
            raise ValueError(f"invalid NamedVectorStruct model: {model}")

    @classmethod
    def convert_search_request(cls, model: rest.SearchRequest, collection_name: str) -> grpc.SearchPoints:
        vector, name = cls.convert_named_vector_struct(model.vector)

        return grpc.SearchPoints(
            collection_name=collection_name,
            vector=vector,
            filter=cls.convert_filter(model.filter) if model.filter is not None else None,
            limit=model.limit,
            with_payload=cls.convert_with_payload_interface(
                model.with_payload) if model.with_payload is not None else None,
            params=cls.convert_search_params(model.params) if model.params is not None else None,
            score_threshold=model.score_threshold,
            offset=model.offset,
            vector_name=name,
            with_vectors=cls.convert_with_vectors(model.with_vector) if model.with_vector is not None else None,
        )

    @classmethod
    def convert_search_points(cls, model: rest.SearchRequest, collection_name: str) -> grpc.SearchPoints:
        return cls.convert_search_request(model, collection_name)

    @classmethod
    def convert_recommend_request(cls, model: rest.RecommendRequest, collection_name: str) -> grpc.RecommendPoints:
        return grpc.RecommendPoints(
            collection_name=collection_name,
            positive=[cls.convert_extended_point_id(point_id) for point_id in model.positive],
            negative=[cls.convert_extended_point_id(point_id) for point_id in model.negative],
            filter=cls.convert_filter(model.filter) if model.filter is not None else None,
            limit=model.limit,
            with_payload=cls.convert_with_payload_interface(
                model.with_payload) if model.with_payload is not None else None,
            params=cls.convert_search_params(model.params) if model.params is not None else None,
            score_threshold=model.score_threshold,
            offset=model.offset,
            with_vectors=cls.convert_with_vectors(model.with_vector) if model.with_vector is not None else None,
            using=model.using,
            lookup_from=cls.convert_lookup_location(model.lookup_from) if model.lookup_from is not None else None,
        )

    @classmethod
    def convert_recommend_points(cls, model: rest.RecommendRequest, collection_name: str) -> grpc.RecommendPoints:
        return cls.convert_recommend_request(model, collection_name)

    @classmethod
    def convert_tokenizer_type(cls, model: rest.TokenizerType) -> grpc.TokenizerType:
        if model == rest.TokenizerType.WORD:
            return grpc.TokenizerType.Word
        elif model == rest.TokenizerType.WHITESPACE:
            return grpc.TokenizerType.Whitespace
        elif model == rest.TokenizerType.PREFIX:
            return grpc.TokenizerType.Prefix
        else:
            raise ValueError(f"invalid TokenizerType model: {model}")

    @classmethod
    def convert_text_index_params(cls, model: rest.TextIndexParams) -> grpc.TextIndexParams:
        return grpc.TextIndexParams(
            tokenizer=cls.convert_tokenizer_type(model.tokenizer) if model.tokenizer is not None else None,
            lowercase=model.lowercase,
            min_token_len=model.min_token_len,
            max_token_len=model.max_token_len,
        )

    @classmethod
    def convert_collection_params_diff(cls, model: rest.CollectionParamsDiff) -> grpc.CollectionParamsDiff:
        return grpc.CollectionParamsDiff(
            replication_factor=model.replication_factor,
            write_consistency_factor=model.write_consistency_factor,
        )

    @classmethod
    def convert_lookup_location(cls, model: rest.LookupLocation) -> grpc.LookupLocation:
        return grpc.LookupLocation(
            collection_name=model.collection,
            vector_name=model.vector,
        )
