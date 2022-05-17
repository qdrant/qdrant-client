from qdrant_client import grpc
from qdrant_client.http.models import models as http


class GrpcToRest:

    @classmethod
    def convert_condition(cls, model: grpc.Condition) -> http.Condition:
        if model.field:
            raise NotImplementedError()
        if model.filter:
            return cls.convert_filter(model.filter)
        if model.has_id:
            raise NotImplementedError()
        if model.is_empty:
            raise NotImplementedError()

    @classmethod
    def convert_filter(cls, model: grpc.Filter) -> http.Filter:
        return http.Filter(
            must=[cls.convert_condition(condition) for condition in model.must],
            should=[cls.convert_condition(condition) for condition in model.should],
            must_not=[cls.convert_condition(condition) for condition in model.must_not]
        )

    @classmethod
    def convert_range(cls, model: grpc.Range) -> http.Range:
        raise NotImplementedError()

    @classmethod
    def convert_geo_radius(cls, model: grpc.GeoRadius) -> http.GeoRadius:
        raise NotImplementedError()

    @classmethod
    def convert_collection_description(cls, model: grpc.CollectionDescription) -> http.CollectionDescription:
        raise NotImplementedError()

    @classmethod
    def convert_collection_info(cls, model: grpc.CollectionInfo) -> http.CollectionInfo:
        raise NotImplementedError()

    @classmethod
    def convert_update_result(cls, model: grpc.UpdateResult) -> http.UpdateResult:
        raise NotImplementedError()

    @classmethod
    def convert_has_id_condition(cls, model: grpc.HasIdCondition) -> http.HasIdCondition:
        raise NotImplementedError()

    @classmethod
    def convert_delete_alias(cls, model: grpc.DeleteAlias) -> http.DeleteAlias:
        raise NotImplementedError()

    @classmethod
    def convert_rename_alias(cls, model: grpc.RenameAlias) -> http.RenameAlias:
        raise NotImplementedError()

    @classmethod
    def convert_is_empty_condition(cls, model: grpc.IsEmptyCondition) -> http.IsEmptyCondition:
        raise NotImplementedError()

    @classmethod
    def convert_search_params(cls, model: grpc.SearchParams) -> http.SearchParams:
        raise NotImplementedError()

    @classmethod
    def convert_create_alias(cls, model: grpc.CreateAlias) -> http.CreateAlias:
        raise NotImplementedError()

    @classmethod
    def convert_create_collection(cls, model: grpc.CreateCollection) -> http.CreateCollection:
        raise NotImplementedError()

    @classmethod
    def convert_scored_point(cls, model: grpc.ScoredPoint) -> http.ScoredPoint:
        raise NotImplementedError()

    @classmethod
    def convert_values_count(cls, model: grpc.ValuesCount) -> http.ValuesCount:
        raise NotImplementedError()

    @classmethod
    def convert_geo_bounding_box(cls, model: grpc.GeoBoundingBox) -> http.GeoBoundingBox:
        raise NotImplementedError()

    @classmethod
    def convert_point_struct(cls, model: grpc.PointStruct) -> http.PointStruct:
        raise NotImplementedError()

    @classmethod
    def convert_hnsw_config_diff(cls, model: grpc.HnswConfigDiff) -> http.HnswConfigDiff:
        raise NotImplementedError()

    @classmethod
    def convert_field_condition(cls, model: grpc.FieldCondition) -> http.FieldCondition:
        raise NotImplementedError()

    @classmethod
    def convert_wal_config_diff(cls, model: grpc.WalConfigDiff) -> http.WalConfigDiff:
        raise NotImplementedError()

    @classmethod
    def convert_collection_config(cls, model: grpc.CollectionConfig) -> http.CollectionConfig:
        raise NotImplementedError()

    @classmethod
    def convert_collection_params(cls, model: grpc.CollectionParams) -> http.CollectionParams:
        raise NotImplementedError()

    @classmethod
    def convert_optimizers_config_diff(cls, model: grpc.OptimizersConfigDiff) -> http.OptimizersConfigDiff:
        raise NotImplementedError()

    @classmethod
    def convert_update_collection(cls, model: grpc.UpdateCollection) -> http.UpdateCollection:
        raise NotImplementedError()

    @classmethod
    def convert_geo_point(cls, model: grpc.GeoPoint) -> http.GeoPoint:
        raise NotImplementedError()


class RestToGrpc:
    @classmethod
    def convert_filter(cls, model: http.Filter) -> grpc.Filter:
        pass

    @classmethod
    def convert_range(cls, model: http.Range) -> grpc.Range:
        raise NotImplementedError()

    @classmethod
    def convert_geo_radius(cls, model: http.GeoRadius) -> grpc.GeoRadius:
        raise NotImplementedError()

    @classmethod
    def convert_collection_description(cls, model: http.CollectionDescription) -> grpc.CollectionDescription:
        raise NotImplementedError()

    @classmethod
    def convert_collection_info(cls, model: http.CollectionInfo) -> grpc.CollectionInfo:
        raise NotImplementedError()

    @classmethod
    def convert_update_result(cls, model: http.UpdateResult) -> grpc.UpdateResult:
        raise NotImplementedError()

    @classmethod
    def convert_has_id_condition(cls, model: http.HasIdCondition) -> grpc.HasIdCondition:
        raise NotImplementedError()

    @classmethod
    def convert_delete_alias(cls, model: http.DeleteAlias) -> grpc.DeleteAlias:
        raise NotImplementedError()

    @classmethod
    def convert_rename_alias(cls, model: http.RenameAlias) -> grpc.RenameAlias:
        raise NotImplementedError()

    @classmethod
    def convert_is_empty_condition(cls, model: http.IsEmptyCondition) -> grpc.IsEmptyCondition:
        raise NotImplementedError()

    @classmethod
    def convert_search_params(cls, model: http.SearchParams) -> grpc.SearchParams:
        raise NotImplementedError()

    @classmethod
    def convert_create_alias(cls, model: http.CreateAlias) -> grpc.CreateAlias:
        raise NotImplementedError()

    @classmethod
    def convert_create_collection(cls, model: http.CreateCollection) -> grpc.CreateCollection:
        raise NotImplementedError()

    @classmethod
    def convert_scored_point(cls, model: http.ScoredPoint) -> grpc.ScoredPoint:
        raise NotImplementedError()

    @classmethod
    def convert_values_count(cls, model: http.ValuesCount) -> grpc.ValuesCount:
        raise NotImplementedError()

    @classmethod
    def convert_geo_bounding_box(cls, model: http.GeoBoundingBox) -> grpc.GeoBoundingBox:
        raise NotImplementedError()

    @classmethod
    def convert_point_struct(cls, model: http.PointStruct) -> grpc.PointStruct:
        raise NotImplementedError()

    @classmethod
    def convert_hnsw_config_diff(cls, model: http.HnswConfigDiff) -> grpc.HnswConfigDiff:
        raise NotImplementedError()

    @classmethod
    def convert_field_condition(cls, model: http.FieldCondition) -> grpc.FieldCondition:
        raise NotImplementedError()

    @classmethod
    def convert_wal_config_diff(cls, model: http.WalConfigDiff) -> grpc.WalConfigDiff:
        raise NotImplementedError()

    @classmethod
    def convert_collection_config(cls, model: http.CollectionConfig) -> grpc.CollectionConfig:
        raise NotImplementedError()

    @classmethod
    def convert_collection_params(cls, model: http.CollectionParams) -> grpc.CollectionParams:
        raise NotImplementedError()

    @classmethod
    def convert_optimizers_config_diff(cls, model: http.OptimizersConfigDiff) -> grpc.OptimizersConfigDiff:
        raise NotImplementedError()

    @classmethod
    def convert_update_collection(cls, model: http.UpdateCollection) -> grpc.UpdateCollection:
        raise NotImplementedError()

    @classmethod
    def convert_geo_point(cls, model: http.GeoPoint) -> grpc.GeoPoint:
        raise NotImplementedError()

