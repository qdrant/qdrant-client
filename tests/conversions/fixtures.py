import datetime
from typing import List

from google.protobuf.message import Message
from google.protobuf.timestamp_pb2 import Timestamp

from qdrant_client import grpc as grpc
from qdrant_client.conversions.conversion import payload_to_grpc
from qdrant_client.grpc import SparseIndices

point_id = grpc.PointId(num=1)
point_id_1 = grpc.PointId(num=2)
point_id_2 = grpc.PointId(uuid="f9bcf279-5e66-40f7-856b-3a9d9b6617ee")

has_id_condition = grpc.HasIdCondition(
    has_id=[
        point_id,
        point_id_1,
        point_id_2,
    ]
)

is_empty = grpc.IsEmptyCondition(key="my.field")
is_null = grpc.IsNullCondition(key="my.field")

match_keyword = grpc.Match(keyword="hello")
match_integer = grpc.Match(integer=42)
match_bool = grpc.Match(boolean=True)
match_text = grpc.Match(text="hello")
match_keywords = grpc.Match(keywords=grpc.RepeatedStrings(strings=["hello", "world"]))
match_integers = grpc.Match(integers=grpc.RepeatedIntegers(integers=[1, 2, 3]))

match_except_keywords = grpc.Match(
    except_keywords=grpc.RepeatedStrings(strings=["hello", "world"])
)
match_except_integers = grpc.Match(except_integers=grpc.RepeatedIntegers(integers=[1, 2, 3]))

field_condition_match = grpc.FieldCondition(key="match_field", match=match_keyword)

field_condition_match_keywords = grpc.FieldCondition(key="match_field", match=match_keywords)
field_condition_match_integers = grpc.FieldCondition(key="match_field", match=match_integers)

field_condition_match_except_keywords = grpc.FieldCondition(
    key="match_field", match=match_except_keywords
)
field_condition_match_except_integers = grpc.FieldCondition(
    key="match_field", match=match_except_integers
)

range_ = grpc.Range(
    lt=1.0,
    lte=2.0,
    gt=3.0,
    gte=4.0,
)

timestamp = Timestamp(seconds=12345678, nanos=123456000)

datetime_range = grpc.DatetimeRange(
    lt=timestamp,
    lte=timestamp,
    gt=timestamp,
    gte=timestamp,
)

field_condition_range = grpc.FieldCondition(key="match_field", range=range_)
field_condition_datetime_range = grpc.FieldCondition(
    key="match_field", datetime_range=datetime_range
)

geo_point = grpc.GeoPoint(lon=12.123, lat=78.212)

geo_radius = grpc.GeoRadius(center=geo_point, radius=10.0)

field_condition_geo_radius = grpc.FieldCondition(key="match_field", geo_radius=geo_radius)

geo_bounding_box = grpc.GeoBoundingBox(top_left=geo_point, bottom_right=geo_point)

field_condition_geo_bounding_box = grpc.FieldCondition(
    key="match_field", geo_bounding_box=geo_bounding_box
)

values_count = grpc.ValuesCount(
    lt=1,
    gt=2,
    gte=3,
    lte=4,
)

field_condition_values_count = grpc.FieldCondition(key="match_field", values_count=values_count)

condition_has_id = grpc.Condition(has_id=has_id_condition)
condition_is_empty = grpc.Condition(is_empty=is_empty)
condition_is_null = grpc.Condition(is_null=is_null)

condition_field_match = grpc.Condition(field=field_condition_match)
condition_range = grpc.Condition(field=field_condition_range)
condition_geo_radius = grpc.Condition(field=field_condition_geo_radius)
condition_geo_bounding_box = grpc.Condition(field=field_condition_geo_bounding_box)
condition_values_count = grpc.Condition(field=field_condition_values_count)

condition_keywords = grpc.Condition(field=field_condition_match_keywords)
condition_integers = grpc.Condition(field=field_condition_match_integers)

condition_except_keywords = grpc.Condition(field=field_condition_match_except_keywords)
condition_except_integers = grpc.Condition(field=field_condition_match_except_integers)

nested = grpc.NestedCondition(
    key="a.b.c", filter=grpc.Filter(must=[grpc.Condition(field=field_condition_range)])
)

condition_nested = grpc.Condition(nested=nested)

filter_nested = grpc.Filter(must=[condition_nested])

filter_ = grpc.Filter(
    must=[
        condition_has_id,
        condition_is_empty,
        condition_is_null,
        condition_keywords,
        condition_integers,
        condition_except_keywords,
        condition_except_integers,
    ],
    should=[
        condition_field_match,
        condition_nested,
    ],
    must_not=[
        grpc.Condition(filter=grpc.Filter(must=[grpc.Condition(field=field_condition_range)]))
    ],
    min_should=grpc.MinShould(
        conditions=[
            condition_has_id,
            condition_is_empty,
            condition_except_keywords,
            condition_except_integers,
        ],
        min_count=3,
    ),
)

vector_param = grpc.VectorParams(
    size=100,
    distance=grpc.Distance.Dot,
)

vector_param_1 = grpc.VectorParams(
    size=100,
    distance=grpc.Distance.Euclid,
)

vector_param_2 = grpc.VectorParams(
    size=100,
    distance=grpc.Distance.Manhattan,
)

vector_param_with_hnsw = grpc.VectorParams(
    size=100,
    distance=grpc.Distance.Cosine,
    hnsw_config=grpc.HnswConfigDiff(
        ef_construct=1000,
    ),
    on_disk=True,
    datatype=grpc.Datatype.Float32,
)

vector_param_with_multivector = grpc.VectorParams(
    size=100,
    distance=grpc.Distance.Cosine,
    hnsw_config=grpc.HnswConfigDiff(
        ef_construct=1000,
    ),
    on_disk=True,
    datatype=grpc.Datatype.Float16,
    multivector_config=grpc.MultiVectorConfig(comparator=grpc.MultiVectorComparator.MaxSim),
)

product_quantizations = [
    grpc.QuantizationConfig(product=grpc.ProductQuantization(compression=ratio, always_ram=False))
    for ratio in [
        grpc.CompressionRatio.x4,
        grpc.CompressionRatio.x8,
        grpc.CompressionRatio.x16,
        grpc.CompressionRatio.x32,
        grpc.CompressionRatio.x64,
    ]
]

scalar_quantization = grpc.ScalarQuantization(
    type=grpc.QuantizationType.Int8,
    quantile=0.99,
    always_ram=True,
)

binary_quantization = grpc.BinaryQuantization(
    always_ram=True,
)

vector_param_with_quant = grpc.VectorParams(
    size=100,
    distance=grpc.Distance.Cosine,
    quantization_config=grpc.QuantizationConfig(scalar=scalar_quantization),
    datatype=grpc.Datatype.Uint8,
)

single_vector_config = grpc.VectorsConfig(params=vector_param)

multiple_vector_config = grpc.VectorsConfig(
    params_map=grpc.VectorParamsMap(map={"text_vector": vector_param})
)

sparse_vector_config = grpc.SparseVectorConfig(
    map={"sparse": grpc.SparseVectorParams(index=grpc.SparseIndexConfig(full_scan_threshold=1212))}
)
collection_params = grpc.CollectionParams(
    vectors_config=single_vector_config,
    shard_number=10,
    on_disk_payload=True,
)

collection_params_2 = grpc.CollectionParams(
    vectors_config=multiple_vector_config,
    replication_factor=2,
    write_consistency_factor=1,
    read_fan_out_factor=2,
    sparse_vectors_config=sparse_vector_config,
)

hnsw_config = grpc.HnswConfigDiff(
    m=16,
    ef_construct=100,
    full_scan_threshold=10000,
    max_indexing_threads=0,
    on_disk=False,
)

hnsw_config_2 = grpc.HnswConfigDiff(
    m=16,
    ef_construct=100,
    full_scan_threshold=10000,
    max_indexing_threads=2,
    on_disk=True,
    payload_m=32,
)

optimizer_config = grpc.OptimizersConfigDiff(
    deleted_threshold=0.2,
    vacuum_min_vector_number=10000,
    default_segment_number=5,
    max_segment_size=200000,
    memmap_threshold=50000,
    indexing_threshold=10000,
    flush_interval_sec=10,
    max_optimization_threads=0,
)

optimizer_config_half = grpc.OptimizersConfigDiff(
    deleted_threshold=0.2,
    vacuum_min_vector_number=10000,
    default_segment_number=5,
    max_segment_size=200000,
)

wal_config = grpc.WalConfigDiff(wal_capacity_mb=32, wal_segments_ahead=2)

collection_config = grpc.CollectionConfig(
    params=collection_params,
    hnsw_config=hnsw_config,
    optimizer_config=optimizer_config,
    wal_config=wal_config,
)

payload_value = {
    "int": 1,
    "float": 0.23,
    "keyword": "hello world",
    "bool": True,
    "null": None,
    "dict": {"a": 1, "b": "bbb"},
    "list": [1, 2, 3, 5, 6],
    "list_with_dict": [{}, {}, {}, []],
    "empty_list": [],
}

payload = payload_to_grpc({"payload": payload_value})

single_vector = grpc.Vectors(vector=grpc.Vector(data=[1.0, 2.0, 3.0, 4.0]))
multi_vector = grpc.Vectors(
    vector=grpc.Vector(data=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], vectors_count=2)
)
order_value_int = grpc.OrderValue(int=42)
order_value_float = grpc.OrderValue(float=42.0)
scored_point = grpc.ScoredPoint(
    id=point_id, payload=payload, score=0.99, vectors=single_vector, version=12
)
scored_point_order_value_int = grpc.ScoredPoint(
    id=point_id,
    payload=payload,
    score=0.99,
    vectors=single_vector,
    version=12,
    order_value=order_value_int,
)
scored_point_order_value_float = grpc.ScoredPoint(
    id=point_id,
    payload=payload,
    score=0.99,
    vectors=single_vector,
    version=12,
    order_value=order_value_int,
)
scored_point_multivector = grpc.ScoredPoint(
    id=point_id,
    payload=payload,
    score=0.99,
    vectors=multi_vector,
    version=12,
    order_value=order_value_float,
)
create_alias = grpc.CreateAlias(collection_name="col1", alias_name="col2")

quantization_search_params = grpc.QuantizationSearchParams(
    ignore=False,
    rescore=True,
    oversampling=10,
)

search_params = grpc.SearchParams(
    hnsw_ef=128,
)

search_params_2 = grpc.SearchParams(
    exact=True,
    indexed_only=True,
)

search_params_3 = grpc.SearchParams(
    exact=True,
    quantization=quantization_search_params,
)

rename_alias = grpc.RenameAlias(old_alias_name="col2", new_alias_name="col3")

collection_status = grpc.CollectionStatus.Yellow
collection_status_green = grpc.CollectionStatus.Green
collection_status_error = grpc.CollectionStatus.Red
collection_status_grey = grpc.CollectionStatus.Grey

optimizer_status = grpc.OptimizerStatus(ok=True)
optimizer_status_error = grpc.OptimizerStatus(ok=False, error="Error!")

payload_schema_keyword = grpc.PayloadSchemaInfo(data_type=grpc.PayloadSchemaType.Keyword, points=0)
payload_schema_integer = grpc.PayloadSchemaInfo(data_type=grpc.PayloadSchemaType.Integer, points=0)
payload_schema_float = grpc.PayloadSchemaInfo(data_type=grpc.PayloadSchemaType.Float, points=0)
payload_schema_geo = grpc.PayloadSchemaInfo(data_type=grpc.PayloadSchemaType.Geo, points=0)
payload_schema_text = grpc.PayloadSchemaInfo(data_type=grpc.PayloadSchemaType.Text, points=0)
payload_schema_bool = grpc.PayloadSchemaInfo(data_type=grpc.PayloadSchemaType.Bool, points=0)
payload_schema_datetime = grpc.PayloadSchemaInfo(
    data_type=grpc.PayloadSchemaType.Datetime, points=0
)

text_index_params_1 = grpc.TextIndexParams(
    tokenizer=grpc.TokenizerType.Prefix,
    lowercase=True,
    min_token_len=2,
    max_token_len=10,
)

text_index_params_2 = grpc.TextIndexParams(
    tokenizer=grpc.TokenizerType.Whitespace,
    lowercase=False,
    max_token_len=10,
)

text_index_params_3 = grpc.TextIndexParams(
    tokenizer=grpc.TokenizerType.Word,
    lowercase=True,
    min_token_len=2,
)

text_index_params_4 = grpc.TextIndexParams(tokenizer=grpc.TokenizerType.Multilingual)

integer_index_params_0 = grpc.IntegerIndexParams(lookup=True, range=False)
integer_index_params_1 = grpc.IntegerIndexParams(lookup=False, range=True)
integer_index_params_2 = grpc.IntegerIndexParams(lookup=True, range=True)


payload_schema_text_prefix = grpc.PayloadSchemaInfo(
    data_type=grpc.PayloadSchemaType.Text,
    params=grpc.PayloadIndexParams(text_index_params=text_index_params_1),
    points=0,
)
payload_schema_text_whitespace = grpc.PayloadSchemaInfo(
    data_type=grpc.PayloadSchemaType.Text,
    params=grpc.PayloadIndexParams(text_index_params=text_index_params_2),
    points=0,
)
payload_schema_text_word = grpc.PayloadSchemaInfo(
    data_type=grpc.PayloadSchemaType.Text,
    params=grpc.PayloadIndexParams(text_index_params=text_index_params_3),
    points=0,
)

payload_schema_text_multilingual = grpc.PayloadSchemaInfo(
    data_type=grpc.PayloadSchemaType.Text,
    params=grpc.PayloadIndexParams(text_index_params=text_index_params_4),
    points=0,
)

payload_schema_integer_lookup = grpc.PayloadSchemaInfo(
    data_type=grpc.PayloadSchemaType.Integer,
    params=grpc.PayloadIndexParams(integer_index_params=integer_index_params_0),
    points=0,
)

payload_schema_integer_range = grpc.PayloadSchemaInfo(
    data_type=grpc.PayloadSchemaType.Integer,
    params=grpc.PayloadIndexParams(integer_index_params=integer_index_params_1),
    points=0,
)

payload_schema_integer_lookup_and_range = grpc.PayloadSchemaInfo(
    data_type=grpc.PayloadSchemaType.Integer,
    params=grpc.PayloadIndexParams(integer_index_params=integer_index_params_2),
    points=0,
)

collection_info_grey = grpc.CollectionInfo(
    status=collection_status_grey,
    optimizer_status=optimizer_status_error,
    # vectors_count=100000,
    points_count=100000,
    segments_count=6,
    config=collection_config,
    payload_schema={},
)

collection_info_ok = grpc.CollectionInfo(
    status=collection_status_green,
    optimizer_status=optimizer_status,
    vectors_count=100000,
    points_count=100000,
    segments_count=6,
    config=collection_config,
    payload_schema={
        "keyword_field": payload_schema_keyword,
        "integer_field": payload_schema_integer,
        "float_field": payload_schema_float,
        "geo_field": payload_schema_geo,
        "text_field": payload_schema_text,
        "text_field_prefix": payload_schema_text_prefix,
        "text_field_whitespace": payload_schema_text_whitespace,
        "text_field_word": payload_schema_text_word,
        "text_field_multilingual": payload_schema_text_multilingual,
        "bool_field": payload_schema_bool,
        "datetime_field": payload_schema_datetime,
        "integer_lookup": payload_schema_integer_lookup,
        "integer_range": payload_schema_integer_range,
        "integer_lookup_and_range": payload_schema_integer_lookup_and_range,
    },
)

collection_info = grpc.CollectionInfo(
    status=collection_status,
    optimizer_status=optimizer_status_error,
    vectors_count=100000,
    points_count=100000,
    segments_count=6,
    config=collection_config,
    payload_schema={
        "keyword_field": payload_schema_keyword,
        "integer_field": payload_schema_integer,
        "float_field": payload_schema_float,
        "geo_field": payload_schema_geo,
        "text_field": payload_schema_text,
        "text_field_prefix": payload_schema_text_prefix,
        "text_field_whitespace": payload_schema_text_whitespace,
        "text_field_word": payload_schema_text_word,
        "text_field_multilingual": payload_schema_text_multilingual,
        "bool_field": payload_schema_bool,
        "datetime_field": payload_schema_datetime,
        "integer_lookup": payload_schema_integer_lookup,
        "integer_range": payload_schema_integer_range,
        "integer_lookup_and_range": payload_schema_integer_lookup_and_range,
    },
)

collection_info_red = grpc.CollectionInfo(
    status=collection_status_error,
    optimizer_status=optimizer_status_error,
    vectors_count=100000,
    points_count=100000,
    segments_count=6,
    config=collection_config,
    payload_schema={
        "keyword_field": payload_schema_keyword,
        "integer_field": payload_schema_integer,
        "float_field": payload_schema_float,
        "geo_field": payload_schema_geo,
        "text_field": payload_schema_text,
        "text_field_prefix": payload_schema_text_prefix,
        "text_field_whitespace": payload_schema_text_whitespace,
        "text_field_word": payload_schema_text_word,
        "text_field_multilingual": payload_schema_text_multilingual,
        "bool_field": payload_schema_bool,
        "datetime_field": payload_schema_datetime,
        "integer_lookup": payload_schema_integer_lookup,
        "integer_range": payload_schema_integer_range,
        "integer_lookup_and_range": payload_schema_integer_lookup_and_range,
    },
)
quantization_config = grpc.QuantizationConfig(
    scalar=scalar_quantization,
)

binary_quantization_config = grpc.QuantizationConfig(
    binary=binary_quantization,
)


sparse_vector_params = grpc.SparseVectorParams(
    index=grpc.SparseIndexConfig(
        full_scan_threshold=1000,
        on_disk=True,
    ),
    modifier=grpc.Modifier.Idf,
)

sparse_vector_params_datatype = grpc.SparseVectorParams(
    index=grpc.SparseIndexConfig(
        full_scan_threshold=1000,
        on_disk=True,
        datatype=grpc.Datatype.Float16,
    ),
    modifier=grpc.Modifier.Idf,
)

sparse_vector_config = grpc.SparseVectorConfig(
    map={
        "sparse": sparse_vector_params,
        "sparse_float16": sparse_vector_params_datatype,
    }
)

update_status = grpc.UpdateStatus.Acknowledged

update_result = grpc.UpdateResult(operation_id=201, status=update_status)

update_status_completed = grpc.UpdateStatus.Completed

update_result_completed = grpc.UpdateResult(operation_id=201, status=update_status_completed)

delete_alias = grpc.DeleteAlias(alias_name="col3")

point_struct = grpc.PointStruct(
    id=point_id_1,
    vectors=grpc.Vectors(vector=grpc.Vector(data=[1.0, 2.0, -1.0, -0.2])),
    payload=payload_to_grpc({"my_payload": payload_value}),
)

many_vectors = grpc.Vectors(
    vectors=grpc.NamedVectors(
        vectors={
            "image": grpc.Vector(data=[1.0, 2.0, -1.0, -0.2]),
            "text": grpc.Vector(data=[1.0, 2.0, -1.0, -0.2]),
            "sparse": grpc.Vector(
                data=[1.0, 2.0, -1.0, -0.2], indices=SparseIndices(data=[1, 2, 3])
            ),
            "multi": grpc.Vector(data=[1.0, 2.0, 3.0, 4.0], vectors_count=2),
        }
    )
)

point_struct_many = grpc.PointStruct(
    id=point_id_1,
    vectors=many_vectors,
    payload=payload_to_grpc({"my_payload": payload_value}),
)

collection_description = grpc.CollectionDescription(name="my_col")

quantization_config_diff_disabled = grpc.QuantizationConfigDiff(disabled=grpc.Disabled())

quantization_config_diff_scalar = grpc.QuantizationConfigDiff(scalar=scalar_quantization)

quantization_config_diff_product = grpc.QuantizationConfigDiff(
    product=product_quantizations[0].product
)

update_collection = grpc.UpdateCollection(
    collection_name="my_col3",
    optimizers_config=optimizer_config,
    hnsw_config=hnsw_config,
    quantization_config=quantization_config_diff_disabled,
)

collections_params_diff = grpc.CollectionParamsDiff(
    replication_factor=2,
    write_consistency_factor=2,
    on_disk_payload=True,
)

vector_params_diff = grpc.VectorParamsDiff(
    hnsw_config=hnsw_config,
    quantization_config=quantization_config_diff_product,
    on_disk=True,
)

vector_config_diff_map = grpc.VectorsConfigDiff(
    params_map=grpc.VectorParamsDiffMap(
        map={
            "image": vector_params_diff,
        }
    )
)

vector_config_diff = grpc.VectorsConfigDiff(
    params=vector_params_diff,
)

update_collection_2 = grpc.UpdateCollection(
    collection_name="my_col3",
    optimizers_config=optimizer_config,
    hnsw_config=hnsw_config,
    quantization_config=quantization_config_diff_scalar,
    vectors_config=vector_config_diff,
)

update_collection_3 = grpc.UpdateCollection(
    collection_name="my_col3",
    optimizers_config=optimizer_config,
    hnsw_config=hnsw_config,
    quantization_config=quantization_config_diff_product,
    params=collections_params_diff,
    vectors_config=vector_config_diff_map,
)

points_ids_list = grpc.PointsIdsList(ids=[point_id, point_id_2, point_id_2])

points_selector_list = grpc.PointsSelector(points=points_ids_list)
points_selector_filter = grpc.PointsSelector(filter=filter_)

alias_description = grpc.AliasDescription(collection_name="my_col4", alias_name="col4")
alias_operations_create = grpc.AliasOperations(create_alias=create_alias)
alias_operations_rename = grpc.AliasOperations(rename_alias=rename_alias)
alias_operations_delete = grpc.AliasOperations(delete_alias=delete_alias)

with_payload_bool = grpc.WithPayloadSelector(enable=True)
with_payload_include = grpc.WithPayloadSelector(
    include=grpc.PayloadIncludeSelector(fields=["color", "price"])
)
with_payload_exclude = grpc.WithPayloadSelector(
    exclude=grpc.PayloadExcludeSelector(fields=["color", "price"])
)

retrieved_point = grpc.RetrievedPoint(
    id=point_id_1,
    payload=payload_to_grpc({"key": payload_value}),
    vectors=single_vector,
)

retrieved_point_with_order_value = grpc.RetrievedPoint(
    id=point_id_1,
    payload=payload_to_grpc({"key": payload_value}),
    vectors=single_vector,
    order_value=order_value_int,
)


count_result = grpc.CountResult(count=5)

timestamp = Timestamp()
timestamp.FromDatetime(datetime.datetime.now())

snapshot_description = grpc.SnapshotDescription(
    name="my_snapshot", creation_time=timestamp, size=100500
)

vector_config = grpc.VectorsConfig(
    params_map=grpc.VectorParamsMap(
        map={
            "image": vector_param,
            "text": grpc.VectorParams(
                size=123,
                distance=grpc.Distance.Cosine,
            ),
        }
    )
)


shard_key_selector = grpc.ShardKeySelector(
    shard_keys=[
        grpc.ShardKey(number=123),
    ]
)

shard_key_selector_2 = grpc.ShardKeySelector(
    shard_keys=[
        grpc.ShardKey(number=123),
        grpc.ShardKey(keyword="abc"),
    ]
)

search_points = grpc.SearchPoints(
    collection_name="collection-123",
    vector=[1.0, 2.0, 3.0, 5.0],
    filter=filter_,
    limit=100,
    with_payload=with_payload_bool,
    params=search_params,
    score_threshold=0.123,
    offset=10,
    vector_name="abc",
    with_vectors=grpc.WithVectorsSelector(include=grpc.VectorsSelector(names=["abc", "def"])),
    shard_key_selector=shard_key_selector,
)

search_points_all_vectors = grpc.SearchPoints(
    collection_name="collection-123",
    vector=[1.0, 2.0, 3.0, 5.0],
    filter=filter_,
    limit=100,
    with_payload=with_payload_bool,
    params=search_params,
    score_threshold=0.123,
    offset=10,
    vector_name="abc",
    with_vectors=grpc.WithVectorsSelector(enable=True),
    shard_key_selector=shard_key_selector_2,
)

lookup_location_1 = grpc.LookupLocation(
    collection_name="collection-123",
)

lookup_location_2 = grpc.LookupLocation(
    collection_name="collection-123",
    vector_name="vector-123",
)

query_points = grpc.QueryPoints(
    collection_name="collection-123",
    prefetch=[grpc.PrefetchQuery(using="cba")],
    query=grpc.Query(nearest=grpc.VectorInput(dense=grpc.DenseVector(data=[0.1, 0.2, 0.3]))),
    lookup_from=lookup_location_1,
    filter=filter_,
    limit=100,
    with_payload=with_payload_bool,
    params=search_params,
    score_threshold=0.123,
    offset=10,
    using="abc",
    with_vectors=grpc.WithVectorsSelector(include=grpc.VectorsSelector(names=["abc", "def"])),
    shard_key_selector=shard_key_selector,
)

recommend_strategy = grpc.RecommendStrategy.BestScore
recommend_strategy2 = grpc.RecommendStrategy.AverageVector

recommend_points = grpc.RecommendPoints(
    collection_name="collection-123",
    positive=[point_id_1, point_id_2],
    negative=[point_id],
    filter=filter_,
    limit=100,
    with_payload=with_payload_bool,
    params=search_params,
    score_threshold=0.123,
    offset=10,
    using="abc",
    with_vectors=grpc.WithVectorsSelector(enable=True),
    strategy=recommend_strategy,
    positive_vectors=[
        grpc.Vector(data=[1.0, 2.0, -1.0, -0.2]),
        grpc.Vector(data=[2.0, 2.0, -1.0, -0.2]),
    ],
    negative_vectors=[
        grpc.Vector(data=[3.0, 2.0, -1.0, -0.2]),
    ],
    shard_key_selector=shard_key_selector_2,
    lookup_from=lookup_location_1,
)
legacy_sparse_vector = grpc.Vector(
    data=[0.2, 0.3, 0.4],
    indices=SparseIndices(data=[1, 2, 3]),
)
recommend_points_sparse = grpc.RecommendPoints(
    collection_name="collection-123",
    positive=[point_id_1, point_id_2],
    negative=[point_id],
    filter=filter_,
    limit=100,
    with_payload=with_payload_bool,
    params=search_params,
    score_threshold=0.123,
    offset=10,
    using="abc",
    with_vectors=grpc.WithVectorsSelector(enable=True),
    strategy=recommend_strategy,
    positive_vectors=[legacy_sparse_vector],
    negative_vectors=[legacy_sparse_vector],
    shard_key_selector=shard_key_selector_2,
)

read_consistency = grpc.ReadConsistency(
    factor=1,
)

read_consistency_0 = grpc.ReadConsistency(
    type=grpc.ReadConsistencyType.Majority,
)

read_consistency_1 = grpc.ReadConsistency(
    type=grpc.ReadConsistencyType.All,
)

read_consistency_2 = grpc.ReadConsistency(
    type=grpc.ReadConsistencyType.Quorum,
)

ordering_0 = grpc.WriteOrdering(
    type=grpc.WriteOrderingType.Weak,
)

ordering_1 = grpc.WriteOrdering(
    type=grpc.WriteOrderingType.Medium,
)

ordering_2 = grpc.WriteOrdering(
    type=grpc.WriteOrderingType.Strong,
)

point_vector_1 = grpc.PointVectors(
    id=point_id_1,
    vectors=single_vector,
)

point_vector_2 = grpc.PointVectors(
    id=point_id_2,
    vectors=many_vectors,
)

group_id_1 = grpc.GroupId(unsigned_value=123)
group_id_2 = grpc.GroupId(integer_value=-456)
group_id_3 = grpc.GroupId(string_value="abc")

groups = [
    grpc.PointGroup(id=group_id_1, hits=[scored_point]),
    grpc.PointGroup(id=group_id_2, hits=[scored_point]),
    grpc.PointGroup(
        id=group_id_3,
        hits=[
            scored_point,
            scored_point,
            scored_point,
        ],
    ),
]

group_result = grpc.GroupsResult(groups=groups)

with_lookup = grpc.WithLookup(
    collection="lalala",
    with_vectors=grpc.WithVectorsSelector(enable=True),
    with_payload=with_payload_include,
)

vector_example_1 = grpc.VectorExample(
    vector=grpc.Vector(data=[1.0, 2.0, 3.0, 5.0]),
)

vector_example_2 = grpc.VectorExample(
    id=point_id_1,
)

vector_example_3 = grpc.VectorExample(
    vector=grpc.Vector(
        data=[1.0, 2.0, 3.0, 5.0],
        indices=SparseIndices(data=[1, 2, 3, 4]),
    ),
    id=point_id_1,
)

target_vector_1 = grpc.TargetVector(
    single=vector_example_1,
)

context_example_pair_1 = grpc.ContextExamplePair(
    positive=vector_example_1,
    negative=vector_example_2,
)

discover_points = grpc.DiscoverPoints(
    collection_name="collection-123",
    target=target_vector_1,
    context=[context_example_pair_1, context_example_pair_1],
    filter=filter_,
    limit=100,
    with_payload=with_payload_bool,
    params=search_params,
    offset=10,
    using="abc",
    with_vectors=grpc.WithVectorsSelector(enable=True),
    shard_key_selector=shard_key_selector_2,
)

sparse_vector_example = grpc.VectorExample(
    vector=legacy_sparse_vector,
)
target_vector_sparse = grpc.TargetVector(
    single=sparse_vector_example,
)

context_example_pair_sparse = grpc.ContextExamplePair(
    positive=sparse_vector_example,
    negative=sparse_vector_example,
)
discover_points_sparse = grpc.DiscoverPoints(
    collection_name="collection-123",
    target=target_vector_sparse,
    context=[context_example_pair_sparse, context_example_pair_sparse],
    filter=filter_,
    limit=100,
    with_payload=with_payload_bool,
    params=search_params,
    offset=10,
    using="abc",
    with_vectors=grpc.WithVectorsSelector(enable=True),
    shard_key_selector=shard_key_selector_2,
    lookup_from=lookup_location_1,
)

upsert_operation = grpc.PointsUpdateOperation(
    upsert=grpc.PointsUpdateOperation.PointStructList(
        points=[point_struct],
    ),
)

delete_operation_1 = grpc.PointsUpdateOperation(
    delete_points=grpc.PointsUpdateOperation.DeletePoints(points=points_selector_list),
)

delete_operation_2 = grpc.PointsUpdateOperation(
    delete_points=grpc.PointsUpdateOperation.DeletePoints(points=points_selector_filter),
)

set_payload_operation_1 = grpc.PointsUpdateOperation(
    set_payload=grpc.PointsUpdateOperation.SetPayload(
        payload=payload_to_grpc({"my_payload": payload_value}),
        points_selector=points_selector_list,
    ),
)

set_payload_operation_2 = grpc.PointsUpdateOperation(
    set_payload=grpc.PointsUpdateOperation.SetPayload(
        payload=payload_to_grpc({"my_payload": payload_value}),
        points_selector=points_selector_filter,
    ),
)

overwrite_payload_operation_1 = grpc.PointsUpdateOperation(
    overwrite_payload=grpc.PointsUpdateOperation.OverwritePayload(
        payload=payload_to_grpc({"my_payload": payload_value}),
        points_selector=points_selector_list,
    ),
)

overwrite_payload_operation_2 = grpc.PointsUpdateOperation(
    overwrite_payload=grpc.PointsUpdateOperation.OverwritePayload(
        payload=payload_to_grpc({"my_payload": payload_value}),
        points_selector=points_selector_filter,
    ),
)

delete_payload_operation_1 = grpc.PointsUpdateOperation(
    delete_payload=grpc.PointsUpdateOperation.DeletePayload(
        keys=["my_payload", "my_payload2"],
        points_selector=points_selector_list,
    ),
)

delete_payload_operation_2 = grpc.PointsUpdateOperation(
    delete_payload=grpc.PointsUpdateOperation.DeletePayload(
        keys=["my_payload", "my_payload2"],
        points_selector=points_selector_filter,
    ),
)

clear_payload_operation_1 = grpc.PointsUpdateOperation(
    clear_payload=grpc.PointsUpdateOperation.ClearPayload(points=points_selector_list),
)

clear_payload_operation_2 = grpc.PointsUpdateOperation(
    clear_payload=grpc.PointsUpdateOperation.ClearPayload(points=points_selector_filter),
)

update_vectors_operation = grpc.PointsUpdateOperation(
    update_vectors=grpc.PointsUpdateOperation.UpdateVectors(
        points=[point_vector_1, point_vector_2]
    ),
)

delete_vectors_operation = grpc.PointsUpdateOperation(
    delete_vectors=grpc.PointsUpdateOperation.DeleteVectors(
        points_selector=points_selector_list,
        vectors=grpc.VectorsSelector(names=["image", "text"]),
    ),
)

delete_vectors_operation_2 = grpc.PointsUpdateOperation(
    delete_vectors=grpc.PointsUpdateOperation.DeleteVectors(
        points_selector=points_selector_filter,
        vectors=grpc.VectorsSelector(names=["image", "text"]),
    ),
)

sharding_method_1 = grpc.Auto
sharding_method_2 = grpc.Custom

float_start_from = grpc.StartFrom(
    float=1.0,
)

integer_start_from = grpc.StartFrom(
    integer=1,
)

timestamp_start_from = grpc.StartFrom(
    timestamp=timestamp,
)

datetime_start_from = grpc.StartFrom(
    datetime=datetime.datetime.now().isoformat(),
)

direction_asc = grpc.Asc
direction_desc = grpc.Desc

order_by = grpc.OrderBy(
    key="my_field",
    direction=direction_asc,
    start_from=float_start_from,
)

dense_vector = grpc.DenseVector(data=[1.0, 2.0, 3.0, 4.0])
dense_vector_2 = grpc.DenseVector(data=[5.0, 6.0, 7.0, 8.0])
sparse_vector = grpc.SparseVector(values=[1.0, 2.0, 3.0, 4.0], indices=[1, 2, 3, 4])
multi_dense_vector = grpc.MultiDenseVector(vectors=[dense_vector, dense_vector_2])
vector_input_id = grpc.VectorInput(id=point_id_2)
vector_input_dense = grpc.VectorInput(dense=dense_vector)
vector_input_dense_2 = grpc.VectorInput(dense=dense_vector_2)
vector_input_sparse = grpc.VectorInput(sparse=sparse_vector)
vector_input_multi = grpc.VectorInput(multi_dense=multi_dense_vector)
recommend_input = grpc.RecommendInput(
    positive=[vector_input_dense],
    negative=[vector_input_dense_2],
)
recommend_input_strategy = grpc.RecommendInput(
    positive=[vector_input_id],
    strategy=recommend_strategy,
)

context_input_pair = grpc.ContextInputPair(
    positive=vector_input_dense, negative=vector_input_multi
)
context_input = grpc.ContextInput(pairs=[context_input_pair])
discover_input = grpc.DiscoverInput(target=vector_input_dense, context=context_input)

query_nearest = grpc.Query(nearest=vector_input_sparse)
query_recommend = grpc.Query(recommend=recommend_input)
query_recommend_id = grpc.Query(recommend=recommend_input_strategy)
query_discover = grpc.Query(discover=discover_input)
query_context = grpc.Query(context=context_input)
query_order_by = grpc.Query(order_by=order_by)
query_fusion = grpc.Query(fusion=grpc.Fusion.RRF)
query_fusion_dbsf = grpc.Query(fusion=grpc.Fusion.DBSF)

deep_prefetch_query = grpc.PrefetchQuery(query=query_recommend)
prefetch_query = grpc.PrefetchQuery(
    prefetch=[deep_prefetch_query],
    filter=filter_,
    query=query_fusion_dbsf,
)
prefetch_full_query = grpc.PrefetchQuery(
    prefetch=[prefetch_query],
    query=query_fusion,
    filter=filter_,
    params=search_params_2,
    score_threshold=0.123,
    limit=100,
    lookup_from=lookup_location_1,
)
prefetch_many = grpc.PrefetchQuery(
    prefetch=[prefetch_query, prefetch_full_query],
)

fixtures = {
    "CollectionParams": [collection_params, collection_params_2],
    "CollectionConfig": [collection_config],
    "ScoredPoint": [
        scored_point,
        scored_point_order_value_int,
        scored_point_order_value_float,
        scored_point_multivector,
    ],
    "CreateAlias": [create_alias],
    "GeoBoundingBox": [geo_bounding_box],
    "SearchParams": [search_params, search_params_2, search_params_3],
    "HasIdCondition": [has_id_condition],
    "RenameAlias": [rename_alias],
    "ValuesCount": [values_count],
    "Filter": [filter_nested, filter_],
    "OptimizersConfigDiff": [optimizer_config, optimizer_config_half],
    "CollectionInfo": [
        collection_info,
        collection_info_ok,
        collection_info_red,
        collection_info_grey,
    ],
    "FieldCondition": [
        field_condition_match,
        field_condition_range,
        field_condition_datetime_range,
        field_condition_geo_radius,
        field_condition_geo_bounding_box,
        field_condition_values_count,
    ],
    "Range": [range_],
    "DatetimeRange": [datetime_range],
    "GeoRadius": [geo_radius],
    "UpdateResult": [update_result, update_result_completed],
    "IsEmptyCondition": [is_empty],
    "IsNullCondition": [is_null],
    "DeleteAlias": [delete_alias],
    "PointStruct": [point_struct, point_struct_many],
    "CollectionDescription": [collection_description],
    "GeoPoint": [geo_point],
    "WalConfigDiff": [wal_config],
    "HnswConfigDiff": [hnsw_config, hnsw_config_2],
    "UpdateCollection": [update_collection, update_collection_2, update_collection_3],
    "Condition": [
        condition_field_match,
        condition_range,
        condition_geo_radius,
        condition_geo_bounding_box,
        condition_values_count,
    ],
    "PointsSelector": [points_selector_list, points_selector_filter],
    "AliasDescription": [alias_description],
    "AliasOperations": [
        alias_operations_create,
        alias_operations_rename,
        alias_operations_delete,
    ],
    "Match": [match_keyword, match_integer, match_bool, match_text],
    "WithPayloadSelector": [
        with_payload_bool,
        with_payload_include,
        with_payload_exclude,
    ],
    "RetrievedPoint": [retrieved_point, retrieved_point_with_order_value],
    "CountResult": [count_result],
    "SnapshotDescription": [snapshot_description],
    "VectorParams": [
        vector_param,
        vector_param_with_hnsw,
        vector_param_with_quant,
        vector_param_1,
        vector_param_2,
        vector_param_with_multivector,
    ],
    "VectorsConfig": [single_vector_config, vector_config],
    "SearchPoints": [search_points, search_points_all_vectors],
    "QueryPoints": [query_points],
    "RecommendPoints": [recommend_points, recommend_points_sparse],
    "RecommendStrategy": [recommend_strategy, recommend_strategy2],
    "TextIndexParams": [
        text_index_params_1,
        text_index_params_2,
        text_index_params_3,
    ],
    "IntegerIndexParams": [
        integer_index_params_0,
        integer_index_params_1,
        integer_index_params_2,
    ],
    "CollectionParamsDiff": [collections_params_diff],
    "LookupLocation": [lookup_location_1, lookup_location_2],
    "ReadConsistency": [
        read_consistency,
        read_consistency_0,
        read_consistency_1,
        read_consistency_2,
    ],
    "WriteOrdering": [ordering_0, ordering_1, ordering_2],
    "QuantizationConfig": [quantization_config, binary_quantization_config]
    + product_quantizations,
    "QuantizationSearchParams": [quantization_search_params],
    "PointVectors": [point_vector_1, point_vector_2],
    "GroupId": [group_id_1, group_id_2, group_id_3],
    "GroupsResult": [group_result],
    "WithLookup": [with_lookup],
    "PointsUpdateOperation": [
        upsert_operation,
        delete_operation_1,
        delete_operation_2,
        set_payload_operation_1,
        set_payload_operation_2,
        overwrite_payload_operation_1,
        overwrite_payload_operation_2,
        delete_payload_operation_1,
        delete_payload_operation_2,
        clear_payload_operation_1,
        clear_payload_operation_2,
        update_vectors_operation,
        delete_vectors_operation,
        delete_vectors_operation_2,
    ],
    "DiscoverPoints": [discover_points, discover_points_sparse],
    "ContextExamplePair": [context_example_pair_1],
    "VectorExample": [vector_example_1, vector_example_2, vector_example_3],
    "TargetVector": [target_vector_1],
    "SparseVectorParams": [sparse_vector_params, sparse_vector_params_datatype],
    "SparseVectorConfig": [sparse_vector_config],
    "ShardKeySelector": [shard_key_selector, shard_key_selector_2],
    "ShardingMethod": [sharding_method_1, sharding_method_2],
    "StartFrom": [float_start_from, integer_start_from, datetime_start_from, timestamp_start_from],
    "Direction": [direction_asc, direction_desc],
    "OrderBy": [order_by],
    "OrderValue": [order_value_int, order_value_float],
    "Query": [
        query_nearest,
        query_recommend,
        query_discover,
        query_context,
        query_order_by,
        query_fusion,
        query_recommend_id,
    ],
    "PrefetchQuery": [deep_prefetch_query, prefetch_query, prefetch_full_query, prefetch_many],
}


def get_grpc_fixture(model_name: str) -> List[Message]:
    if model_name not in fixtures:
        raise RuntimeError(f"Model {model_name} not found in fixtures")
    return fixtures[model_name]
