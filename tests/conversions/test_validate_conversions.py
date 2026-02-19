import inspect
import logging
import re
from datetime import date, datetime, timedelta, timezone
from inspect import getmembers

import pytest
from google.protobuf.json_format import MessageToDict

from tests.conversions.fixtures import fixtures as class_fixtures
from tests.conversions.fixtures import get_grpc_fixture


def camel_to_snake(name: str):
    name = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
    return re.sub("([a-z0-9])([A-Z])", r"\1_\2", name).lower()


def test_conversion_completeness():
    from qdrant_client.conversions.conversion import GrpcToRest, RestToGrpc

    grpc_to_rest_convert = dict(
        (method_name, method)
        for method_name, method in getmembers(GrpcToRest)
        if method_name.startswith("convert_")
    )

    rest_to_grpc_convert = dict(
        (method_name, method)
        for method_name, method in getmembers(RestToGrpc)
        if method_name.startswith("convert_")
    )

    for model_class_name in class_fixtures:
        convert_function_name = f"convert_{camel_to_snake(model_class_name)}"

        fixtures = get_grpc_fixture(model_class_name)
        for fixture in fixtures:
            if fixture is ...:
                logging.warning(f"Fixture for {model_class_name} skipped")
                continue

            try:
                result = list(
                    inspect.signature(
                        grpc_to_rest_convert[convert_function_name]
                    ).parameters.keys()
                )
                if "collection_name" in result:
                    rest_fixture = grpc_to_rest_convert[convert_function_name](
                        fixture, collection_name=fixture.collection_name
                    )
                else:
                    rest_fixture = grpc_to_rest_convert[convert_function_name](fixture)

                back_convert_function_name = convert_function_name

                print(
                    f"back_convert_function_name: {back_convert_function_name} for {type(rest_fixture)}"
                )

                result = list(
                    inspect.signature(
                        rest_to_grpc_convert[back_convert_function_name]
                    ).parameters.keys()
                )
                if "collection_name" in result:
                    grpc_fixture = rest_to_grpc_convert[back_convert_function_name](
                        rest_fixture, collection_name=fixture.collection_name
                    )
                else:
                    grpc_fixture = rest_to_grpc_convert[back_convert_function_name](rest_fixture)
            except Exception as e:
                logging.warning(f"Error with {fixture}")
                raise e
            if isinstance(grpc_fixture, int):
                # Is an enum
                assert grpc_fixture == fixture, f"{model_class_name} conversion is broken"
            elif MessageToDict(grpc_fixture) != MessageToDict(fixture):
                assert MessageToDict(grpc_fixture) == MessageToDict(
                    fixture
                ), f"{model_class_name} conversion is broken"


def test_nested_filter():
    from qdrant_client.conversions.conversion import GrpcToRest
    from qdrant_client.http.models import models as rest

    from .fixtures import condition_nested

    rest_condition = GrpcToRest.convert_condition(condition_nested)

    rest_filter = rest.Filter(must=[rest_condition])
    assert isinstance(rest_filter.must[0], type(rest_condition))


def test_vector_batch_conversion():
    from qdrant_client import grpc
    from qdrant_client.conversions.conversion import RestToGrpc

    batch = []
    res = RestToGrpc.convert_batch_vector_struct(batch, 1)
    assert len(res) == 0

    batch = {}
    res = RestToGrpc.convert_batch_vector_struct(batch, 1)
    assert len(res) == 1
    assert res == [grpc.Vectors(vectors=grpc.NamedVectors(vectors={}))]

    batch = []
    res = RestToGrpc.convert_batch_vector_struct(batch, 1)
    assert len(res) == 0

    batch = [[]]
    res = RestToGrpc.convert_batch_vector_struct(batch, 1)
    assert len(res) == 1
    assert res == [grpc.Vectors(vector=grpc.Vector(dense=grpc.DenseVector(data=[])))]

    batch = [[1, 2, 3]]
    res = RestToGrpc.convert_batch_vector_struct(batch, 1)
    assert len(res) == 1
    assert res == [grpc.Vectors(vector=grpc.Vector(dense=grpc.DenseVector(data=[1, 2, 3])))]

    batch = [[1, 2, 3]]
    res = RestToGrpc.convert_batch_vector_struct(batch, 1)
    assert len(res) == 1
    assert res == [grpc.Vectors(vector=grpc.Vector(dense=grpc.DenseVector(data=[1, 2, 3])))]

    batch = [[1, 2, 3], [3, 4, 5]]
    res = RestToGrpc.convert_batch_vector_struct(batch, 0)
    assert len(res) == 2
    assert res == [
        grpc.Vectors(vector=grpc.Vector(dense=grpc.DenseVector(data=[1, 2, 3]))),
        grpc.Vectors(vector=grpc.Vector(dense=grpc.DenseVector(data=[3, 4, 5]))),
    ]

    batch = {"image": [[1, 2, 3]]}
    res = RestToGrpc.convert_batch_vector_struct(batch, 1)
    assert len(res) == 1
    assert res == [
        grpc.Vectors(
            vectors=grpc.NamedVectors(
                vectors={"image": grpc.Vector(dense=grpc.DenseVector(data=[1, 2, 3]))}
            )
        )
    ]

    batch = {"image": [[1, 2, 3], [3, 4, 5]]}
    res = RestToGrpc.convert_batch_vector_struct(batch, 2)
    assert len(res) == 2
    assert res == [
        grpc.Vectors(
            vectors=grpc.NamedVectors(
                vectors={"image": grpc.Vector(dense=grpc.DenseVector(data=[1, 2, 3]))}
            )
        ),
        grpc.Vectors(
            vectors=grpc.NamedVectors(
                vectors={"image": grpc.Vector(dense=grpc.DenseVector(data=[3, 4, 5]))}
            )
        ),
    ]

    batch = {"image": [[1, 2, 3], [3, 4, 5]], "restaurants": [[6, 7, 8], [9, 10, 11]]}
    res = RestToGrpc.convert_batch_vector_struct(batch, 2)
    assert len(res) == 2
    assert res == [
        grpc.Vectors(
            vectors=grpc.NamedVectors(
                vectors={
                    "image": grpc.Vector(dense=grpc.DenseVector(data=[1, 2, 3])),
                    "restaurants": grpc.Vector(dense=grpc.DenseVector(data=[6, 7, 8])),
                }
            )
        ),
        grpc.Vectors(
            vectors=grpc.NamedVectors(
                vectors={
                    "image": grpc.Vector(dense=grpc.DenseVector(data=[3, 4, 5])),
                    "restaurants": grpc.Vector(dense=grpc.DenseVector(data=[9, 10, 11])),
                }
            )
        ),
    ]


def test_sparse_vector_batch_conversion():
    from qdrant_client import grpc
    from qdrant_client.conversions.conversion import RestToGrpc
    from qdrant_client.http.models import SparseVector

    batch = {"image": [SparseVector(values=[1.5, 2.4, 8.1], indices=[10, 20, 30])]}
    res = RestToGrpc.convert_batch_vector_struct(batch, 1)
    assert len(res) == 1
    assert res == [
        grpc.Vectors(
            vectors=grpc.NamedVectors(
                vectors={
                    "image": grpc.Vector(
                        sparse=grpc.SparseVector(values=[1.5, 2.4, 8.1], indices=[10, 20, 30])
                    )
                }
            )
        ),
    ]

    batch = {
        "image": [
            SparseVector(values=[1.5, 2.4, 8.1], indices=[10, 20, 30]),
            SparseVector(values=[7.8, 3.2, 9.5], indices=[100, 200, 300]),
        ]
    }
    res = RestToGrpc.convert_batch_vector_struct(batch, 2)
    assert len(res) == 2
    assert res == [
        grpc.Vectors(
            vectors=grpc.NamedVectors(
                vectors={
                    "image": grpc.Vector(
                        sparse=grpc.SparseVector(values=[1.5, 2.4, 8.1], indices=[10, 20, 30])
                    )
                }
            )
        ),
        grpc.Vectors(
            vectors=grpc.NamedVectors(
                vectors={
                    "image": grpc.Vector(
                        sparse=grpc.SparseVector(values=[7.8, 3.2, 9.5], indices=[100, 200, 300])
                    )
                }
            )
        ),
    ]


def test_grpc_payload_scheme_conversion():
    from qdrant_client.conversions.conversion import (
        grpc_field_type_to_payload_schema,
        grpc_payload_schema_to_field_type,
    )
    from qdrant_client.grpc import PayloadSchemaType

    for payload_schema in (
        PayloadSchemaType.Keyword,
        PayloadSchemaType.Integer,
        PayloadSchemaType.Float,
        PayloadSchemaType.Geo,
        PayloadSchemaType.Text,
        PayloadSchemaType.Bool,
        PayloadSchemaType.Datetime,
        PayloadSchemaType.Uuid,
    ):
        assert payload_schema == grpc_field_type_to_payload_schema(
            grpc_payload_schema_to_field_type(payload_schema)
        )


@pytest.mark.parametrize(
    "dt",
    [
        datetime(2021, 1, 1, 0, 0, 0, tzinfo=timezone.utc),
        datetime(2021, 1, 1, 0, 0, 0, tzinfo=timezone(timedelta(hours=5))),
        datetime(2021, 1, 1, 0, 0, 0),
        datetime.utcnow(),
        datetime.now(),
        date.today(),
    ],
)
def test_datetime_to_timestamp_conversions(dt: datetime | date):
    from qdrant_client.conversions.conversion import GrpcToRest, RestToGrpc

    rest_to_grpc = RestToGrpc.convert_datetime(dt)
    grpc_to_rest = GrpcToRest.convert_timestamp(rest_to_grpc)

    if isinstance(dt, date) and not isinstance(dt, datetime):
        dt = datetime.combine(dt, datetime.min.time())

    assert (
        dt.utctimetuple() == grpc_to_rest.utctimetuple()
    ), f"Failed for {dt}, should be equal to {grpc_to_rest}"


def test_convert_context_input_flat_pair():
    from qdrant_client import models
    from qdrant_client.conversions.conversion import GrpcToRest, RestToGrpc

    rest_context_pair = models.ContextPair(
        positive=1,
        negative=2,
    )
    grpc_context_input = RestToGrpc.convert_context_input(rest_context_pair)
    recovered = GrpcToRest.convert_context_input(grpc_context_input)

    assert recovered[0] == rest_context_pair


def test_convert_query_interface():
    from qdrant_client import models
    from qdrant_client.conversions.conversion import GrpcToRest, RestToGrpc

    rest_query = 1
    expected = models.NearestQuery(nearest=rest_query)
    grpc_query = RestToGrpc.convert_query_interface(rest_query)
    recovered = GrpcToRest.convert_query(grpc_query)

    assert recovered == expected

    grpc_query = RestToGrpc.convert_query_interface(expected)
    recovered = GrpcToRest.convert_query(grpc_query)

    assert recovered == expected


def test_convert_flat_prefetch():
    from qdrant_client import models
    from qdrant_client.conversions.conversion import GrpcToRest, RestToGrpc

    rest_prefetch = models.Prefetch(prefetch=models.Prefetch(using="test"))
    grpc_prefetch = RestToGrpc.convert_prefetch_query(rest_prefetch)
    recovered = GrpcToRest.convert_prefetch_query(grpc_prefetch)

    assert recovered.prefetch[0] == rest_prefetch.prefetch


def test_convert_flat_filter():
    from qdrant_client import models
    from qdrant_client.conversions.conversion import GrpcToRest, RestToGrpc

    rest_filter = models.Filter(
        must=models.FieldCondition(key="mandatory", match=models.MatchValue(value=1)),
        should=models.FieldCondition(key="desirable", range=models.DatetimeRange(lt=3.0)),
        must_not=models.HasIdCondition(has_id=[1, 2, 3]),
        min_should=models.MinShould(
            conditions=[
                models.FieldCondition(key="at_least_one", values_count=models.ValuesCount(gte=1)),
                models.FieldCondition(key="fallback", match=models.MatchValue(value=42)),
            ],
            min_count=1,
        ),
    )
    grpc_filter = RestToGrpc.convert_filter(rest_filter)
    recovered = GrpcToRest.convert_filter(grpc_filter)

    assert recovered.must[0] == rest_filter.must
    assert recovered.should[0] == rest_filter.should
    assert recovered.must_not[0] == rest_filter.must_not


def test_query_points():
    from qdrant_client import models
    from qdrant_client.conversions.conversion import GrpcToRest, RestToGrpc

    prefetch = models.Prefetch(query=models.NearestQuery(nearest=[1.0, 2.0]))
    query_request = models.QueryRequest(
        query=1,
        limit=5,
        using="test",
        with_payload=True,
        prefetch=prefetch,
    )
    grpc_query_request = RestToGrpc.convert_query_request(query_request, "check")
    recovered = GrpcToRest.convert_query_points(grpc_query_request)

    assert recovered.query == models.NearestQuery(nearest=query_request.query)
    assert recovered.limit == query_request.limit
    assert recovered.using == query_request.using
    assert recovered.with_payload == query_request.with_payload
    assert recovered.prefetch[0] == query_request.prefetch


def test_convert_text_index_params_stopwords():
    from qdrant_client import models
    from qdrant_client.conversions.conversion import GrpcToRest, RestToGrpc

    text_index_params = models.TextIndexParams(
        type=models.TextIndexType.TEXT,
        stopwords=models.Language.ENGLISH,
    )

    grpc_text_index_params = RestToGrpc.convert_text_index_params(text_index_params)
    recovered = GrpcToRest.convert_text_index_params(grpc_text_index_params)

    assert recovered == text_index_params

    text_index_params_1 = models.TextIndexParams(
        type=models.TextIndexType.TEXT,
        stopwords=models.StopwordsSet(custom=["custom1", "custom2", "custom3"]),
    )

    grpc_text_index_params_1 = RestToGrpc.convert_text_index_params(text_index_params_1)
    recovered_1 = GrpcToRest.convert_text_index_params(grpc_text_index_params_1)

    assert recovered_1.stopwords.custom == text_index_params_1.stopwords.custom
    assert recovered_1.stopwords.languages == []

    text_index_params_2 = models.TextIndexParams(
        type=models.TextIndexType.TEXT,
        stopwords=models.StopwordsSet(
            custom=["custom1", "custom2", "custom3"],
            languages=[models.Language.ENGLISH, models.Language.GERMAN],
        ),
    )
    grpc_text_index_params_2 = RestToGrpc.convert_text_index_params(text_index_params_2)

    recovered_2 = GrpcToRest.convert_text_index_params(grpc_text_index_params_2)
    assert recovered_2 == text_index_params_2

    text_index_params_3 = models.TextIndexParams(
        type=models.TextIndexType.TEXT,
        stopwords=models.StopwordsSet(
            languages=[
                "english",
                "german",
            ],  # though it's not directly supported by the interface, strings might
            # be convenient to use
        ),
    )
    grpc_text_index_params_3 = RestToGrpc.convert_text_index_params(text_index_params_3)
    recovered_3 = GrpcToRest.convert_text_index_params(grpc_text_index_params_3)

    assert recovered_3.stopwords.languages == text_index_params_3.stopwords.languages
    assert recovered_3.stopwords.custom == []

    text_index_params_4 = models.TextIndexParams(
        type=models.TextIndexType.TEXT,
        stopwords=models.StopwordsSet(
            languages=[models.Language.ENGLISH],
        ),
    )
    grpc_text_index_params_4 = RestToGrpc.convert_text_index_params(text_index_params_4)

    recovered_4 = GrpcToRest.convert_text_index_params(grpc_text_index_params_4)

    assert recovered_4.stopwords == models.Language.ENGLISH


def test_inference_without_options():
    from qdrant_client import models
    from qdrant_client.conversions.conversion import GrpcToRest, RestToGrpc

    doc_wo_options = models.Document(text="qwerty-text", model="qwerty-text-model")
    image_wo_options = models.Image(image="qwerty-image", model="qwerty-image-model")
    inference_wo_options = models.InferenceObject(object="qwerty-any", model="qwerty-any-model")

    grpc_doc_wo_options = RestToGrpc.convert_document(doc_wo_options)
    grpc_image_wo_options = RestToGrpc.convert_image(image_wo_options)
    grpc_inference_wo_options = RestToGrpc.convert_inference_object(inference_wo_options)

    recovered_doc_wo_options = GrpcToRest.convert_document(grpc_doc_wo_options)
    recovered_image_wo_options = GrpcToRest.convert_image(grpc_image_wo_options)
    recovered_inference_wo_options = GrpcToRest.convert_inference_object(grpc_inference_wo_options)

    assert recovered_doc_wo_options.options == {}
    assert recovered_image_wo_options.options == {}
    assert recovered_inference_wo_options.options == {}


def test_convert_shard_key_with_fallback():
    from qdrant_client import models, grpc as q_grpc
    from qdrant_client.conversions.conversion import GrpcToRest, RestToGrpc

    single_int_shard_key = 2
    single_str_shard_key = "abc"
    shard_keys = [2, "qwerty"]
    shard_key_with_int_fallback = models.ShardKeyWithFallback(target="123", fallback=3)
    shard_key_with_str_fallback = models.ShardKeyWithFallback(target=123, fallback="zxc")

    for key in (
        single_int_shard_key,
        single_str_shard_key,
        shard_keys,
        shard_key_with_int_fallback,
        shard_key_with_str_fallback,
    ):
        grpc_key = RestToGrpc.convert_shard_key_selector(key)
        restored_key = GrpcToRest.convert_shard_key_selector(grpc_key)
        assert restored_key == key

    single_int_shard_key_list = [3]
    single_str_shard_key_list = ["abc"]
    for keys in single_int_shard_key_list, single_str_shard_key_list:
        grpc_keys = RestToGrpc.convert_shard_key_selector(keys)
        restored_key = GrpcToRest.convert_shard_key_selector(grpc_keys)
        assert keys[0] == restored_key

    invalid_grpc_fallback_shard_key = q_grpc.ShardKeySelector(
        shard_keys=[q_grpc.ShardKey(number=3), q_grpc.ShardKey(number=2)],
        fallback=q_grpc.ShardKey(number=2),
    )

    with pytest.raises(ValueError):
        GrpcToRest.convert_shard_key_selector(invalid_grpc_fallback_shard_key)


def test_legacy_vector():
    from qdrant_client import grpc as q_grpc
    from qdrant_client.conversions.conversion import GrpcToRest, RestToGrpc

    legacy_sparse_vector = q_grpc.Vector(
        data=[0.2, 0.3, 0.4],
        indices=q_grpc.SparseIndices(data=[1, 2, 3]),
    )

    rest_sparse_vector = GrpcToRest.convert_vector(legacy_sparse_vector)
    restored_sparse_vector = RestToGrpc.convert_sparse_vector_to_vector(rest_sparse_vector)

    assert restored_sparse_vector == q_grpc.Vector(
        sparse=q_grpc.SparseVector(
            values=legacy_sparse_vector.data, indices=legacy_sparse_vector.indices.data
        )
    )

    legacy_dense_vector = q_grpc.Vector(data=[1.0, 2.0])
    rest_dense_vector = GrpcToRest.convert_vector(legacy_dense_vector)
    restored_dense_vector = RestToGrpc.convert_vector_struct(rest_dense_vector)

    assert restored_dense_vector.vector == q_grpc.Vector(
        dense=q_grpc.DenseVector(data=legacy_dense_vector.data)
    )

    legacy_multi_dense_vector = q_grpc.Vector(data=[1.0, 2.0, 3.0, 4.0], vectors_count=2)
    rest_multidense_vector = GrpcToRest.convert_vector(legacy_multi_dense_vector)
    restored_multi_dense_vector = RestToGrpc.convert_vector_struct(rest_multidense_vector)

    assert restored_multi_dense_vector.vector == q_grpc.Vector(
        multi_dense=q_grpc.MultiDenseVector(
            vectors=[
                q_grpc.DenseVector(data=legacy_multi_dense_vector.data[:2]),
                q_grpc.DenseVector(data=legacy_multi_dense_vector.data[2:]),
            ]
        )
    )


def test_optimizers_config_diff_max_threads():
    from qdrant_client import grpc as q_grpc, models
    from qdrant_client.conversions.conversion import GrpcToRest, RestToGrpc

    for value in (0, 2):
        grpc_opt_conf = q_grpc.OptimizersConfigDiff(
            max_optimization_threads=q_grpc.MaxOptimizationThreads(value=value)
        )

        rest_opt_conf = GrpcToRest.convert_optimizers_config_diff(grpc_opt_conf)
        restored_grpc_opt_conf = RestToGrpc.convert_optimizers_config_diff(rest_opt_conf)

        assert (
            grpc_opt_conf.max_optimization_threads
            == restored_grpc_opt_conf.max_optimization_threads
        )
        assert restored_grpc_opt_conf.deprecated_max_optimization_threads == value

    grpc_opt_conf = q_grpc.OptimizersConfigDiff(
        deleted_threshold=10.0,
        vacuum_min_vector_number=10,
        default_segment_number=2,
        max_optimization_threads=q_grpc.MaxOptimizationThreads(
            setting=q_grpc.MaxOptimizationThreads.Setting.Auto
        ),
    )
    rest_opt_conf = GrpcToRest.convert_optimizers_config_diff(grpc_opt_conf)
    restored_grpc_opt_conf = RestToGrpc.convert_optimizers_config_diff(rest_opt_conf)

    assert grpc_opt_conf == restored_grpc_opt_conf

    rest_opt_conf = GrpcToRest.convert_optimizer_config(grpc_opt_conf)
    assert rest_opt_conf.max_optimization_threads is None

    rest_opt_conf = models.OptimizersConfig(
        deleted_threshold=10.0,
        vacuum_min_vector_number=200,
        default_segment_number=2,
        flush_interval_sec=3,
        max_optimization_threads=3,
    )
    grpc_opt_conf = RestToGrpc.convert_optimizers_config(rest_opt_conf)
    restored_rest_opt_conf = GrpcToRest.convert_optimizer_config(grpc_opt_conf)

    assert rest_opt_conf == restored_rest_opt_conf

    value = 3
    grpc_deprecated_opt_conf = q_grpc.OptimizersConfigDiff(
        deprecated_max_optimization_threads=value,
    )
    rest_opt_conf = GrpcToRest.convert_optimizers_config_diff(grpc_deprecated_opt_conf)

    restored_grpc_opt_conf = RestToGrpc.convert_optimizers_config_diff(rest_opt_conf)

    assert (
        grpc_deprecated_opt_conf.deprecated_max_optimization_threads
        == restored_grpc_opt_conf.deprecated_max_optimization_threads
    )
    assert restored_grpc_opt_conf.max_optimization_threads == q_grpc.MaxOptimizationThreads(
        value=value
    )
