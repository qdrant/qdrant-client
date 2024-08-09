import inspect
import logging
import re
from datetime import date, datetime, timedelta, timezone
from inspect import getmembers
from typing import Union

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
    assert res == [grpc.Vectors(vector=grpc.Vector(data=[]))]

    batch = [[1, 2, 3]]
    res = RestToGrpc.convert_batch_vector_struct(batch, 1)
    assert len(res) == 1
    assert res == [grpc.Vectors(vector=grpc.Vector(data=[1, 2, 3]))]

    batch = [[1, 2, 3]]
    res = RestToGrpc.convert_batch_vector_struct(batch, 1)
    assert len(res) == 1
    assert res == [grpc.Vectors(vector=grpc.Vector(data=[1, 2, 3]))]

    batch = [[1, 2, 3], [3, 4, 5]]
    res = RestToGrpc.convert_batch_vector_struct(batch, 0)
    assert len(res) == 2
    assert res == [
        grpc.Vectors(vector=grpc.Vector(data=[1, 2, 3])),
        grpc.Vectors(vector=grpc.Vector(data=[3, 4, 5])),
    ]

    batch = {"image": [[1, 2, 3]]}
    res = RestToGrpc.convert_batch_vector_struct(batch, 1)
    assert len(res) == 1
    assert res == [
        grpc.Vectors(vectors=grpc.NamedVectors(vectors={"image": grpc.Vector(data=[1, 2, 3])}))
    ]

    batch = {"image": [[1, 2, 3], [3, 4, 5]]}
    res = RestToGrpc.convert_batch_vector_struct(batch, 2)
    assert len(res) == 2
    assert res == [
        grpc.Vectors(vectors=grpc.NamedVectors(vectors={"image": grpc.Vector(data=[1, 2, 3])})),
        grpc.Vectors(vectors=grpc.NamedVectors(vectors={"image": grpc.Vector(data=[3, 4, 5])})),
    ]

    batch = {"image": [[1, 2, 3], [3, 4, 5]], "restaurants": [[6, 7, 8], [9, 10, 11]]}
    res = RestToGrpc.convert_batch_vector_struct(batch, 2)
    assert len(res) == 2
    assert res == [
        grpc.Vectors(
            vectors=grpc.NamedVectors(
                vectors={
                    "image": grpc.Vector(data=[1, 2, 3]),
                    "restaurants": grpc.Vector(data=[6, 7, 8]),
                }
            )
        ),
        grpc.Vectors(
            vectors=grpc.NamedVectors(
                vectors={
                    "image": grpc.Vector(data=[3, 4, 5]),
                    "restaurants": grpc.Vector(data=[9, 10, 11]),
                }
            )
        ),
    ]


def test_sparse_vector_conversion():
    from qdrant_client import grpc
    from qdrant_client.conversions.conversion import GrpcToRest, RestToGrpc

    sparse_vector = grpc.Vector(data=[0.2, 0.3, 0.4], indices=grpc.SparseIndices(data=[3, 2, 5]))
    recovered = RestToGrpc.convert_sparse_vector_to_vector(
        GrpcToRest.convert_vector(sparse_vector)
    )

    assert sparse_vector == recovered


def test_sparse_vector_batch_conversion():
    from qdrant_client import grpc
    from qdrant_client.conversions.conversion import RestToGrpc
    from qdrant_client.grpc import SparseIndices
    from qdrant_client.http.models import SparseVector

    batch = {"image": [SparseVector(values=[1.5, 2.4, 8.1], indices=[10, 20, 30])]}
    res = RestToGrpc.convert_batch_vector_struct(batch, 1)
    assert len(res) == 1
    assert res == [
        grpc.Vectors(
            vectors=grpc.NamedVectors(
                vectors={
                    "image": grpc.Vector(
                        data=[1.5, 2.4, 8.1], indices=SparseIndices(data=[10, 20, 30])
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
                        data=[1.5, 2.4, 8.1], indices=SparseIndices(data=[10, 20, 30])
                    )
                }
            )
        ),
        grpc.Vectors(
            vectors=grpc.NamedVectors(
                vectors={
                    "image": grpc.Vector(
                        data=[7.8, 3.2, 9.5], indices=SparseIndices(data=[100, 200, 300])
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


def test_init_from_conversion():
    from qdrant_client.conversions.conversion import GrpcToRest, RestToGrpc

    init_from = "collection_name"
    recovered = RestToGrpc.convert_init_from(GrpcToRest.convert_init_from(init_from))
    assert init_from == recovered


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
def test_datetime_to_timestamp_conversions(dt: Union[datetime, date]):
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
