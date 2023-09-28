import inspect
import logging
import re
from inspect import getmembers

from google.protobuf.json_format import MessageToDict

from tests.conversions.fixtures import fixtures as class_fixtures
from tests.conversions.fixtures import get_grpc_fixture


def camel_to_snake(name):
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

            if MessageToDict(grpc_fixture) != MessageToDict(fixture):
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
    ):
        assert payload_schema == grpc_field_type_to_payload_schema(
            grpc_payload_schema_to_field_type(payload_schema)
        )        


def test_init_from_conversion():
    from qdrant_client.conversions.conversion import GrpcToRest, RestToGrpc

    init_from = "collection_name"
    recovered = RestToGrpc.convert_init_from(GrpcToRest.convert_init_from(init_from))
    assert init_from == recovered


def test_recommend_examples_to_grpc_conversion():
    from qdrant_client.conversions.conversion import RestToGrpc
    from qdrant_client.grpc import PointId, Vector
    
    vector = [0.0, 2.0, 3.0, 4.0, 5.0]
    fixture = [10, "uuid_1", "uuid_2", vector, 20]
    
    ids = RestToGrpc.convert_recommend_examples_to_ids(fixture)
    
    assert ids == [PointId(num=10), PointId(uuid="uuid_1"), PointId(uuid="uuid_2"), PointId(num=20)]
    
    vectors = RestToGrpc.convert_recommend_examples_to_vectors(fixture)
    
    assert vectors == [Vector(data=vector)]
