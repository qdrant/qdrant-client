import re
from ctypes import Union

import betterproto
from loguru import logger
from pydantic import BaseModel
from inspect import getmembers

from tests.conversions.fixtures import get_grpc_fixture


def camel_to_snake(name):
    name = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', name).lower()


def test_conversion_completeness():
    from qdrant_client.http.models import models

    http_classes = dict([
        (name, cls)
        for name, cls in models.__dict__.items()
        if isinstance(cls, type) and issubclass(cls, (BaseModel, Union))
    ])

    from qdrant_client import grpc
    grpc_classes = dict([
        (name, cls)
        for name, cls in grpc.__dict__.items()
        if isinstance(cls, type) and issubclass(cls, betterproto.Message)
    ])

    common_classes = set(http_classes).intersection(set(grpc_classes))

    from qdrant_client.conversions.conversion import GrpcToRest, RestToGrpc

    grpc_to_rest_convert = dict(
        (method_name, method) for method_name, method
        in getmembers(GrpcToRest) if method_name.startswith("convert_")
    )

    rest_to_grpc_convert = dict(
        (method_name, method) for method_name, method
        in getmembers(RestToGrpc) if method_name.startswith("convert_")
    )

    print("")
    print("---- grpc_to_rest_convert ----")
    for method_name, method in grpc_to_rest_convert.items():
        print(method_name)

    print("---- rest_to_grpc_convert ----")
    for method_name, method in rest_to_grpc_convert.items():
        print(method_name)

    has_missing = False
    for common_class in common_classes:
        convert_function_name = f"convert_{camel_to_snake(common_class)}"
        if convert_function_name not in grpc_to_rest_convert:
            has_missing = True
            logger.warning(f"Missing method {convert_function_name} for {common_class} in GrpcToRest")
            continue

        if convert_function_name not in rest_to_grpc_convert:
            has_missing = True
            logger.warning(f"Missing method {convert_function_name} for {common_class} in RestToGrpc")
            continue

        fixtures = get_grpc_fixture(common_class)
        for fixture in fixtures:
            rest_fixture = grpc_to_rest_convert[convert_function_name](fixture)
            grpc_fixture = rest_to_grpc_convert[convert_function_name](rest_fixture)

            assert grpc_fixture == fixture, f"{common_class} conversion is broken"

    assert not has_missing
