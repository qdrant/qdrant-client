from inspect import getmembers, isfunction
from typing import Any, Callable, get_args

import pytest
from google.protobuf.json_format import MessageToDict

from qdrant_client._pydantic_compat import model_fields
from qdrant_client.serverless import conversions, models
from qdrant_client.serverless.conversions import (
    _DISTANCE_TO_GRPC,
    _PRECISION_TO_GRPC,
    _TOKENIZER_TO_GRPC,
)
from qdrant_client.serverless.grpc import serverless_collections_pb2 as pb2
from tests.conversions.serverless_fixtures import (
    fixtures as class_fixtures,
)


def _converters(suffix: str) -> dict[str, Callable[[Any], Any]]:
    """Map converters by the model they convert, keyed independently of privacy.

    Sub-message converters are module-private (`_stemmer_to_grpc`), so the
    leading underscore is stripped: a fixture key stays valid if a converter
    later becomes public, or the reverse.
    """
    matched = [
        (name, func) for name, func in getmembers(conversions, isfunction) if name.endswith(suffix)
    ]
    converters = {name[: -len(suffix)].lstrip("_"): func for name, func in matched}
    assert len(converters) == len(matched), f"stem collision among {suffix} converters"
    return converters


def test_conversion_completeness() -> None:
    """Round-trip every fixture grpc -> model -> grpc, as tests/conversions does.

    Starting from the grpc side is what exercises server-authored messages; a
    model -> grpc -> model round-trip cannot detect a mis-mapped enum, because
    the decode maps are derived by inverting the encode maps.
    """
    to_grpc, from_grpc = _converters("_to_grpc"), _converters("_from_grpc")

    assert set(to_grpc) == set(from_grpc), "every converter needs both directions"
    assert set(to_grpc) == set(class_fixtures), "every converter needs a fixture"

    for model_name, fixtures in class_fixtures.items():
        for fixture in fixtures:
            model_fixture = from_grpc[model_name](fixture)

            back_convert_function_name = f"{model_name}_to_grpc"

            print(
                f"back_convert_function_name: {back_convert_function_name} for {type(model_fixture)}"
            )

            grpc_fixture = to_grpc[model_name](model_fixture)
            assert MessageToDict(grpc_fixture) == MessageToDict(
                fixture
            ), f"{model_name} conversion is broken for {fixture}"


def test_every_payload_index_kind_has_a_fixture() -> None:
    """Fails when the proto gains an index kind that nothing covers."""
    oneof = pb2.PayloadIndexConfig.DESCRIPTOR.oneofs_by_name["index"]
    covered = {fixture.WhichOneof("index") for fixture in class_fixtures["payload_index"]}
    assert covered == {field.name for field in oneof.fields}


@pytest.mark.parametrize(
    "mapping,descriptor",
    [
        (_DISTANCE_TO_GRPC, pb2.Distance.DESCRIPTOR),
        (_PRECISION_TO_GRPC, pb2.PrecisionTier.DESCRIPTOR),
        (_TOKENIZER_TO_GRPC, pb2.Tokenizer.DESCRIPTOR),
    ],
)
def test_enum_maps_match_proto(mapping: dict, descriptor: Any) -> None:
    """Pin each enum mapping by name, and require every proto member to be mapped.

    The round-trip alone cannot catch a swapped pair of constants; comparing the
    proto member name against the model member name can.
    """
    proto_names = {
        value.name for value in descriptor.values if not value.name.endswith("_UNSPECIFIED")
    }
    assert proto_names == {descriptor.values_by_number[value].name for value in mapping.values()}

    for model_value, grpc_value in mapping.items():
        assert descriptor.values_by_number[grpc_value].name == model_value.name


@pytest.mark.parametrize(
    "model,message",
    [
        (models.DenseVectorConfig, pb2.DenseVectorConfig),
        (models.SparseVectorConfig, pb2.SparseVectorConfig),
        (models.CollectionConfig, pb2.CollectionConfig),
        (models.KeywordIndex, pb2.KeywordIndex),
        (models.KeywordPrefixParams, pb2.KeywordPrefixParams),
        (models.IntegerIndex, pb2.IntegerIndex),
        (models.TextIndex, pb2.TextIndex),
        (models.StopwordsSet, pb2.StopwordsSet),
        (models.SnowballParams, pb2.SnowballParams),
        (models.StemmingAlgorithm, pb2.StemmingAlgorithm),
        # response shapes: a dropped field here silently loses server data
        (models.CollectionInfo, pb2.GetCollectionResponse),
        (models.CollectionSummary, pb2.CollectionSummary),
        (models.CollectionsList, pb2.ListCollectionsResponse),
    ],
)
def test_models_match_proto_messages(model: Any, message: Any) -> None:
    """A field added on either side has to be added on the other."""
    # `type` is the client-side union tag, it has no proto counterpart
    assert set(model_fields(model)) - {"type"} == {
        field.name for field in message.DESCRIPTOR.fields
    }


def test_payload_index_union_covers_every_model() -> None:
    kinds = [index_type().type for index_type in get_args(models.PayloadIndex)]
    assert sorted(kinds) == sorted(
        field.name for field in pb2.PayloadIndexConfig.DESCRIPTOR.oneofs_by_name["index"].fields
    )
