import json
from typing import Any

import pytest
from pydantic import ValidationError

from qdrant_client._pydantic_compat import PYDANTIC_V2, to_dict
from qdrant_client.common.response_parsing import _validator, parse_as_type, parse_response
from qdrant_client.http import models

UUID = "c0a8e1d2-3b4c-4d5e-8f60-718293a4b5c6"
DENSE = [0.25, -1.5, 3.0e-7, 1.0]
# multivectors of different points usually have different numbers of rows
MULTI = [[0.5, 1.5], [2.5, -3.5], [0.125, 0.0]]
MULTI_SHORT = [[1.0, 2.0]]
SPARSE = {"indices": [3, 7], "values": [0.5, 1.25]}
PAYLOAD = {
    "city": "Berlin",
    "tags": ["a", "b"],
    "nested": {"ключ": "значение", "n": 1},
    "none": None,
}


def scored(point_id: Any, **fields: Any) -> dict[str, Any]:
    return {"id": point_id, "version": 3, "score": 0.5, **fields}


POINTS = [
    scored(1, payload=PAYLOAD, vector=DENSE),
    scored(UUID, vector=MULTI),
    scored(2, vector=MULTI_SHORT, shard_key="a", order_value=3),
    scored(3, payload={}, vector={"dense": DENSE, "multi": MULTI, "sparse": SPARSE}),
    scored(4, vector=None),
    scored(5),
]
RECORDS = [
    {"id": 1, "payload": PAYLOAD, "vector": DENSE},
    {"id": UUID, "vector": {"sparse": SPARSE}},
]


def response(result: Any) -> bytes:
    return json.dumps({"result": result, "status": "ok", "time": 0.001}).encode()


QUERY = (models.InlineResponse20022, response({"points": POINTS}))  # query_points
SCROLL = (models.InlineResponse20019, response({"points": RECORDS, "next_page_offset": UUID}))
GROUPS = (  # query_points_groups: hits and lookup share models, which exercises definition refs
    models.InlineResponse20024,
    response(
        {"groups": [{"id": "g", "hits": POINTS, "lookup": RECORDS[0]}, {"id": 7, "hits": []}]}
    ),
)
RETRIEVE = (models.InlineResponse20017, response(RECORDS))
COUNT = (models.InlineResponse20020, response({"count": 5}))


@pytest.mark.parametrize(
    "type_, body",
    [QUERY, SCROLL, GROUPS, RETRIEVE, COUNT],
    ids=["query", "scroll", "groups", "retrieve", "count"],
)
def test_same_result_as_regular_parsing(type_: Any, body: bytes) -> None:
    parsed = parse_response(body, type_)
    expected = parse_as_type(json.loads(body), type_)

    # models compare equal only when their classes match, and exclude_unset compares the fields set
    assert parsed == expected
    assert to_dict(parsed, exclude_unset=True) == to_dict(expected, exclude_unset=True)


@pytest.mark.skipif(not PYDANTIC_V2, reason="the fast path is for pydantic 2 only")
def test_vectors_and_payloads_are_not_copied() -> None:
    # The fast path keeps the decoded lists and dicts, the regular validator copies them. If a pydantic
    # change makes the fast path fall back to the regular validator, results stay correct and this fails.
    type_, body = GROUPS
    raw = json.loads(body)
    parsed = _validator(type_)(raw)

    raw_points, points = raw["result"]["groups"][0]["hits"], parsed.result.groups[0].hits
    assert points[0].vector is raw_points[0]["vector"]
    assert points[0].payload is raw_points[0]["payload"]
    assert points[1].vector is raw_points[1]["vector"]
    assert points[3].vector["dense"] is raw_points[3]["vector"]["dense"]
    assert points[3].vector["multi"] is raw_points[3]["vector"]["multi"]
    assert parsed.result.groups[0].lookup.vector is raw["result"]["groups"][0]["lookup"]["vector"]


def test_invalid_body_raises_the_regular_error() -> None:
    type_ = models.InlineResponse20022
    body = response({"points": [scored(1.5, vector=DENSE)]})

    with pytest.raises(ValidationError) as fast:
        parse_response(body, type_)
    with pytest.raises(ValidationError) as regular:
        parse_as_type(json.loads(body), type_)
    assert str(fast.value) == str(regular.value)
