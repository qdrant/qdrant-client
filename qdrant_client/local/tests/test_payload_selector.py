"""`with_payload=[]` selects no payload fields, it does not drop the payload.

`query_points` documents the selector as `False` meaning "do not attach any payload" and a list of
strings meaning "include only specified fields", so an empty list is a selection of nothing and the
record still carries a payload - an empty one. Local mode tested the selector with
`if not with_payload`, so `[]` fell into the `False` branch and came back with `payload=None`, while
the same request spelled `PayloadSelectorInclude(include=[])` already returned `{}` and
`_get_vectors` in the same file already separates `is False or is None` from an empty list. Over
gRPC, `RestToGrpc.convert_with_payload_interface` maps a list to
`include=PayloadIncludeSelector(fields=<that list>)` and only uses `enable` for a bool, so the
remote path never reads an empty list as "off".
"""

import pytest

from qdrant_client import QdrantClient, models

PAYLOAD = {"a": 1, "b": 2}


@pytest.fixture
def client() -> QdrantClient:
    client = QdrantClient(":memory:")
    client.create_collection(
        "test", vectors_config=models.VectorParams(size=2, distance=models.Distance.DOT)
    )
    client.upsert("test", [models.PointStruct(id=1, vector=[1.0, 2.0], payload=PAYLOAD)])
    return client


@pytest.mark.parametrize("selector", [[], models.PayloadSelectorInclude(include=[])])
def test_empty_include_selects_no_fields_but_keeps_payload(
    client: QdrantClient, selector: models.WithPayloadInterface
) -> None:
    records, _ = client.scroll("test", limit=3, with_payload=selector)
    assert [record.payload for record in records] == [{}]

    retrieved = client.retrieve("test", ids=[1], with_payload=selector)
    assert [record.payload for record in retrieved] == [{}]

    points = client.query_points("test", query=[1.0, 2.0], with_payload=selector).points
    assert [point.payload for point in points] == [{}]


def test_false_still_drops_payload(client: QdrantClient) -> None:
    records, _ = client.scroll("test", limit=3, with_payload=False)
    assert [record.payload for record in records] == [None]

    points = client.query_points("test", query=[1.0, 2.0], with_payload=False).points
    assert [point.payload for point in points] == [None]
