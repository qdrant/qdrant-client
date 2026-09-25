from qdrant_client import QdrantClient, models


def test_quoted_payload_key_selector() -> None:
    client = QdrantClient(":memory:")
    client.create_collection("quoted_payload", vectors_config={})
    client.upsert(
        "quoted_payload",
        [models.PointStruct(id=1, vector={}, payload={"a.b": {"x": 1, "y": 2}, "other": 3})],
    )

    included = client.retrieve(
        "quoted_payload",
        [1],
        with_payload=models.PayloadSelectorInclude(include=['"a.b".x']),
    )[0]
    assert included.payload == {"a.b": {"x": 1}}

    excluded = client.retrieve(
        "quoted_payload",
        [1],
        with_payload=models.PayloadSelectorExclude(exclude=['"a.b".x']),
    )[0]
    assert excluded.payload == {"a.b": {"y": 2}, "other": 3}


def test_quoted_key_does_not_select_nested_path() -> None:
    client = QdrantClient(":memory:")
    client.create_collection("quoted_payload", vectors_config={})
    client.upsert(
        "quoted_payload",
        [models.PointStruct(id=1, vector={}, payload={"a.b": 1, "a": {"b": 2}})],
    )

    quoted = client.retrieve(
        "quoted_payload",
        [1],
        with_payload=models.PayloadSelectorInclude(include=['"a.b"']),
    )[0]
    assert quoted.payload == {"a.b": 1}

    nested = client.retrieve(
        "quoted_payload",
        [1],
        with_payload=models.PayloadSelectorInclude(include=["a.b"]),
    )[0]
    assert nested.payload == {"a": {"b": 2}}


def test_quoted_key_inside_array() -> None:
    client = QdrantClient(":memory:")
    client.create_collection("quoted_payload", vectors_config={})
    client.upsert(
        "quoted_payload",
        [
            models.PointStruct(
                id=1,
                vector={},
                payload={"items": [{"a.b": 1, "other": 2}, {"a.b": 3, "other": 4}]},
            )
        ],
    )

    included = client.retrieve(
        "quoted_payload",
        [1],
        with_payload=models.PayloadSelectorInclude(include=['items[]."a.b"']),
    )[0]
    assert included.payload == {"items": [{"a.b": 1}, {"a.b": 3}]}

    excluded = client.retrieve(
        "quoted_payload",
        [1],
        with_payload=models.PayloadSelectorExclude(exclude=['items[]."a.b"']),
    )[0]
    assert excluded.payload == {"items": [{"other": 2}, {"other": 4}]}
