from qdrant_client import QdrantClient, models


def test_set_payload_keeps_array_elements_independent() -> None:
    client = QdrantClient(":memory:")
    client.create_collection("test", vectors_config={})
    client.upsert(
        "test",
        [models.PointStruct(id=1, vector={}, payload={"items": [{"id": 1}, {"id": 2}]})],
    )
    client.set_payload("test", payload={"metadata": {"color": "blue"}}, points=[1], key="items[]")
    client.set_payload("test", payload={"color": "red"}, points=[1], key="items[0].metadata")

    assert client.retrieve("test", [1])[0].payload == {
        "items": [
            {"id": 1, "metadata": {"color": "red"}},
            {"id": 2, "metadata": {"color": "blue"}},
        ]
    }
