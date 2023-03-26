from typing import Any, Callable, List, Optional

from qdrant_client import QdrantClient
from qdrant_client.client_base import QdrantBase
from qdrant_client.http import models
from qdrant_client.local.qdrant_local import QdrantLocal
from tests.fixtures.points import generate_records

COLLECTION_NAME = "test_collection"
text_vector_size = 50
image_vector_size = 100
code_vector_size = 80

NUM_VECTORS = 1000


def initialize_fixture_collection(client: QdrantBase) -> None:
    client.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config={
            "text": models.VectorParams(
                size=text_vector_size,
                distance=models.Distance.COSINE,
            ),
            "image": models.VectorParams(
                size=image_vector_size,
                distance=models.Distance.DOT,
            ),
            "code": models.VectorParams(
                size=code_vector_size,
                distance=models.Distance.EUCLID,
            ),
        },
    )


def generate_fixtures(num: Optional[int] = NUM_VECTORS) -> List[models.Record]:
    return generate_records(
        num_records=num or NUM_VECTORS,
        vector_sizes={
            "text": text_vector_size,
            "image": image_vector_size,
            "code": code_vector_size,
        },
        with_payload=True,
        random_ids=False,
    )


def compare_client_results(
    client1: QdrantBase,
    client2: QdrantBase,
    foo: Callable[[QdrantBase, Any], Any],
    **kwargs: Any,
) -> None:
    res1 = foo(client1, **kwargs)
    res2 = foo(client2, **kwargs)

    if isinstance(res1, list):
        assert len(res1) == len(res2), f"len(res1) = {len(res1)}, len(res2) = {len(res2)}"
        for i in range(len(res1)):
            res1_item = res1[i]
            res2_item = res2[i]

            if isinstance(res1_item, models.ScoredPoint) and isinstance(
                res2_item, models.ScoredPoint
            ):
                assert (
                    res1_item.id == res2_item.id
                ), f"res1[{i}].id = {res1_item.id}, res2[{i}].id = {res2_item.id}"
                assert (
                    res1_item.score - res2_item.score < 1e-5
                ), f"res1[{i}].score = {res1_item.score}, res2[{i}].score = {res2_item.score}"
                assert (
                    res1_item.payload == res2_item.payload
                ), f"res1[{i}].payload = {res1_item.payload}, res2[{i}].payload = {res2_item.payload}"
            elif isinstance(res1_item, models.Record) and isinstance(res2_item, models.Record):
                assert (
                    res1_item.id == res2_item.id
                ), f"res1[{i}].id = {res1_item.id}, res2[{i}].id = {res2_item.id}"
                assert (
                    res1_item.payload == res2_item.payload
                ), f"res1[{i}].payload = {res1_item.payload}, res2[{i}].payload = {res2_item.payload}"
                assert (
                    res1_item.vector == res2_item.vector
                ), f"res1[{i}].vectors = {res1_item.vector}, res2[{i}].vectors = {res2_item.vector}"
            else:
                assert res1[i] == res2[i], f"res1[{i}] = {res1[i]}, res2[{i}] = {res2[i]}"
    else:
        assert res1 == res2


def init_client(client: QdrantBase, records: List[models.Record]) -> None:
    initialize_fixture_collection(client)
    client.upload_records(COLLECTION_NAME, records)


def init_local() -> QdrantBase:
    client = QdrantLocal(location=":memory:")
    return client


def init_remote() -> QdrantBase:
    client = QdrantClient(host="localhost", port=6333)
    return client
