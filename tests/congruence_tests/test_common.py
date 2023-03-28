from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np

from qdrant_client import QdrantClient
from qdrant_client.client_base import QdrantBase
from qdrant_client.http import models
from qdrant_client.http.models import VectorStruct
from qdrant_client.local.qdrant_local import QdrantLocal
from tests.fixtures.points import generate_records

COLLECTION_NAME = "test_collection"
text_vector_size = 50
image_vector_size = 100
code_vector_size = 80

NUM_VECTORS = 1000


def initialize_fixture_collection(
    client: QdrantBase,
    collection_name: str = COLLECTION_NAME,
    vectors_config: Optional[Union[Dict[str, models.VectorParams], models.VectorParams]] = None,
) -> None:
    if vectors_config is None:
        vectors_config = {
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
        }

    client.recreate_collection(
        collection_name=collection_name,
        vectors_config=vectors_config,
    )


def delete_fixture_collection(client: QdrantBase) -> None:
    client.delete_collection(COLLECTION_NAME)


def generate_fixtures(
    num: Optional[int] = NUM_VECTORS,
    random_ids: bool = False,
    vectors_sizes: Optional[Union[Dict[str, int], int]] = None,
) -> List[models.Record]:
    if vectors_sizes is None:
        vectors_sizes = {
            "text": text_vector_size,
            "image": image_vector_size,
            "code": code_vector_size,
        }
    return generate_records(
        num_records=num or NUM_VECTORS,
        vector_sizes=vectors_sizes,
        with_payload=True,
        random_ids=random_ids,
    )


def compare_collections(
    client_1,
    client_2,
    num_vectors,
    attrs=("vectors_count", "indexed_vectors_count", "points_count"),
):
    collection_1 = client_1.get_collection(COLLECTION_NAME)
    collection_2 = client_2.get_collection(COLLECTION_NAME)

    assert all(getattr(collection_1, attr) == getattr(collection_2, attr) for attr in attrs)

    # num_vectors * 2 to be sure that we have no excess points uploaded
    compare_client_results(
        client_1,
        client_2,
        lambda client: client.scroll(COLLECTION_NAME, limit=num_vectors * 2),
    )


def compare_client_results(
    client1: QdrantBase,
    client2: QdrantBase,
    foo: Callable[[QdrantBase, Any], Any],
    **kwargs: Any,
) -> None:
    def compare_vectors(
        vec1: Optional[VectorStruct], vec2: Optional[VectorStruct], i: int
    ) -> None:
        assert type(vec1) == type(vec2)

        if vec1 is None:
            return

        if isinstance(vec1, dict):
            for key, value in vec1.items():
                assert np.allclose(vec1[key], vec2[key]), (
                    f"res1[{i}].vectors[{key}] = {value}, "
                    f"res2[{i}].vectors[{key}] = {res2_item.vector[key]}"
                )
        else:
            assert np.allclose(
                vec1, vec2
            ), f"res1[{i}].vectors = {vec1}, res2[{i}].vectors = {vec2}"

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

                compare_vectors(res1_item.vector, res2_item.vector, i)

            elif isinstance(res1_item, models.Record) and isinstance(res2_item, models.Record):
                assert (
                    res1_item.id == res2_item.id
                ), f"res1[{i}].id = {res1_item.id}, res2[{i}].id = {res2_item.id}"
                assert (
                    res1_item.payload == res2_item.payload
                ), f"res1[{i}].payload = {res1_item.payload}, res2[{i}].payload = {res2_item.payload}"

                compare_vectors(res1_item.vector, res2_item.vector, i)
            else:
                assert res1[i] == res2[i], f"res1[{i}] = {res1[i]}, res2[{i}] = {res2[i]}"
    else:
        assert res1 == res2


def init_client(
    client: QdrantBase,
    records: List[models.Record],
    collection_name: str = COLLECTION_NAME,
    vectors_config: Optional[Union[Dict[str, models.VectorParams], models.VectorParams]] = None,
) -> None:
    initialize_fixture_collection(
        client, collection_name=collection_name, vectors_config=vectors_config
    )
    client.upload_records(collection_name, records)


def init_local(storage: str = ":memory:") -> QdrantBase:
    client = QdrantClient(location=storage)
    return client


def init_remote() -> QdrantBase:
    client = QdrantClient(host="localhost", port=6333)
    return client
