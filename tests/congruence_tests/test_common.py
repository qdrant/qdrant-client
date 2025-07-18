import math
from typing import Any, Callable, Optional, Union

import numpy as np

from qdrant_client import QdrantClient
from qdrant_client.client_base import QdrantBase
from qdrant_client.conversions import common_types as types
from qdrant_client.http import models
from qdrant_client.http.models import SparseVector, VectorStruct
from tests.congruence_tests.settings import TIMEOUT
from tests.fixtures.points import generate_points

COLLECTION_NAME = "congruence_test_collection"

# dense vectors sizes
text_vector_size = 50
image_vector_size = 100
code_vector_size = 80

# sparse vectors sizes
sparse_text_vector_size = 100
sparse_image_vector_size = 1_000
sparse_code_vector_size = 10_000

# number of vectors to generate
NUM_VECTORS = 1000

dense_vectors_config = {
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
# default sparse vectors config
sparse_vectors_config = {
    "sparse-text": models.SparseVectorParams(),
    "sparse-image": models.SparseVectorParams(),
    "sparse-code": models.SparseVectorParams(),
}

dense_vectors_sizes = {
    "text": text_vector_size,
    "image": image_vector_size,
    "code": code_vector_size,
}

sparse_vectors_sizes = {
    "sparse-text": sparse_text_vector_size,
    "sparse-image": sparse_image_vector_size,
    "sparse-code": sparse_code_vector_size,
}

multivectors_sizes = {
    "multi-text": text_vector_size,
    "multi-image": image_vector_size,
    "multi-code": code_vector_size,
}

multi_vector_config = {
    "multi-text": models.VectorParams(
        size=text_vector_size,
        distance=models.Distance.COSINE,
        multivector_config=models.MultiVectorConfig(
            comparator=models.MultiVectorComparator.MAX_SIM,
        ),
    ),
    "multi-image": models.VectorParams(
        size=image_vector_size,
        distance=models.Distance.DOT,
        multivector_config=models.MultiVectorConfig(
            comparator=models.MultiVectorComparator.MAX_SIM,
        ),
    ),
    "multi-code": models.VectorParams(
        size=code_vector_size,
        distance=models.Distance.EUCLID,
        multivector_config=models.MultiVectorConfig(
            comparator=models.MultiVectorComparator.MAX_SIM,
        ),
    ),
}


def initialize_fixture_collection(
    client: QdrantBase,
    collection_name: str = COLLECTION_NAME,
    vectors_config: Optional[Union[dict[str, models.VectorParams], models.VectorParams]] = None,
    sparse_vectors_config: Optional[dict[str, models.SparseVectorParams]] = None,
) -> None:
    if vectors_config is None:
        vectors_config = dense_vectors_config
    # no sparse vector config generated by default
    if client.collection_exists(collection_name):
        client.delete_collection(collection_name=collection_name, timeout=TIMEOUT)
    client.create_collection(
        collection_name=collection_name,
        vectors_config=vectors_config,
        timeout=TIMEOUT,
        sparse_vectors_config=sparse_vectors_config,
    )


def delete_fixture_collection(client: QdrantBase) -> None:
    if client.collection_exists(COLLECTION_NAME):
        client.delete_collection(COLLECTION_NAME)


def generate_fixtures(
    num: Optional[int] = NUM_VECTORS,
    random_ids: bool = False,
    vectors_sizes: Optional[Union[dict[str, int], int]] = None,
    skip_vectors: bool = False,
) -> list[models.PointStruct]:
    if vectors_sizes is None:
        vectors_sizes = dense_vectors_sizes
    return generate_points(
        num_points=num or NUM_VECTORS,
        vector_sizes=vectors_sizes,
        with_payload=True,
        random_ids=random_ids,
        skip_vectors=skip_vectors,
        sparse=False,
    )


def generate_sparse_fixtures(
    num: Optional[int] = NUM_VECTORS,
    random_ids: bool = False,
    vectors_sizes: Optional[Union[dict[str, int], int]] = None,
    skip_vectors: bool = False,
    with_payload: bool = True,
    even_sparse: bool = True,
) -> list[models.PointStruct]:
    if vectors_sizes is None:
        vectors_sizes = sparse_vectors_sizes
    return generate_points(
        num_points=num or NUM_VECTORS,
        vector_sizes=vectors_sizes,
        with_payload=with_payload,
        random_ids=random_ids,
        skip_vectors=skip_vectors,
        sparse=True,
        even_sparse=even_sparse,
    )


def generate_multivector_fixtures(
    num: Optional[int] = NUM_VECTORS,
    random_ids: bool = False,
    vectors_sizes: Optional[Union[dict[str, int], int]] = None,
    skip_vectors: bool = False,
    with_payload: bool = True,
) -> list[models.PointStruct]:
    if vectors_sizes is None:
        vectors_sizes = multivectors_sizes
    return generate_points(
        num_points=num or NUM_VECTORS,
        vector_sizes=vectors_sizes,
        with_payload=with_payload,
        random_ids=random_ids,
        skip_vectors=skip_vectors,
        multivector=True,
    )


def compare_collections(
    client_1,
    client_2,
    num_vectors,
    attrs=("indexed_vectors_count", "points_count"),
    collection_name: str = COLLECTION_NAME,
):
    collection_1 = client_1.get_collection(collection_name)
    collection_2 = client_2.get_collection(collection_name)

    for attr in attrs:
        if attr != "indexed_vectors_count":
            assert getattr(collection_1, attr) == getattr(collection_2, attr), (
                f"client_1.{attr} = {getattr(collection_1, attr)}, "
                f"client_2.{attr} = {getattr(collection_2, attr)}"
            )

    # num_vectors * 2 to be sure that we have no excess points uploaded
    compare_client_results(
        client_1,
        client_2,
        lambda client: client.scroll(collection_name, with_vectors=True, limit=num_vectors * 2),
    )


def compare_vectors(vec1: Optional[VectorStruct], vec2: Optional[VectorStruct], i: int) -> None:
    assert type(vec1) is type(vec2)
    if vec1 is None:
        return

    if isinstance(vec1, dict):
        assert (
            vec1.keys() == vec2.keys()
        ), f"res1[{i}].vectors.keys() = {list(vec1.keys())}, res2[{i}].vectors.keys() = {list(vec2.keys())}"
        for key, value in vec1.items():
            if isinstance(value, SparseVector):
                assert vec1[key].indices == vec2[key].indices, (
                    f"res1[{i}].vectors[{key}].indices = {value}, "
                    f"res2[{i}].vectors[{key}].indices = {vec2[key].indices}"
                )
                assert np.allclose(vec1[key].values, vec2[key].values, atol=1.0e-3), (
                    f"res1[{i}].vectors[{key}].values = {value}, "
                    f"res2[{i}].vectors[{key}].values = {vec2[key].values}"
                )
            else:
                assert np.allclose(vec1[key], vec2[key], atol=1.0e-3), (
                    f"res1[{i}].vectors[{key}] = {value}, "
                    f"res2[{i}].vectors[{key}] = {vec2[key]}"
                )
    else:
        assert np.allclose(
            vec1, vec2, atol=1.0e-3
        ), f"res1[{i}].vectors = {vec1}, res2[{i}].vectors = {vec2}"


def compare_scored_record(
    point1: models.ScoredPoint,
    point2: models.ScoredPoint,
    idx: int,
    rel_tol: float = 1e-4,
    abs_tol: float = 1e-6,
) -> None:
    # This is a special case, likely the result of scroll or context search
    # We need to ensure ordering by another field
    is_score_zero = point1.score == 0.0 and point2.score == 0.0

    assert math.isclose(
        np.float32(point1.score), np.float32(point2.score), rel_tol=rel_tol, abs_tol=abs_tol
    ), f"point1[{idx}].score = {point1.score}, point2[{idx}].score = {point2.score}, rel_tol={rel_tol}"

    assert (
        point1.order_value == point2.order_value
    ), f"point1[{idx}].order_value = {point1.order_value}, point2[{idx}].order_value = {point2.order_value}"

    if is_score_zero:
        assert (
            point1.id == point2.id
        ), f"point1[{idx}].id = {point1.id}, point2[{idx}].id = {point2.id}"

    if point1.id == point2.id:
        # same id means same payload
        assert (
            point1.payload == point2.payload
        ), f"id:{point1.id} point1[{idx}].payload = {point1.payload}, point2[{idx}].payload = {point2.payload}"

        compare_vectors(point1.vector, point2.vector, idx)


def compare_records(res1: list, res2: list, rel_tol: float = 1e-4, abs_tol: float = 1e-6) -> None:
    assert len(res1) == len(res2), f"len(res1) = {len(res1)}, len(res2) = {len(res2)}"
    for i in range(len(res2)):
        res1_item = res1[i]
        res2_item = res2[i]

        if isinstance(res1_item, list) and isinstance(res2_item, list):
            compare_records(res1_item, res2_item)

        elif isinstance(res1_item, models.QueryResponse) and isinstance(
            res2_item, models.QueryResponse
        ):
            compare_records(res1_item.points, res2_item.points, rel_tol=rel_tol, abs_tol=abs_tol)

        elif isinstance(res1_item, models.ScoredPoint) and isinstance(
            res2_item, models.ScoredPoint
        ):
            compare_scored_record(res1_item, res2_item, i, rel_tol=rel_tol, abs_tol=abs_tol)

        elif isinstance(res1_item, models.Record) and isinstance(res2_item, models.Record):
            assert (
                res1_item.id == res2_item.id
            ), f"res1[{i}].id = {res1_item.id}, res2[{i}].id = {res2_item.id}"
            # same id means same payload
            assert (
                res1_item.payload == res2_item.payload
            ), f"id:{res1_item.id} res1[{i}].payload = {res1_item.payload}, res2[{i}].payload = {res2_item.payload}"

            compare_vectors(res1_item.vector, res2_item.vector, i)
        else:
            assert res1[i] == res2[i], f"res1[{i}] = {res1[i]}, res2[{i}] = {res2[i]}"


def compare_client_results(
    client1: QdrantBase,
    client2: QdrantBase,
    foo: Callable[[QdrantBase, Any], Any],
    **kwargs: Any,
) -> None:
    # context search can have many points with the same 0.0 score
    is_context_search = kwargs.pop("is_context_search", False)

    # get results from both clients
    res1 = foo(client1, **kwargs)
    res2 = foo(client2, **kwargs)

    # compare scroll results
    if isinstance(res1, tuple) and len(res1) == 2:
        if isinstance(res1[0], list) and (res1[1] is None or isinstance(res1[1], types.PointId)):
            res1, offset1 = res1
            res2, offset2 = res2
            assert offset1 == offset2, f"offset1 = {offset1}, offset2 = {offset2}"

    if isinstance(res1, list):
        if is_context_search is True:
            sorted_1 = sorted(res1, key=lambda x: (x.id))
            sorted_2 = sorted(res2, key=lambda x: (x.id))
            compare_records(sorted_1, sorted_2, abs_tol=1e-5)
        else:
            compare_records(res1, res2)
    elif isinstance(res1, models.QueryResponse) and isinstance(res2, models.QueryResponse):
        if is_context_search is True:
            sorted_1 = sorted(res1.points, key=lambda x: (x.id))
            sorted_2 = sorted(res2.points, key=lambda x: (x.id))
            compare_records(sorted_1, sorted_2, abs_tol=1e-5)
        else:
            compare_records(res1.points, res2.points)
    elif isinstance(res1, models.SearchMatrixOffsetsResponse):
        assert res1.ids == res2.ids, f"res1.ids = {res1.ids}, res2.ids = {res2.ids}"
        # compare scores with margin
        assert np.allclose(
            res1.scores, res2.scores, atol=1e-4
        ), f"res1.scores = {res1.scores}, res2.scores = {res2.scores}"
        # we don't compare offsets_col, because due to slight differences in score computation in
        # local and remote modes, ordering can be different
        assert (
            res1.offsets_row == res2.offsets_row
        ), f"res1.offsets_row = {res1.offsets_row}, res2.offsets_row = {res2.offsets_row}"
    elif isinstance(res1, models.SearchMatrixPairsResponse):
        assert len(res1.pairs) == len(
            res2.pairs
        ), f"len(res1.pairs) = {len(res1.pairs)}, len(res2.pairs) = {len(res2.pairs)}"
        for pair_1, pair_2 in zip(res1.pairs, res2.pairs):
            # we don't compare pair_1.b to pair_2.b, because due to slight differences in score computation in
            # local and remote modes, ordering can be different
            assert pair_1.a == pair_2.a, f"pair_1.a = {pair_1.a}, pair_2.a = {pair_2.a}"
            # compare scores with margin
            assert math.isclose(
                pair_1.score, pair_2.score, rel_tol=1e-4
            ), f"pair_1.score = {pair_1.score}, pair_2.score = {pair_2.score}"
    elif isinstance(res1, models.GroupsResult):
        groups_1 = sorted(res1.groups, key=lambda x: (x.hits[0].score, x.id))
        groups_2 = sorted(res2.groups, key=lambda x: (x.hits[0].score, x.id))

        assert len(groups_1) == len(
            groups_2
        ), f"len(groups_1) = {len(groups_1)}, len(groups_2) = {len(groups_2)}"

        for i in range(len(groups_1)):
            group_1 = groups_1[i]
            group_2 = groups_2[i]

            assert (
                group_1.hits[0].score - group_2.hits[0].score < 1e-4
            ), f"groups_1[{i}].hits[0].score = {group_1.hits[0].score}, groups_2[{i}].hits[0].score = {group_2.hits[0].score}"

            # We can't assert ids because they are not stable, order of groups with same score is guaranteed
            # assert (
            #     group_1.id == group_2.id
            # ), f"groups_1[{i}].id = {group_1.id}, groups_2[{i}].id = {group_2.id}"

            if group_1.id == group_2.id:
                compare_records(group_1.hits, group_2.hits)
            else:
                # If group ids are different, but scores are the same, we assume that the top hits are the same
                compare_scored_record(group_1.hits[0], group_2.hits[0], 0)
    else:
        assert res1 == res2


def init_client(
    client: QdrantBase,
    points: list[models.PointStruct],
    collection_name: str = COLLECTION_NAME,
    vectors_config: Optional[Union[dict[str, models.VectorParams], models.VectorParams]] = None,
    sparse_vectors_config: Optional[dict[str, models.SparseVectorParams]] = None,
) -> None:
    initialize_fixture_collection(
        client=client,
        collection_name=collection_name,
        vectors_config=vectors_config,
        sparse_vectors_config=sparse_vectors_config,
    )
    client.upload_points(collection_name, points, wait=True)


def init_local(storage: Optional[str] = None) -> QdrantClient:
    if storage is None or storage == ":memory:":
        client = QdrantClient(location=":memory:")
    else:
        client = QdrantClient(path=storage)
    return client


def init_remote(prefer_grpc: bool = False) -> QdrantClient:
    client = QdrantClient(host="localhost", port=6333, timeout=30, prefer_grpc=prefer_grpc)
    return client
