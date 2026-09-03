import numpy as np
import pytest

from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.http.models import models
from tests.congruence_tests.test_common import (
    COLLECTION_NAME,
    NUM_VECTORS,
    compare_client_results,
    generate_fixtures,
    generate_sparse_fixtures,
    image_vector_size,
    init_client,
    init_local,
    init_remote,
    sparse_image_vector_size,
    sparse_vectors_config,
)
from tests.fixtures.points import random_sparse_vectors


def test_simple_opt_vectors_search():
    fixture_points = generate_fixtures()

    local_client = init_local()
    init_client(local_client, fixture_points)

    remote_client = init_remote()
    init_client(remote_client, fixture_points)

    ids_to_delete = [x for x in range(NUM_VECTORS) if x % 5 == 0]

    vectors_to_retrieve = [x for x in range(20)]

    local_client.delete_vectors(
        collection_name=COLLECTION_NAME,
        vectors=["image"],
        points=ids_to_delete,
    )
    remote_client.delete_vectors(
        collection_name=COLLECTION_NAME,
        vectors=["image"],
        points=ids_to_delete,
    )

    compare_client_results(
        local_client,
        remote_client,
        lambda c: sorted(
            c.retrieve(
                COLLECTION_NAME,
                vectors_to_retrieve,
                with_payload=False,
                with_vectors=["image", "code"],
            ),
            key=lambda x: x.id,
        ),
    )

    new_vector = np.random.rand(image_vector_size).tolist()
    update_vectors = [
        models.PointVectors(
            id=i,
            vector={"image": new_vector},
        )
        for i in range(6)
    ]

    local_client.update_vectors(
        collection_name=COLLECTION_NAME,
        points=update_vectors,
    )

    remote_client.update_vectors(
        collection_name=COLLECTION_NAME,
        points=update_vectors,
    )

    compare_client_results(
        local_client,
        remote_client,
        lambda c: sorted(
            c.retrieve(
                COLLECTION_NAME,
                vectors_to_retrieve,
                with_payload=False,
                with_vectors=["image", "code"],
            ),
            key=lambda x: x.id,
        ),
    )


def test_simple_opt_sparse_vectors_search():
    fixture_points = generate_sparse_fixtures()

    local_client = init_local()
    init_client(
        local_client,
        fixture_points,
        vectors_config={},
        sparse_vectors_config=sparse_vectors_config,
    )

    remote_client = init_remote()
    init_client(
        remote_client,
        fixture_points,
        vectors_config={},
        sparse_vectors_config=sparse_vectors_config,
    )

    ids_to_delete = [x for x in range(NUM_VECTORS) if x % 5 == 0]

    vectors_to_retrieve = [x for x in range(20)]

    local_client.delete_vectors(
        collection_name=COLLECTION_NAME,
        vectors=["sparse-image"],
        points=ids_to_delete,
    )
    remote_client.delete_vectors(
        collection_name=COLLECTION_NAME,
        vectors=["sparse-image"],
        points=ids_to_delete,
    )

    compare_client_results(
        local_client,
        remote_client,
        lambda c: sorted(
            c.retrieve(
                COLLECTION_NAME,
                vectors_to_retrieve,
                with_payload=False,
                with_vectors=["sparse-image", "sparse-text"],
            ),
            key=lambda x: x.id,
        ),
    )

    new_vector = random_sparse_vectors({"sparse-image": sparse_image_vector_size})
    update_vectors = [
        models.PointVectors(
            id=i,
            vector=new_vector,
        )
        for i in range(6)
    ]

    local_client.update_vectors(
        collection_name=COLLECTION_NAME,
        points=update_vectors,
    )

    remote_client.update_vectors(
        collection_name=COLLECTION_NAME,
        points=update_vectors,
    )

    compare_client_results(
        local_client,
        remote_client,
        lambda c: sorted(
            c.retrieve(
                COLLECTION_NAME,
                vectors_to_retrieve,
                with_payload=False,
                with_vectors=["sparse-image", "sparse-text"],
            ),
            key=lambda x: x.id,
        ),
    )


def test_point_id_input_with_missing_vector():
    vectors_config = {
        "dense": models.VectorParams(size=4, distance=models.Distance.COSINE),
        "multi": models.VectorParams(
            size=4,
            distance=models.Distance.COSINE,
            multivector_config=models.MultiVectorConfig(
                comparator=models.MultiVectorComparator.MAX_SIM
            ),
        ),
    }
    # point 1 has no value for either named vector, it can neither be scored nor be used as a query
    points = [
        models.PointStruct(id=1, vector={}),
        models.PointStruct(
            id=2, vector={"dense": [1.0, 0.0, 0.0, 0.0], "multi": [[1.0, 0.0, 0.0, 0.0]]}
        ),
        models.PointStruct(
            id=3, vector={"dense": [0.0, 1.0, 0.0, 0.0], "multi": [[0.0, 1.0, 0.0, 0.0]]}
        ),
    ]

    local_client = init_local()
    init_client(local_client, points, vectors_config=vectors_config)

    remote_client = init_remote()
    init_client(remote_client, points, vectors_config=vectors_config)

    queries = [
        1,
        models.RecommendQuery(
            recommend=models.RecommendInput(
                positive=[1], strategy=models.RecommendStrategy.BEST_SCORE
            )
        ),
        models.DiscoverQuery(
            discover=models.DiscoverInput(
                target=1, context=models.ContextPair(positive=2, negative=3)
            )
        ),
        models.ContextQuery(context=models.ContextPair(positive=1, negative=3)),
        models.RelevanceFeedbackQuery(
            relevance_feedback=models.RelevanceFeedbackInput(
                target=2,
                feedback=[models.FeedbackItem(example=1, score=0.9)],
                strategy=models.NaiveFeedbackStrategy(
                    naive=models.NaiveFeedbackStrategyParams(a=0.5, b=1.0, c=0.7)
                ),
            )
        ),
    ]

    for using in ("dense", "multi"):
        for query in queries:
            with pytest.raises(ValueError):
                local_client.query_points(COLLECTION_NAME, query=query, using=using)

            with pytest.raises(UnexpectedResponse):
                remote_client.query_points(COLLECTION_NAME, query=query, using=using)
