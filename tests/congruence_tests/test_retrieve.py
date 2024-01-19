import random

from qdrant_client.http.models import PayloadSelectorExclude, PayloadSelectorInclude
from tests.congruence_tests.test_common import (
    COLLECTION_NAME,
    compare_client_results,
    generate_fixtures,
    generate_sparse_fixtures,
    init_client,
    init_local,
    init_remote,
    sparse_vectors_config,
)


def test_retrieve(local_client, remote_client) -> None:
    num_vectors = 1000
    fixture_points = generate_fixtures(num_vectors)
    keys = list(fixture_points[0].payload.keys())

    local_client.upload_points(COLLECTION_NAME, fixture_points)
    remote_client.upload_points(COLLECTION_NAME, fixture_points, wait=True)

    id_ = random.randint(0, num_vectors)

    compare_client_results(
        local_client, remote_client, lambda c: c.retrieve(COLLECTION_NAME, [id_])
    )
    compare_client_results(
        local_client,
        remote_client,
        lambda c: c.retrieve(COLLECTION_NAME, [id_], with_payload=False),
    )
    # with_vectors is not tested with `True` because `text` vectors are used with Cosine distance,
    # and we do not normalize them in local version

    compare_client_results(
        local_client,
        remote_client,
        lambda c: c.retrieve(COLLECTION_NAME, [id_], with_vectors=["image", "code"]),
    )
    compare_client_results(
        local_client,
        remote_client,
        lambda c: c.retrieve(
            COLLECTION_NAME, [id_], with_vectors=["image", "code"], with_payload=False
        ),
    )

    sample_keys = random.sample(keys, 3)
    compare_client_results(
        local_client,
        remote_client,
        lambda c: c.retrieve(COLLECTION_NAME, [id_], with_payload=sample_keys),
    )
    compare_client_results(
        local_client,
        remote_client,
        lambda c: c.retrieve(
            COLLECTION_NAME,
            [id_],
            with_payload=PayloadSelectorInclude(include=sample_keys),
        ),
    )
    compare_client_results(
        local_client,
        remote_client,
        lambda c: c.retrieve(
            COLLECTION_NAME,
            [id_],
            with_payload=PayloadSelectorExclude(exclude=sample_keys),
        ),
    )


def test_sparse_retrieve() -> None:
    num_vectors = 1000
    fixture_points = generate_sparse_fixtures(num_vectors)

    local_client = init_local()
    init_client(local_client, fixture_points, sparse_vectors_config=sparse_vectors_config)

    remote_client = init_remote()
    init_client(remote_client, fixture_points, sparse_vectors_config=sparse_vectors_config)

    keys = list(fixture_points[0].payload.keys())

    local_client.upload_points(COLLECTION_NAME, fixture_points)
    remote_client.upload_points(COLLECTION_NAME, fixture_points, wait=True)

    id_ = random.randint(0, num_vectors)

    compare_client_results(
        local_client, remote_client, lambda c: c.retrieve(COLLECTION_NAME, [id_])
    )
    compare_client_results(
        local_client,
        remote_client,
        lambda c: c.retrieve(COLLECTION_NAME, [id_], with_payload=False),
    )
    # with_vectors is not tested with `True` because `text` vectors are used with Cosine distance,
    # and we do not normalize them in local version

    compare_client_results(
        local_client,
        remote_client,
        lambda c: c.retrieve(COLLECTION_NAME, [id_], with_vectors=["sparse-image", "sparse-code"]),
    )
    compare_client_results(
        local_client,
        remote_client,
        lambda c: c.retrieve(
            COLLECTION_NAME,
            [id_],
            with_vectors=["sparse-image", "sparse-code"],
            with_payload=False,
        ),
    )

    sample_keys = random.sample(keys, 3)
    compare_client_results(
        local_client,
        remote_client,
        lambda c: c.retrieve(COLLECTION_NAME, [id_], with_payload=sample_keys),
    )
    compare_client_results(
        local_client,
        remote_client,
        lambda c: c.retrieve(
            COLLECTION_NAME,
            [id_],
            with_payload=PayloadSelectorInclude(include=sample_keys),
        ),
    )
    compare_client_results(
        local_client,
        remote_client,
        lambda c: c.retrieve(
            COLLECTION_NAME,
            [id_],
            with_payload=PayloadSelectorExclude(exclude=sample_keys),
        ),
    )
