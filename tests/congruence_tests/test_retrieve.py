import random

import pytest

from qdrant_client.http import models
from qdrant_client.http.models import PayloadSelectorExclude, PayloadSelectorInclude
from tests.congruence_tests.test_common import (
    COLLECTION_NAME,
    compare_client_results,
    delete_fixture_collection,
    generate_fixtures,
    init_local,
    init_remote,
    initialize_fixture_collection,
)


@pytest.fixture
def local_client():
    client = init_local()
    initialize_fixture_collection(client)
    yield client
    delete_fixture_collection(client)


@pytest.fixture
def remote_client():
    client = init_remote()
    initialize_fixture_collection(client)
    yield client
    delete_fixture_collection(client)


def test_retrieve(local_client, remote_client) -> None:
    num_vectors = 1000
    fixture_records = generate_fixtures(num_vectors)
    keys = list(fixture_records[0].payload.keys())

    local_client.upload_records(COLLECTION_NAME, fixture_records)
    remote_client.upload_records(COLLECTION_NAME, fixture_records)

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
