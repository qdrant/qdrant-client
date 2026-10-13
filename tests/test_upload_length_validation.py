"""Regression tests for https://github.com/qdrant/qdrant-client/issues/1486.

`upload_collection` must raise `ValueError` instead of silently truncating
when a supplied `ids`/`payload` iterable ends before `vectors`.
Runs against local mode only — no server required.
"""

import itertools

import numpy as np
import pytest

from qdrant_client import QdrantClient, models

VECTOR_SIZE = 2


@pytest.fixture()
def collection_name():
    client = QdrantClient(":memory:")
    name = "upload_validation"
    client.create_collection(
        collection_name=name,
        vectors_config=models.VectorParams(size=VECTOR_SIZE, distance=models.Distance.DOT),
    )
    return client, name


def _vectors(n):
    return [[float(i), 0.0] for i in range(n)]


def test_short_ids_raise(collection_name):
    client, name = collection_name
    with pytest.raises(ValueError, match="ids iterable is shorter"):
        client.upload_collection(name, vectors=_vectors(3), ids=[1, 2])


def test_short_payload_raises(collection_name):
    client, name = collection_name
    with pytest.raises(ValueError, match="payload iterable is shorter"):
        client.upload_collection(
            name, vectors=_vectors(3), ids=[1, 2, 3], payload=[{"p": 1}, {"p": 2}]
        )


def test_empty_ids_list_raises(collection_name):
    client, name = collection_name
    with pytest.raises(ValueError, match="ids iterable is shorter"):
        client.upload_collection(name, vectors=_vectors(2), ids=[])


def test_short_ids_raise_across_batches(collection_name):
    client, name = collection_name
    with pytest.raises(ValueError, match="ids iterable is shorter"):
        client.upload_collection(name, vectors=_vectors(5), ids=[1, 2, 3, 4], batch_size=2)


def test_short_generator_ids_raise(collection_name):
    client, name = collection_name
    with pytest.raises(ValueError, match="ids iterable is shorter"):
        client.upload_collection(
            name, vectors=([float(i), 0.0] for i in range(3)), ids=iter([1, 2])
        )


def test_short_numpy_ids_raise(collection_name):
    client, name = collection_name
    with pytest.raises(ValueError, match="ids iterable is shorter"):
        client.upload_collection(name, vectors=np.array(_vectors(3)), ids=[1, 2])


def test_valid_upload_unaffected(collection_name):
    client, name = collection_name
    client.upload_collection(
        name, vectors=_vectors(3), ids=[1, 2, 3], payload=[{"p": i} for i in range(3)]
    )
    assert client.count(name, exact=True).count == 3


def test_omitted_ids_and_payload_still_work(collection_name):
    client, name = collection_name
    client.upload_collection(name, vectors=_vectors(3))
    assert client.count(name, exact=True).count == 3


def test_infinite_id_generator_still_supported(collection_name):
    client, name = collection_name
    client.upload_collection(name, vectors=_vectors(5), ids=itertools.count(), batch_size=2)
    assert client.count(name, exact=True).count == 5


def test_longer_ids_than_vectors_unchanged(collection_name):
    # Pre-existing behavior (extra ids ignored) is out of scope for #1486.
    client, name = collection_name
    client.upload_collection(name, vectors=_vectors(3), ids=[1, 2, 3, 4, 5])
    assert client.count(name, exact=True).count == 3
