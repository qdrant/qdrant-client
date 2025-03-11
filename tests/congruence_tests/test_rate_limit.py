from time import sleep

import numpy as np
import pytest

from qdrant_client.common.client_exceptions import ResourceQuotaExceeded, ResourceExhaustedResponse
from qdrant_client.http import models
from tests.congruence_tests.test_common import (
    COLLECTION_NAME,
    compare_collections,
    generate_fixtures,
    init_remote,
)


UPLOAD_NUM_VECTORS = 100
WRITE_LIMIT = 3
READ_LIMIT = 2


def test_upsert_hits_large_request_limit(remote_client):
    grpc_client = init_remote(prefer_grpc=True)
    points = generate_fixtures(UPLOAD_NUM_VECTORS)
    ids, payload = [], []
    vectors = {}
    for point in points:
        ids.append(point.id)
        payload.append(point.payload)
        for vector_name, vector in point.vector.items():
            if vector_name not in vectors:
                vectors[vector_name] = []
            vectors[vector_name].append(vector)

    points_batch = models.Batch(
        ids=ids,
        vectors=vectors,
        payloads=payload,
    )

    remote_client.update_collection(
        collection_name=COLLECTION_NAME,
        strict_mode_config=models.StrictModeConfig(
            enabled=True, read_rate_limit=READ_LIMIT, write_rate_limit=WRITE_LIMIT
        ),
    )

    with pytest.raises(
        ResourceQuotaExceeded,
        match="Write rate limit exceeded, request larger than than rate limiter capacity, please try to split your request",
    ):
        remote_client.upsert(COLLECTION_NAME, points_batch)

    with pytest.raises(
        ResourceQuotaExceeded,
        match="Write rate limit exceeded, request larger than than rate limiter capacity, please try to split your request",
    ):
        grpc_client.upsert(COLLECTION_NAME, points_batch)


def test_upsert_hits_write_rate_limit(remote_client):
    grpc_client = init_remote(prefer_grpc=True)
    points = generate_fixtures(WRITE_LIMIT)
    ids, payload = [], []
    vectors = {}
    for point in points:
        ids.append(point.id)
        payload.append(point.payload)
        for vector_name, vector in point.vector.items():
            if vector_name not in vectors:
                vectors[vector_name] = []
            vectors[vector_name].append(vector)

    points_batch = models.Batch(
        ids=ids,
        vectors=vectors,
        payloads=payload,
    )

    remote_client.update_collection(
        collection_name=COLLECTION_NAME,
        strict_mode_config=models.StrictModeConfig(
            enabled=True, read_rate_limit=READ_LIMIT, write_rate_limit=WRITE_LIMIT
        ),
    )

    flag = False
    time_to_sleep = 0
    try:
        for _ in range(10):
            remote_client.upsert(COLLECTION_NAME, points_batch)
    except ResourceExhaustedResponse as ex:
        print(ex.message)
        assert ex.retry_after_s > 0, f"Unexpected retry_after_s value: {ex.retry_after_s}"
        flag = True
        time_to_sleep = int(ex.retry_after_s)

    if flag:
        # verify next response after sleep succeeds
        sleep(time_to_sleep)
        remote_client.upsert(COLLECTION_NAME, points_batch)
    else:
        raise AssertionError(
            "No ResourceExhaustedResponse exception was raised for remote_client."
        )

    flag = False
    try:
        for _ in range(10):
            grpc_client.upsert(COLLECTION_NAME, points_batch)
    except ResourceExhaustedResponse as ex:
        print(f"{ex.message}")
        assert ex.retry_after_s > 0, f"Unexpected retry_after_s value: {ex.retry_after_s}"
        flag = True

    if flag:
        # verify next response after sleep succeeds
        sleep(time_to_sleep)
        grpc_client.upsert(COLLECTION_NAME, points_batch)
    else:
        raise AssertionError("No ResourceExhaustedResponse exception was raised for grpc_client.")


def test_upload_collection_succeeds_with_limits(local_client, remote_client):
    grpc_client = init_remote(prefer_grpc=True)

    points = generate_fixtures(10)

    vectors = []
    payload = []
    ids = []
    for point in points:
        (ids.append(point.id),)
        vectors.append(point.vector)
        payload.append(point.payload)

    remote_client.update_collection(
        collection_name=COLLECTION_NAME,
        strict_mode_config=models.StrictModeConfig(
            enabled=True, read_rate_limit=READ_LIMIT, write_rate_limit=WRITE_LIMIT
        ),
    )

    local_client.upload_collection(COLLECTION_NAME, vectors, payload, ids=ids)
    remote_client.upload_collection(
        COLLECTION_NAME, vectors, payload, ids=ids, wait=True, max_retries=1
    )
    grpc_client.upload_collection(
        COLLECTION_NAME, vectors, payload, ids=ids, wait=True, max_retries=1
    )

    compare_collections(local_client, remote_client, UPLOAD_NUM_VECTORS)
    compare_collections(local_client, grpc_client, UPLOAD_NUM_VECTORS)


def test_upload_points_succeeds_with_limits(local_client, remote_client):
    grpc_client = init_remote(prefer_grpc=True)
    points = generate_fixtures(10)

    remote_client.update_collection(
        collection_name=COLLECTION_NAME,
        strict_mode_config=models.StrictModeConfig(
            enabled=True, read_rate_limit=READ_LIMIT, write_rate_limit=WRITE_LIMIT
        ),
    )

    local_client.upload_points(COLLECTION_NAME, points)
    remote_client.upload_points(COLLECTION_NAME, points, wait=True, max_retries=1)
    grpc_client.upload_points(COLLECTION_NAME, points, wait=True, max_retries=1)

    compare_collections(local_client, remote_client, UPLOAD_NUM_VECTORS)
    compare_collections(local_client, grpc_client, UPLOAD_NUM_VECTORS)


def test_query_hits_read_rate_limit(remote_client):
    grpc_client = init_remote(prefer_grpc=True)

    dense_vector_query_batch_text = []
    for _ in range(READ_LIMIT):
        dense_vector_query_batch_text.append(
            models.QueryRequest(
                query=np.random.random(50).tolist(),
                prefetch=models.Prefetch(
                    query=np.random.random(50).tolist(), limit=5, using="text"
                ),
                limit=5,
                using="text",
                with_payload=True,
            )
        )

    remote_client.update_collection(
        collection_name=COLLECTION_NAME,
        strict_mode_config=models.StrictModeConfig(
            enabled=True, read_rate_limit=READ_LIMIT, write_rate_limit=WRITE_LIMIT
        ),
    )

    flag = False
    time_to_sleep = 0
    try:
        for _ in range(10):
            remote_client.query_batch_points(
                collection_name=COLLECTION_NAME, requests=dense_vector_query_batch_text
            )
    except ResourceExhaustedResponse as ex:
        print(ex.message)
        assert ex.retry_after_s > 0, f"Unexpected retry_after_s value: {ex.retry_after_s}"
        flag = True
        time_to_sleep = int(ex.retry_after_s)

    if flag:
        # verify next response after sleep succeeds
        sleep(time_to_sleep)
        remote_client.query_batch_points(
            collection_name=COLLECTION_NAME, requests=dense_vector_query_batch_text
        )
    else:
        raise AssertionError(
            "No ResourceExhaustedResponse exception was raised for remote_client."
        )

    flag = False
    time_to_sleep = 0
    try:
        for _ in range(10):
            grpc_client.query_batch_points(
                collection_name=COLLECTION_NAME, requests=dense_vector_query_batch_text
            )
    except ResourceExhaustedResponse as ex:
        print(ex.message)
        assert ex.retry_after_s > 0, f"Unexpected retry_after_s value: {ex.retry_after_s}"
        flag = True
        time_to_sleep = int(ex.retry_after_s)

    if flag:
        # verify next response after sleep succeeds
        sleep(time_to_sleep)
        grpc_client.query_batch_points(
            collection_name=COLLECTION_NAME, requests=dense_vector_query_batch_text
        )
    else:
        raise AssertionError("No ResourceExhaustedResponse exception was raised for grpc_client.")
