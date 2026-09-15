import pytest

from qdrant_client.http.models import models
from tests.congruence_tests.test_common import (
    COLLECTION_NAME,
    compare_client_results,
    compare_collections,
    generate_fixtures,
    generate_sparse_fixtures,
    init_client,
    init_local,
    init_remote,
    sparse_vectors_config,
)


def test_delete_points(local_client, remote_client):
    points = generate_fixtures(100)
    vector = points[0].vector["image"]
    local_client.upload_points(COLLECTION_NAME, points)
    remote_client.upload_points(COLLECTION_NAME, points, wait=True)

    compare_client_results(
        local_client,
        remote_client,
        lambda c: c.query_points(COLLECTION_NAME, query=vector, using="image").points,
    )

    found_ids = [
        scored_point.id
        for scored_point in local_client.query_points(
            COLLECTION_NAME, query=vector, using="image"
        ).points
    ]

    local_client.delete(COLLECTION_NAME, found_ids)
    remote_client.delete(COLLECTION_NAME, found_ids)

    compare_collections(local_client, remote_client, 100, attrs=("points_count",))

    compare_client_results(
        local_client,
        remote_client,
        lambda c: c.query_points(COLLECTION_NAME, query=vector, using="image").points,
    )

    # delete non-existent points
    local_client.delete(COLLECTION_NAME, found_ids)
    remote_client.delete(COLLECTION_NAME, found_ids)

    compare_collections(local_client, remote_client, 100, attrs=("points_count",))

    compare_client_results(
        local_client,
        remote_client,
        lambda c: c.query_points(COLLECTION_NAME, query=vector, using="image").points,
    )


def test_delete_sparse_points():
    points = generate_sparse_fixtures(100)
    vector = points[0].vector["sparse-image"]

    local_client = init_local()
    init_client(local_client, [], sparse_vectors_config=sparse_vectors_config)

    remote_client = init_remote()
    init_client(remote_client, [], sparse_vectors_config=sparse_vectors_config)

    local_client.upload_points(COLLECTION_NAME, points)
    remote_client.upload_points(COLLECTION_NAME, points, wait=True)

    compare_client_results(
        local_client,
        remote_client,
        lambda c: c.query_points(
            COLLECTION_NAME,
            query=vector,
            using="sparse-image",
        ).points,
    )

    found_ids = [
        scored_point.id
        for scored_point in local_client.query_points(
            COLLECTION_NAME, query=vector, using="sparse-image"
        ).points
    ]

    local_client.delete(COLLECTION_NAME, found_ids)
    remote_client.delete(COLLECTION_NAME, found_ids)

    compare_collections(local_client, remote_client, 100, attrs=("points_count",))

    compare_client_results(
        local_client,
        remote_client,
        lambda c: c.query_points(COLLECTION_NAME, query=vector, using="sparse-image").points,
    )


@pytest.mark.parametrize(
    "operation",
    ["set_payload", "overwrite_payload", "delete_payload", "clear_payload", "update_vectors"],
)
@pytest.mark.parametrize("absent", ["unknown", "deleted"])
def test_write_to_an_absent_point(operation: str, absent: str):
    """A point the collection no longer holds is as absent as one it never held.

    Deleted points keep their slot locally so that internal ids stay stable, which is why
    they have to be excluded explicitly: writing to one would resurrect it in storage. And a
    request that names an absent point is not abandoned at that point - the server writes
    every other point in it and reports the missing id afterwards, so the ids listed after
    the absent one must be written too.
    """
    points = generate_fixtures(50)

    local_client = init_local()
    remote_client = init_remote()
    for client in (local_client, remote_client):
        init_client(client, points)
        client.delete(COLLECTION_NAME, [points[-1].id], wait=True)

    missing_id = points[-1].id if absent == "deleted" else max(p.id for p in points) + 1_000
    targets = [points[0].id, missing_id, points[1].id]

    for client in (local_client, remote_client):
        with pytest.raises(Exception):
            if operation == "set_payload":
                client.set_payload(COLLECTION_NAME, payload={"tag": 1}, points=targets, wait=True)
            elif operation == "overwrite_payload":
                client.overwrite_payload(
                    COLLECTION_NAME, payload={"tag": 2}, points=targets, wait=True
                )
            elif operation == "delete_payload":
                client.delete_payload(
                    COLLECTION_NAME, keys=["rand_digit"], points=targets, wait=True
                )
            elif operation == "clear_payload":
                client.clear_payload(COLLECTION_NAME, points_selector=targets, wait=True)
            else:
                client.update_vectors(
                    COLLECTION_NAME,
                    [
                        models.PointVectors(
                            id=point_id, vector={"image": points[2].vector["image"]}
                        )
                        for point_id in targets
                    ],
                    wait=True,
                )

    compare_collections(local_client, remote_client, 100, attrs=("points_count",))
    compare_client_results(
        local_client,
        remote_client,
        lambda c: c.scroll(
            COLLECTION_NAME, limit=len(points), with_payload=True, with_vectors=True
        )[0],
    )


@pytest.mark.parametrize("absent", ["unknown", "deleted"])
def test_recommend_from_an_absent_point(absent: str):
    """Recommending from a deleted point must fail, not fall back on its leftover vector."""
    points = generate_fixtures(50)

    local_client = init_local()
    remote_client = init_remote()
    for client in (local_client, remote_client):
        init_client(client, points)
        client.delete(COLLECTION_NAME, [points[-1].id], wait=True)

    missing_id = points[-1].id if absent == "deleted" else max(p.id for p in points) + 1_000

    for client in (local_client, remote_client):
        with pytest.raises(Exception):
            client.query_points(
                COLLECTION_NAME,
                query=models.RecommendQuery(
                    recommend=models.RecommendInput(positive=[missing_id])
                ),
                using="image",
                limit=5,
            )
