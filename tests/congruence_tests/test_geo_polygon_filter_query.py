import json
import random

from qdrant_client.http.models import models
from tests.congruence_tests.test_common import (
    COLLECTION_NAME,
    compare_client_results,
    generate_fixtures,
    init_client,
    init_local,
    init_remote,
    initialize_fixture_collection,
)


def test_geo_polygon_filter_query():
    # fix random seed
    random.seed(42)

    fixture_points = generate_fixtures(num=100)

    local_client = init_local()
    init_client(local_client, fixture_points)

    remote_client = init_remote()
    init_client(remote_client, fixture_points)

    filter_ = models.Filter(
        **{
            "should": [
                {
                    "key": "city.geo",
                    "geo_polygon": {
                        "exterior": {
                            "points": [
                                {"lon": -55.0, "lat": -55.0},
                                {"lon": 65.0, "lat": -55.0},
                                {"lon": 65.0, "lat": 65.0},
                                {"lon": 55.0, "lat": -65.0},
                                {"lon": -55.0, "lat": -55.0},
                            ]
                        },
                    },
                },
                {
                    "key": "city.geo",
                    "geo_polygon": {
                        "exterior": {
                            "points": [
                                {"lon": 75.0, "lat": 75.0},
                                {"lon": 155.0, "lat": 75.0},
                                {"lon": 155.0, "lat": 85.0},
                                {"lon": 75.0, "lat": 85.0},
                                {"lon": 75.0, "lat": 75.0},
                            ]
                        },
                    },
                },
            ]
        }
    )

    local_result, _next_page = local_client.scroll(
        collection_name=COLLECTION_NAME,
        scroll_filter=filter_,
        limit=100,
        with_payload=True,
    )

    remote_result, _next_page = remote_client.scroll(
        collection_name=COLLECTION_NAME,
        scroll_filter=filter_,
        limit=100,
        with_payload=True,
    )

    print("local_result:", len(local_result))
    print("remote_result", len(remote_result))

    assert len(local_result) == len(remote_result)

    for local, remote in zip(local_result, remote_result):
        if local.id != remote.id:
            print(f"Local: {local.id}, Remote: {remote.id}")

            print(f"Local:", json.dumps(local.payload["nested"]["array"], indent=2))
            print(f"Remote:", json.dumps(remote.payload["nested"]["array"], indent=2))

            assert False


def test_geo_bounding_box_edge_point():
    """Points on bounding box edges use strict inequalities and should be excluded."""
    collection_name = COLLECTION_NAME

    local_client = init_local()
    remote_client = init_remote()

    vectors_config = models.VectorParams(size=2, distance=models.Distance.DOT)
    initialize_fixture_collection(local_client, collection_name, vectors_config=vectors_config)
    initialize_fixture_collection(remote_client, collection_name, vectors_config=vectors_config)

    # bbox: top_left=(lon=158.75, lat=90.0), bottom_right=(lon=180.0, lat=69.4)
    points = [
        # on top edge (lat == top_left.lat)
        models.PointStruct(
            id=1, vector=[0.1, 0.1], payload={"location": {"lat": 90.0, "lon": 170.0}}
        ),
        # on bottom edge (lat == bottom_right.lat)
        models.PointStruct(
            id=2, vector=[0.2, 0.2], payload={"location": {"lat": 69.4, "lon": 170.0}}
        ),
        # on left edge (lon == top_left.lon)
        models.PointStruct(
            id=3, vector=[0.3, 0.3], payload={"location": {"lat": 80.0, "lon": 158.75}}
        ),
        # on right edge (lon == bottom_right.lon)
        models.PointStruct(
            id=4, vector=[0.4, 0.4], payload={"location": {"lat": 80.0, "lon": 180.0}}
        ),
        # corner: top-right
        models.PointStruct(
            id=5, vector=[0.5, 0.5], payload={"location": {"lat": 90.0, "lon": 180.0}}
        ),
        # corner: bottom-left
        models.PointStruct(
            id=6, vector=[0.6, 0.6], payload={"location": {"lat": 69.4, "lon": 158.75}}
        ),
        # strictly inside
        models.PointStruct(
            id=7, vector=[0.7, 0.7], payload={"location": {"lat": 80.0, "lon": 170.0}}
        ),
        # strictly outside
        models.PointStruct(
            id=8, vector=[0.8, 0.8], payload={"location": {"lat": 50.0, "lon": 170.0}}
        ),
    ]

    local_client.upload_points(collection_name, points, wait=True)
    remote_client.upload_points(collection_name, points, wait=True)

    bbox = models.GeoBoundingBox(
        top_left=models.GeoPoint(lon=158.75, lat=90.0),
        bottom_right=models.GeoPoint(lon=180.0, lat=69.4),
    )
    geo_filter = models.Filter(must=[models.FieldCondition(key="location", geo_bounding_box=bbox)])

    compare_client_results(
        local_client,
        remote_client,
        lambda client: client.scroll(
            collection_name=collection_name,
            scroll_filter=geo_filter,
            limit=100,
            with_payload=True,
        ),
    )
