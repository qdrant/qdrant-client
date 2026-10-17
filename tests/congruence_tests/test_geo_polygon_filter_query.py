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


def test_geo_filters_ignore_unusable_coordinates():
    """A stored location that is not a pair of numbers matches no geo filter, and does not
    keep the valid locations from matching (qdrant-client#1422). The server applies the
    geometry only when both coordinates read back as JSON numbers.
    """
    local_client = init_local()
    remote_client = init_remote()

    vectors_config = models.VectorParams(size=2, distance=models.Distance.DOT)
    initialize_fixture_collection(local_client, COLLECTION_NAME, vectors_config=vectors_config)
    initialize_fixture_collection(remote_client, COLLECTION_NAME, vectors_config=vectors_config)

    inside = {"lon": 0, "lat": 0}
    locations = [
        inside,  # integer coordinates
        {"lon": 0.0, "lat": 0.0},  # float coordinates
        # a coordinate the server stores verbatim but never reads as a number
        {"lon": 0, "lat": None},
        {"lon": None, "lat": 0},
        {"lon": 0, "lat": "0"},
        {"lon": "0", "lat": 0},
        {"lon": 0, "lat": True},
        {"lon": True, "lat": 0},
        {"lon": 0, "lat": False},
        {"lon": 0, "lat": []},
        {"lon": 0, "lat": [0]},
        {"lon": 0, "lat": {}},
        # a location that is not a geo point
        {"lat": 0},
        {"lon": 0},
        {},
        "0,0",
        0,
        [],
        [{"lon": 0, "lat": None}, inside],  # an array matches on its valid location
    ]
    points = [
        models.PointStruct(id=i, vector=[0.1, 0.1], payload={"location": location})
        for i, location in enumerate(locations, start=1)
    ]
    local_client.upload_points(COLLECTION_NAME, points, wait=True)
    remote_client.upload_points(COLLECTION_NAME, points, wait=True)

    conditions = [
        {"geo_radius": models.GeoRadius(center=models.GeoPoint(lon=0, lat=0), radius=1000)},
        {
            "geo_bounding_box": models.GeoBoundingBox(
                top_left=models.GeoPoint(lon=-1, lat=1),
                bottom_right=models.GeoPoint(lon=1, lat=-1),
            )
        },
        {
            "geo_polygon": models.GeoPolygon(
                exterior=models.GeoLineString(
                    points=[
                        models.GeoPoint(lon=-1, lat=-1),
                        models.GeoPoint(lon=1, lat=-1),
                        models.GeoPoint(lon=1, lat=1),
                        models.GeoPoint(lon=-1, lat=1),
                        models.GeoPoint(lon=-1, lat=-1),
                    ]
                )
            )
        },
    ]

    for condition in conditions:
        geo_filter = models.Filter(must=[models.FieldCondition(key="location", **condition)])

        def scroll(client):
            return client.scroll(
                collection_name=COLLECTION_NAME,
                scroll_filter=geo_filter,
                limit=len(locations),
                with_payload=True,
            )

        compare_client_results(local_client, remote_client, scroll)
        # the two valid locations and the array, so the comparison above is not vacuous
        assert [point.id for point in scroll(local_client)[0]] == [1, 2, len(locations)]
