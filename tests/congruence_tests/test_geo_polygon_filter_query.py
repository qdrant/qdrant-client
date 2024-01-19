import json
import random

from qdrant_client.http.models import models
from tests.congruence_tests.test_common import (
    COLLECTION_NAME,
    generate_fixtures,
    init_client,
    init_local,
    init_remote,
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
