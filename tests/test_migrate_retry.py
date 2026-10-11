from itertools import islice
from unittest.mock import MagicMock

import pytest

from qdrant_client.client_base import QdrantBase
from qdrant_client.http import models
from qdrant_client.migrate.migrate import upload_with_retry


@pytest.mark.parametrize(
    "points_factory",
    [
        pytest.param(list, id="list"),
        pytest.param(iter, id="iterator"),
        pytest.param(lambda values: (value for value in values), id="generator"),
    ],
)
@pytest.mark.parametrize("consumed_count", [1, 3])
def test_upload_with_retry_replays_the_complete_batch(points_factory, consumed_count):
    points = [models.PointStruct(id=idx, vector=[float(idx)]) for idx in range(1, 4)]
    client = MagicMock(spec=QdrantBase)
    attempted_ids = []

    def upload_points(*, collection_name, points, wait):
        if not attempted_ids:
            attempted_ids.append([point.id for point in islice(points, consumed_count)])
            raise ConnectionError("Upload interrupted after consuming points")
        attempted_ids.append([point.id for point in points])

    client.upload_points.side_effect = upload_points

    upload_with_retry(client, "collection", points_factory(points), max_attempts=2, pause=0)

    assert attempted_ids == [list(range(1, consumed_count + 1)), [1, 2, 3]]
    assert client.upload_points.call_count == 2
