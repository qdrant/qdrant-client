import pytest

from qdrant_client import AsyncQdrantClient, QdrantClient, models
from qdrant_client.uploader.grpc_uploader import upload_batch_grpc
from qdrant_client.uploader.rest_uploader import upload_batch
from qdrant_client.uploader.uploader import BaseUploader


VECTORS = [[1.0], [2.0], [3.0]]


@pytest.mark.parametrize(
    ("ids", "payload", "message"),
    [
        ([1, 2], None, "ids iterable is shorter than vectors iterable"),
        (None, [{}, {}], "payload iterable is shorter than vectors iterable"),
        (iter([]), None, "ids iterable is shorter than vectors iterable"),
        (None, iter([]), "payload iterable is shorter than vectors iterable"),
    ],
)
def test_iterate_batches_rejects_shorter_auxiliary_iterables(ids, payload, message):
    with pytest.raises(ValueError, match=message):
        list(BaseUploader.iterate_batches(VECTORS, payload, ids, batch_size=2))


def test_iterate_batches_allows_infinite_auxiliary_iterables():
    def ids():
        point_id = 1
        while True:
            yield point_id
            point_id += 1

    batches = list(BaseUploader.iterate_batches(VECTORS, None, ids(), batch_size=2))

    assert batches == [([1, 2], [[1.0], [2.0]], None), ([3], [[3.0]], None)]


def test_iterate_batches_preserves_matching_upload_items():
    payload = [{"position": 1}, {"position": 2}, {"position": 3}]

    batches = list(BaseUploader.iterate_batches(VECTORS, payload, [1, 2, 3], batch_size=2))

    assert batches == [
        ([1, 2], [[1.0], [2.0]], [{"position": 1}, {"position": 2}]),
        ([3], [[3.0]], [{"position": 3}]),
    ]


@pytest.mark.parametrize("uploader", [upload_batch, upload_batch_grpc])
def test_batch_uploaders_reject_shorter_ids_before_request(uploader):
    class NoUploadClient:
        points_api = None

        def __init__(self):
            self.points_api = self

        def upsert_points(self, **kwargs):
            raise AssertionError("REST request should not be sent")

        def Upsert(self, request, timeout=None):
            raise AssertionError("gRPC request should not be sent")

    with pytest.raises(ValueError, match="ids iterable is shorter than vectors iterable"):
        uploader(
            NoUploadClient(),
            "collection",
            ([1], VECTORS, None),
            1,
            None,
            None,
        )


def create_local_client():
    client = QdrantClient(":memory:")
    client.create_collection(
        "collection",
        vectors_config=models.VectorParams(size=1, distance=models.Distance.DOT),
    )
    return client


def test_local_upload_collection_preserves_matching_upload_items():
    client = create_local_client()
    payload = [{"position": 1}, {"position": 2}, {"position": 3}]

    client.upload_collection("collection", vectors=VECTORS, ids=[1, 2, 3], payload=payload)

    points = client.retrieve("collection", ids=[1, 2, 3], with_vectors=True)
    assert [(point.id, point.vector, point.payload) for point in points] == [
        (1, [1.0], {"position": 1}),
        (2, [2.0], {"position": 2}),
        (3, [3.0], {"position": 3}),
    ]


@pytest.mark.parametrize(
    ("ids", "payload", "message"),
    [
        ([1], None, "ids iterable is shorter than vectors iterable"),
        ([1, 2, 3], [{}], "payload iterable is shorter than vectors iterable"),
        ([], None, "ids iterable is shorter than vectors iterable"),
        ([1, 2, 3], [], "payload iterable is shorter than vectors iterable"),
    ],
)
def test_local_upload_collection_rejects_shorter_auxiliary_iterables(ids, payload, message):
    client = create_local_client()

    with pytest.raises(ValueError, match=message):
        client.upload_collection("collection", vectors=VECTORS, ids=ids, payload=payload)

    assert client.count("collection", exact=True).count == 0


@pytest.mark.asyncio
async def test_async_local_upload_collection_rejects_shorter_ids():
    client = AsyncQdrantClient(":memory:")
    await client.create_collection(
        "collection",
        vectors_config=models.VectorParams(size=1, distance=models.Distance.COSINE),
    )

    with pytest.raises(ValueError, match="ids iterable is shorter than vectors iterable"):
        await client.upload_collection("collection", vectors=VECTORS, ids=[])

    assert (await client.count("collection", exact=True)).count == 0
