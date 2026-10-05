from collections.abc import Iterator

import pytest

from qdrant_client import QdrantClient, models


@pytest.fixture
def client() -> Iterator[QdrantClient]:
    client = QdrantClient(":memory:")
    client.create_collection(
        "test",
        vectors_config={},
        sparse_vectors_config={
            "sparse": models.SparseVectorParams(
                modifier=models.Modifier.IDF,
                index=models.SparseIndexParams(
                    full_scan_threshold=100, on_disk=True, datatype=models.Datatype.FLOAT16
                ),
            )
        },
    )
    client.upsert(
        "test",
        [
            models.PointStruct(
                id=i, vector={"sparse": models.SparseVector(indices=[i], values=[1.0])}
            )
            for i in [1, 2]
        ],
    )
    try:
        yield client
    finally:
        client.close()


def query_score(client: QdrantClient) -> float:
    return (
        client.query_points(
            "test", query=models.SparseVector(indices=[1], values=[1.0]), using="sparse"
        )
        .points[0]
        .score
    )


@pytest.mark.parametrize(
    "index_update,expected_threshold,expected_on_disk",
    [
        (None, 100, True),
        (models.SparseIndexParams(), 100, True),
        (models.SparseIndexParams(on_disk=False), 100, False),
        (models.SparseIndexParams(full_scan_threshold=0), 0, True),
    ],
)
def test_partial_update_preserves_sparse_settings_and_idf_scores(
    client: QdrantClient,
    index_update: models.SparseIndexParams | None,
    expected_threshold: int,
    expected_on_disk: bool,
) -> None:
    before = query_score(client)
    client.update_collection(
        "test", sparse_vectors_config={"sparse": models.SparseVectorParams(index=index_update)}
    )

    params = client.get_collection("test").config.params.sparse_vectors["sparse"]
    assert params.modifier == models.Modifier.IDF
    assert params.index == models.SparseIndexParams(
        full_scan_threshold=expected_threshold,
        on_disk=expected_on_disk,
        datatype=models.Datatype.FLOAT16,
    )
    assert query_score(client) == pytest.approx(before)


def test_explicit_modifier_none_preserves_index_config(client: QdrantClient) -> None:
    index = client.get_collection("test").config.params.sparse_vectors["sparse"].index
    assert query_score(client) < 1.0

    client.update_collection(
        "test",
        sparse_vectors_config={"sparse": models.SparseVectorParams(modifier=models.Modifier.NONE)},
    )

    params = client.get_collection("test").config.params.sparse_vectors["sparse"]
    assert params.modifier == models.Modifier.NONE
    assert params.index == index
    assert query_score(client) == 1.0


@pytest.mark.parametrize("existing_index", [None, models.SparseIndexParams(on_disk=True)])
def test_index_update_does_not_share_caller_or_returned_config(
    client: QdrantClient, existing_index: models.SparseIndexParams | None
) -> None:
    client.create_collection(
        "other",
        vectors_config={},
        sparse_vectors_config={
            "sparse": models.SparseVectorParams(modifier=models.Modifier.IDF, index=existing_index)
        },
    )
    update = models.SparseVectorParams(index=models.SparseIndexParams(on_disk=False))
    client.update_collection("other", sparse_vectors_config={"sparse": update})

    update.index.on_disk = True
    returned = client.get_collection("other").config.params.sparse_vectors["sparse"]
    returned.index.on_disk = True

    stored = client.get_collection("other").config.params.sparse_vectors["sparse"]
    assert stored.modifier == models.Modifier.IDF
    assert stored.index.on_disk is False
