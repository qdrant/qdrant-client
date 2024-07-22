from copy import deepcopy

import pytest

import qdrant_client.http.exceptions as qdrant_exceptions
from qdrant_client import QdrantClient, models
from tests.congruence_tests.test_common import (
    compare_collections,
    generate_fixtures,
    generate_sparse_fixtures,
    sparse_vectors_config,
    generate_multivector_fixtures,
    multi_vector_config,
    initialize_fixture_collection,
)

VECTOR_NUMBER = 1000


@pytest.fixture
def local_client() -> QdrantClient:
    client = QdrantClient(":memory:")
    delete_collections(client)
    yield client
    delete_collections(client)
    client.close()


second_local_client = deepcopy(local_client)


@pytest.fixture
def remote_client() -> QdrantClient:
    client = QdrantClient()
    delete_collections(client)
    yield client
    delete_collections(client)
    client.close()


@pytest.fixture
def second_remote_client() -> QdrantClient:
    client = QdrantClient(port=6334)
    delete_collections(client)
    yield client
    delete_collections(client)
    client.close()


def delete_collections(client: QdrantClient) -> None:
    collection_names = [collection.name for collection in client.get_collections().collections]
    for collection_name in collection_names:
        client.delete_collection(collection_name)


@pytest.mark.parametrize(
    "source_client,dest_client",
    [
        ("local_client", "remote_client"),
        ("remote_client", "local_client"),
        ("local_client", "second_local_client"),
        ("remote_client", "second_remote_client"),
    ],
)
def test_single_vector_collection(source_client, dest_client, request) -> None:
    """
    Args:
        source_client: fixture
        dest_client: fixture
        request: pytest internal object to get launch fixtures from parametrize
    """
    source_client: QdrantClient = request.getfixturevalue(source_client)
    dest_client: QdrantClient = request.getfixturevalue(dest_client)
    vectors_config = models.VectorParams(size=10, distance=models.Distance.COSINE)
    collection_name = "single_vector_collection"
    initialize_fixture_collection(
        source_client, collection_name=collection_name, vectors_config=vectors_config
    )
    dense_points = generate_fixtures(VECTOR_NUMBER, vectors_sizes=vectors_config.size)
    # TODO(sparse)
    # sparse_points = generate_sparse_fixtures(VECTOR_NUMBER)
    points = dense_points  # + sparse_points
    source_client.upload_points(collection_name, points, wait=True)
    source_client.migrate(dest_client)
    # dest_client.upload_points(collection_name, points, wait=True)
    compare_collections(
        source_client,
        dest_client,
        num_vectors=VECTOR_NUMBER,
        collection_name=collection_name,
    )


@pytest.mark.parametrize(
    "source_client,dest_client",
    [
        ("local_client", "remote_client"),
        ("remote_client", "local_client"),
        ("local_client", "second_local_client"),
        ("remote_client", "second_remote_client"),
    ],
)
def test_multiple_vectors_collection(source_client, dest_client, request) -> None:
    """
    Args:
        source_client: fixture
        dest_client: fixture
        request: pytest internal object to get launch fixtures from parametrize
    """
    source_client: QdrantClient = request.getfixturevalue(source_client)
    dest_client: QdrantClient = request.getfixturevalue(dest_client)
    vectors_config = multi_vector_config
    collection_name = "multiple_vectors_collection"
    initialize_fixture_collection(
        source_client, collection_name=collection_name, vectors_config=vectors_config
    )
    multi_vectors_points = generate_multivector_fixtures()
    points = multi_vectors_points
    source_client.upload_points(collection_name, points, wait=True)
    source_client.migrate(dest_client)
    compare_collections(
        source_client,
        dest_client,
        num_vectors=VECTOR_NUMBER,
        collection_name=collection_name,
    )


@pytest.mark.parametrize(
    "source_client,dest_client",
    [
        ("local_client", "remote_client"),
        ("remote_client", "local_client"),
        ("local_client", "second_local_client"),
        ("remote_client", "second_remote_client"),
    ],
)
def test_sparse_vector_collection(source_client, dest_client, request) -> None:
    """
    Args:
        source_client: fixture
        dest_client: fixture
        request: pytest internal object to get launch fixtures from parametrize
    """
    source_client: QdrantClient = request.getfixturevalue(source_client)
    dest_client: QdrantClient = request.getfixturevalue(dest_client)
    collection_name = "sparse_vector_collection"
    vectors_config = sparse_vectors_config
    initialize_fixture_collection(
        source_client, collection_name=collection_name, sparse_vectors_config=vectors_config
    )
    sparse_points = generate_sparse_fixtures()
    points = sparse_points
    source_client.upload_points(collection_name, points, wait=True)
    source_client.migrate(dest_client)
    compare_collections(
        source_client,
        dest_client,
        num_vectors=VECTOR_NUMBER,
        collection_name=collection_name,
    )


@pytest.mark.parametrize(
    "source_client,dest_client",
    [
        ("local_client", "remote_client"),
        ("remote_client", "local_client"),
        ("local_client", "second_local_client"),
        ("remote_client", "second_remote_client"),
    ],
)
def test_migrate_all_collections(source_client, dest_client, request) -> None:
    """
    Args:
        source_client: fixture
        dest_client: fixture
        request: pytest internal object to get launch fixtures from parametrize
    """
    vector_number = 100
    collection_names = ["collection_1", "collection_2", "collection_3"]
    source_client: QdrantClient = request.getfixturevalue(source_client)
    dest_client: QdrantClient = request.getfixturevalue(dest_client)
    for collection_name in collection_names:
        initialize_fixture_collection(source_client, collection_name=collection_name)
        points = generate_fixtures(vector_number)
        source_client.upload_points(
            collection_name,
            points,
            wait=True,
        )

    source_client.migrate(dest_client)

    for collection_name in collection_names:
        compare_collections(
            source_client,
            dest_client,
            num_vectors=vector_number,
            collection_name=collection_name,
        )


@pytest.mark.parametrize(
    "source_client,dest_client",
    [
        ("local_client", "remote_client"),
        ("remote_client", "local_client"),
        ("local_client", "second_local_client"),
        ("remote_client", "second_remote_client"),
    ],
)
def test_migrate_particular_collections(source_client, dest_client, request) -> None:
    """
    Args:
        source_client: fixture
        dest_client: fixture
        request: pytest internal object to get launch fixtures from parametrize
    """
    vector_number = 100
    collection_names = ["collection_1", "collection_2", "collection_3"]
    source_client: QdrantClient = request.getfixturevalue(source_client)
    dest_client: QdrantClient = request.getfixturevalue(dest_client)
    for collection_name in collection_names:
        initialize_fixture_collection(source_client, collection_name=collection_name)
        points = generate_fixtures(vector_number)
        source_client.upload_points(
            collection_name,
            points,
            wait=True,
        )

    source_client.migrate(dest_client, collection_names=collection_names[:2])

    for collection_name in collection_names[:2]:
        compare_collections(
            source_client,
            dest_client,
            num_vectors=vector_number,
            collection_name=collection_name,
        )

    for collection_name in collection_names[2:]:
        with pytest.raises((qdrant_exceptions.UnexpectedResponse, ValueError)):  # type: ignore
            dest_client.get_collection(collection_name)


@pytest.mark.parametrize(
    "source_client,dest_client",
    [
        ("local_client", "remote_client"),
        ("remote_client", "local_client"),
        ("local_client", "second_local_client"),
        ("remote_client", "second_remote_client"),
    ],
)
def test_action_on_collision(source_client, dest_client, request) -> None:
    """
    Args:
        source_client: fixture
        dest_client: fixture
        request: pytest internal object to get launch fixtures from parametrize
    """
    collection_name = "test_collection"
    source_client: QdrantClient = request.getfixturevalue(source_client)
    dest_client: QdrantClient = request.getfixturevalue(dest_client)
    initialize_fixture_collection(source_client, collection_name=collection_name)
    initialize_fixture_collection(dest_client, collection_name=collection_name)

    with pytest.raises(ValueError):
        source_client.migrate(dest_client, recreate_on_collision=False)

    points = generate_fixtures(VECTOR_NUMBER)
    source_client.upload_points(
        collection_name,
        points,
        wait=True,
    )
    source_client.migrate(dest_client, recreate_on_collision=True)
    compare_collections(
        source_client,
        dest_client,
        num_vectors=VECTOR_NUMBER,
        collection_name=collection_name,
    )


def test_vector_params(
    local_client: QdrantClient,
    second_local_client: QdrantClient,
    remote_client: QdrantClient,
):
    collection_name = "test_collection"

    image_hnsw_config = models.HnswConfigDiff(
        m=9,
        ef_construct=99,
        full_scan_threshold=42,
        max_indexing_threads=4,
        on_disk=True,
        payload_m=5,
    )
    image_quantization_config = models.ScalarQuantization(
        scalar=models.ScalarQuantizationConfig(
            type=models.ScalarType.INT8, quantile=0.69, always_ram=False
        )
    )

    image_on_disk = True

    vectors_config = {
        "text": models.VectorParams(size=10, distance=models.Distance.COSINE),
        "image": models.VectorParams(
            size=20,
            distance=models.Distance.DOT,
            hnsw_config=image_hnsw_config,
            quantization_config=image_quantization_config,
            on_disk=image_on_disk,
        ),
    }

    local_client.recreate_collection(
        collection_name=collection_name, vectors_config=vectors_config
    )

    local_client.migrate(second_local_client)

    assert local_client.get_collection(collection_name) == second_local_client.get_collection(
        collection_name
    )

    local_client.migrate(remote_client)

    local_collection_vector_params = local_client.get_collection(
        collection_name
    ).config.params.vectors
    remote_collection_vector_params = remote_client.get_collection(
        collection_name
    ).config.params.vectors

    assert local_collection_vector_params == remote_collection_vector_params

    local_client.delete_collection(collection_name)

    remote_client.migrate(local_client)
    local_collection_vector_params = local_client.get_collection(
        collection_name
    ).config.params.vectors

    assert local_collection_vector_params == remote_collection_vector_params


def test_migrate_missing_collections(
    local_client: QdrantClient, second_local_client: QdrantClient
):
    collection_name = "test_collection"
    with pytest.raises(AssertionError):
        local_client.migrate(second_local_client, collection_names=[collection_name])


def test_recreate_collection(remote_client: QdrantClient):
    collection_name = "test_collection"
    initialize_fixture_collection(remote_client, collection_name=collection_name)
    collection_before_migrate = remote_client.get_collection(collection_name)
    remote_client.migrate(remote_client, recreate_on_collision=True)
    assert collection_before_migrate == remote_client.get_collection(collection_name)

    remote_client.delete_collection(collection_name)

    image_hnsw_config = models.HnswConfigDiff(
        m=9,
        ef_construct=99,
        full_scan_threshold=4200,
        max_indexing_threads=2,
        on_disk=True,
        payload_m=5,
    )
    image_quantization_config = models.ScalarQuantization(
        scalar=models.ScalarQuantizationConfig(
            type=models.ScalarType.INT8, quantile=0.89, always_ram=False
        )
    )

    image_on_disk = True

    vectors_config = {
        "text": models.VectorParams(size=10, distance=models.Distance.COSINE),
        "image": models.VectorParams(
            size=20,
            distance=models.Distance.DOT,
            hnsw_config=image_hnsw_config,
            quantization_config=image_quantization_config,
            on_disk=image_on_disk,
        ),
    }

    general_hnsw_config = models.HnswConfigDiff(
        m=13,
        ef_construct=101,
        full_scan_threshold=10_001,
        max_indexing_threads=1,
        on_disk=True,
        payload_m=16,
    )
    optimizers_config = models.OptimizersConfigDiff(
        deleted_threshold=0.21,
        vacuum_min_vector_number=1001,
        default_segment_number=2,
        max_segment_size=42_000,
        memmap_threshold=42_000,
        indexing_threshold=42_000,
        flush_interval_sec=6,
        max_optimization_threads=2,
    )

    wal_config = models.WalConfigDiff(wal_capacity_mb=42, wal_segments_ahead=3)

    general_quantization_config = models.ProductQuantization(
        product=models.ProductQuantizationConfig(
            compression=models.CompressionRatio.X4, always_ram=False
        )
    )

    remote_client.recreate_collection(
        collection_name,
        vectors_config=vectors_config,
        shard_number=3,
        replication_factor=3,
        write_consistency_factor=2,
        on_disk_payload=True,
        hnsw_config=general_hnsw_config,
        optimizers_config=optimizers_config,
        wal_config=wal_config,
        quantization_config=general_quantization_config,
    )

    remote_client.create_payload_index(
        collection_name,
        field_name="title",
        field_schema=models.PayloadSchemaType.KEYWORD,
    )

    remote_client.create_payload_index(
        collection_name,
        field_name="description",
        field_schema=models.TextIndexParams(
            type=models.TextIndexType.TEXT,
            tokenizer=models.TokenizerType.PREFIX,
            min_token_len=3,
            max_token_len=5,
            lowercase=False,
        ),
    )

    collection_before_migrate = remote_client.get_collection(collection_name)
    remote_client.migrate(remote_client, recreate_on_collision=True)
    assert collection_before_migrate == remote_client.get_collection(collection_name)
