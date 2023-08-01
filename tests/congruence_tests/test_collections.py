from tests.congruence_tests.test_common import (
    generate_fixtures,
    init_client,
    init_local,
    init_remote,
)
from qdrant_client.http import models

COLLECTION_NAME = "test_collection"


def test_get_collection():
    fixture_records = generate_fixtures()

    remote_client = init_remote()

    remote_collections = remote_client.get_collections()

    for collection in remote_collections.collections:
        remote_client.delete_collection(collection.name)

    local_client = init_local()
    init_client(local_client, fixture_records)

    remote_client = init_remote()
    init_client(remote_client, fixture_records)

    local_collections = local_client.get_collections()

    remote_collections = remote_client.get_collections()

    assert len(local_collections.collections) == len(remote_collections.collections)

    local_collection = local_collections.collections[0].name
    remote_collection = remote_collections.collections[0].name

    assert local_collection == remote_collection

    local_collection_info = local_client.get_collection(local_collection)

    remote_collection_info = remote_client.get_collection(remote_collection)

    assert local_collection_info.points_count == remote_collection_info.points_count

    assert (
        local_collection_info.config.params.vectors == remote_collection_info.config.params.vectors
    )


def test_update_collection():
    fixture_records = generate_fixtures()

    remote_client = init_remote()
    remote_collections = remote_client.get_collections()
    for collection in remote_collections.collections:
        remote_client.delete_collection(collection.name)

    local_client = init_local()
    init_client(local_client, fixture_records)

    remote_client = init_remote()
    init_client(remote_client, fixture_records)

    local_client.update_collection(
        collection_name=COLLECTION_NAME,
        vectors_config = {
            "text": models.VectorParamsDiff(
                hnsw_config=models.HnswConfigDiff(
                    m=32,
                    ef_construct=123,
                ),
                quantization_config=models.ProductQuantization(
                    product=models.ProductQuantizationConfig(
                        compression=models.CompressionRatio.X32,
                        always_ram=True,
                    ),
                ),
                on_disk=True,
            ),
        },
        hnsw_config=models.HnswConfigDiff(
            ef_construct=123,
        ),
        quantization_config=models.ScalarQuantization(
            scalar=models.ScalarQuantizationConfig(
                type=models.ScalarType.INT8,
                quantile=0.8,
                always_ram=False,
            ),
        ),
    )
    remote_client.update_collection(
        collection_name=COLLECTION_NAME,
        vectors_config = {
            "text": models.VectorParamsDiff(
                hnsw_config=models.HnswConfigDiff(
                    m=32,
                    ef_construct=123,
                ),
                quantization_config=models.ProductQuantization(
                    product=models.ProductQuantizationConfig(
                        compression=models.CompressionRatio.X32,
                        always_ram=True,
                    ),
                ),
                on_disk=True,
            ),
        },
        hnsw_config=models.HnswConfigDiff(
            ef_construct=123,
        ),
        quantization_config=models.ScalarQuantization(
            scalar=models.ScalarQuantizationConfig(
                type=models.ScalarType.INT8,
                quantile=0.8,
                always_ram=False,
            ),
        ),
    )

    local_collection_info = local_client.get_collection(COLLECTION_NAME)
    remote_collection_info = remote_client.get_collection(COLLECTION_NAME)

    assert local_collection_info.config.params.vectors["text"].hnsw_config is None
    assert local_collection_info.config.params.vectors["text"].quantization_config is None
    assert local_collection_info.config.hnsw_config.ef_construct == 123
    assert local_collection_info.config.quantization_config is None

    assert remote_collection_info.config.params.vectors["text"].hnsw_config.m == 32
    assert remote_collection_info.config.params.vectors["text"].hnsw_config.ef_construct == 123
    assert remote_collection_info.config.params.vectors["text"].quantization_config.product.compression == models.CompressionRatio.X32
    assert remote_collection_info.config.params.vectors["text"].quantization_config.product.always_ram
    assert remote_collection_info.config.params.vectors["text"].on_disk
    assert remote_collection_info.config.hnsw_config.ef_construct == 123
    assert remote_collection_info.config.quantization_config.scalar.type == models.ScalarType.INT8
    assert remote_collection_info.config.quantization_config.scalar.quantile == 0.8
    assert not remote_collection_info.config.quantization_config.scalar.always_ram
