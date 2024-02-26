import pytest
from grpc import Compression

from qdrant_client import QdrantClient
from qdrant_client.async_qdrant_client import AsyncQdrantClient
from tests.congruence_tests.test_common import generate_fixtures, init_client


def test_compression_gzip():
    # creates a grpc client with specifying gzip compression algorithm
    client = QdrantClient(prefer_grpc=True, grpc_compression=Compression.Gzip, timeout=30)

    # creates points
    fixtures = generate_fixtures()

    # recreates collection and uploads the points
    init_client(client, fixtures)

    # points successfully uploaded in collection using Gzip compression algorithm


def test_compression_nocompression():
    # creates a grpc client with specifying no-compression algorithm
    client = QdrantClient(prefer_grpc=True, grpc_compression=Compression.NoCompression, timeout=30)

    # creates points
    fixtures = generate_fixtures()

    # recreates collection and uploads the points
    init_client(client, fixtures)

    # points successfully uploaded in collection using no compression algorithm


def test_compression_none():
    # creates a grpc client without specifying compression algorithm
    client = QdrantClient(prefer_grpc=True, timeout=30)  # grpc_compression=None

    # creates points
    fixtures = generate_fixtures()

    # recreates collection and uploads the points
    init_client(client, fixtures)

    # points successfully uploaded in collection using no compression algorithm


def test_compression_deflate():
    """
    Deflate algorithm is not supported by RUST GRPC Server
    Trying Deflate compression should raise ValueError
    """

    with pytest.raises(ValueError):
        # creates a grpc client with invalid enum attribute
        client = QdrantClient(prefer_grpc=True, grpc_compression=Compression.Deflate, timeout=30)

    # Anticipated Errors were successfully raised


def test_compression_invalid():
    # creates a grpc client with invalid type
    with pytest.raises(TypeError):
        client = QdrantClient(prefer_grpc=True, grpc_compression="gzip", timeout=30)

    # TypeError was successfully raised
