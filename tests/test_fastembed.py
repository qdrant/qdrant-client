import pytest

from qdrant_client import QdrantClient


def test_get_embedding_size():
    local_client = QdrantClient(":memory:")

    if not local_client._FASTEMBED_INSTALLED:
        pytest.skip("FastEmbed is not installed, skipping test")

    assert local_client.get_embedding_size(model_name="BAAI/bge-base-en-v1.5") == 768

    assert local_client.get_embedding_size(model_name="Qdrant/resnet50-onnx") == 2048

    assert local_client.get_embedding_size(model_name="colbert-ir/colbertv2.0") == 128

    with pytest.raises(
        ValueError, match="Sparse embeddings do not return fixed embedding size and distance type"
    ):
        local_client.get_embedding_size(model_name="Qdrant/bm25")
