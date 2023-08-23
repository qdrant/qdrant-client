from typing import Any, Dict, List, Union

import pytest

from qdrant_client import QdrantClient


def test_add_without_query(
    local_client: QdrantClient = QdrantClient(":memory:"),
    collection_name: str = "demo_collection",
    docs: List[str] = [
                "Qdrant has Langchain integrations",
                "Qdrant also has Llama Index integrations",
            ],
):
    if not local_client._is_fastembed_installed:
        pytest.skip("FastEmbed is not installed, skipping test")

    local_client.add(
        collection_name=collection_name,
        documents=docs
    )
    assert local_client.count(collection_name).count == 2


def test_no_install(
    local_client: QdrantClient = None,
    collection_name: str = "demo_collection",
    docs: Dict[str, List[Union[str, int, Any]]] = None,
):
    if local_client is None:
        local_client = QdrantClient(":memory:")

    if docs is None:
        docs = {
            "documents": [
                "Qdrant has Langchain integrations",
                "Qdrant also has Llama Index integrations",
            ],
            "metadatas": [{"source": "Langchain-docs"}, {"source": "LlamaIndex-docs"}],
            "ids": [42, 2],
        }

    # When FastEmbed is not installed, the add method should raise an ImportError
    if local_client._is_fastembed_installed:
        pytest.skip("FastEmbed is installed, skipping test")
    else:
        with pytest.raises(ImportError):
            local_client.add(collection_name, docs)


def test_query(
    local_client: QdrantClient = None,
    collection_name: str = "demo_collection",
    docs: Dict[str, List[Union[str, int, Any]]] = None,
):
    if local_client is None:
        local_client = QdrantClient(":memory:")

    if docs is None:
        docs = {
            "documents": [
                "Qdrant has Langchain integrations",
                "Qdrant also has Llama Index integrations",
            ],
            "metadatas": [{"source": "Langchain-docs"}, {"source": "LlamaIndex-docs"}],
            "ids": [42, 2],
        }

    if not local_client._is_fastembed_installed:
        pytest.skip("FastEmbed is not installed, skipping test")

    local_client.add(
        collection_name=collection_name,
        documents=docs["documents"],
        metadata=docs["metadatas"],
        ids=docs["ids"],
    )
    assert local_client.count(collection_name).count == 2
    # Query the added documents

    search_result = local_client.query(
        collection_name=collection_name, query_text="This is a query document"
    )

    assert len(search_result) > 0


if __name__ == "__main__":
    test_add_without_query()
    test_query()
