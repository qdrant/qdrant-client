
from typing import Any, Dict, List, Union

import pytest

from qdrant_client import QdrantClient


def test_add_without_query(local_client: QdrantClient = QdrantClient(":memory:"), 
             collection_name: str= "demo_collection", 
             docs: Dict[str, List[Union[str, int, Any]]] = {
        "documents": ["Qdrant has Langchain integrations", 
                      "Qdrant also has Llama Index integrations"],
        "metadatas": [{"source": "Langchain-docs"}, {"source": "LlamaIndex-docs"}],
        "ids": [42, 2]
    }):
             
    local_client.add(collection_name=collection_name, docs=docs)
    assert local_client.count(collection_name).count == 2
    
def test_no_install(local_client: QdrantClient, 
             collection_name: str, 
             docs: Dict[str, List[Union[str, int, Any]]]):
    with pytest.raises(ImportError):
        local_client.add(collection_name, docs)

def test_query(local_client: QdrantClient = QdrantClient(":memory:"), 
             collection_name: str= "demo_collection", 
             docs: Dict[str, List[Union[str, int, Any]]]={
        "documents": ["Qdrant has Langchain integrations", 
                      "Qdrant also has Llama Index integrations"],
        "metadatas": [{"source": "Langchain-docs"}, {"source": "LlamaIndex-docs"}],
        "ids": [42, 2]
    }):
    local_client.add(collection_name=collection_name, docs=docs)
    # Query the added documents
    search_result = local_client.query(collection_name=collection_name, 
                                 query_texts=["This is a query document"])
    assert len(search_result) > 0
    query_texts = ["Qdrant has Langchain integrations", "Qdrant also has Llama Index integrations"]
    results = local_client.query(collection_name, query_texts)
    assert len(results) > 0

    # TODO: Add assertions to verify that the query returned the expected results

if __name__ == "__main__":
    test_add_without_query()
    test_query()