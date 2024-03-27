import pytest

from qdrant_client import QdrantClient

DOCS_EXAMPLE = {
    "documents": [
        "Qdrant has Langchain integrations",
        "Qdrant also has Llama Index integrations",
    ],
    "metadata": [{"source": "Langchain-docs"}, {"source": "LlamaIndex-docs"}],
    "ids": [42, 2000],
}


def test_dense():
    local_client = QdrantClient(":memory:")
    collection_name = "demo_collection"
    docs = [
        "Qdrant has Langchain integrations",
        "Qdrant also has Llama Index integrations",
    ]

    if not local_client._FASTEMBED_INSTALLED:
        with pytest.raises(ImportError):
            local_client.add(collection_name, docs)
    else:
        local_client.add(collection_name=collection_name, documents=docs)
        assert local_client.count(collection_name).count == 2

        local_client.add(collection_name=collection_name, **DOCS_EXAMPLE)
        assert local_client.count(collection_name).count == 4

        id_ = DOCS_EXAMPLE["ids"][0]
        record = local_client.retrieve(collection_name, ids=[id_])[0]
        assert record.payload == {
            "document": DOCS_EXAMPLE["documents"][0],
            **DOCS_EXAMPLE["metadata"][0],
        }

        search_result = local_client.query(
            collection_name=collection_name, query_text="This is a query document"
        )

        assert len(search_result) > 0


def test_hybrid_query():
    local_client = QdrantClient(":memory:")
    collection_name = "hybrid_collection"

    if not local_client._FASTEMBED_INSTALLED:
        pytest.skip("FastEmbed is not installed, skipping test")

    local_client.set_sparse_model(embedding_model_name="prithvida/Splade_PP_en_v1")

    local_client.add(collection_name=collection_name, **DOCS_EXAMPLE)

    hybrid_search_result = local_client.query(
        collection_name=collection_name, query_text="This is a query document"
    )

    assert len(hybrid_search_result) > 0

    local_client.set_sparse_model(None)
    dense_search_result = local_client.query(
        collection_name=collection_name, query_text="This is a query document"
    )
    assert len(dense_search_result) > 0

    assert (
        hybrid_search_result[0].score != dense_search_result[0].score
    )  # hybrid search has score from fusion


def test_query_batch():
    local_client = QdrantClient(":memory:")

    dense_collection_name = "dense_collection"
    hybrid_collection_name = "hybrid_collection"

    if not local_client._FASTEMBED_INSTALLED:
        pytest.skip("FastEmbed is not installed, skipping test")

    local_client.add(collection_name=dense_collection_name, **DOCS_EXAMPLE)
    query_texts = ["This is a query document", "This is another query document"]
    dense_search_result = local_client.query_batch(
        collection_name=dense_collection_name, query_texts=query_texts
    )
    assert len(dense_search_result) == len(query_texts)
    assert all(len(result) > 0 for result in dense_search_result)

    local_client.set_sparse_model(embedding_model_name="prithvida/Splade_PP_en_v1")

    local_client.add(collection_name=hybrid_collection_name, **DOCS_EXAMPLE)

    hybrid_search_result = local_client.query_batch(
        collection_name=hybrid_collection_name, query_texts=query_texts
    )

    assert len(hybrid_search_result) == len(query_texts)
    assert all(len(result) > 0 for result in hybrid_search_result)

    single_dense_response = next(iter(dense_search_result))
    single_hybrid_response = next(iter(hybrid_search_result))

    assert (
        single_hybrid_response[0].score != single_dense_response[0].score
    )  # hybrid search has score from fusion


def test_set_model():
    local_client = QdrantClient(":memory:")
    collection_name = "demo_collection"
    embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
    if not local_client._FASTEMBED_INSTALLED:
        pytest.skip("FastEmbed is not installed, skipping test")

    local_client.set_model(
        embedding_model_name=embedding_model_name,
    )

    # Check if the model is initialized & cls.embeddings_models is set with expected values
    dim, dist = local_client._get_model_params(embedding_model_name)
    assert dim == 384

    # Use the initialized model to add documents with vector embeddings
    local_client.add(collection_name=collection_name, **DOCS_EXAMPLE)
    assert local_client.count(collection_name).count == 2
