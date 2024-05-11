import pytest

from qdrant_client import QdrantClient
from tests.config import TEST_DATA_DIR

DOCS_EXAMPLE = {
    "documents": [
        "Qdrant has Langchain integrations",
        "Qdrant also has Llama Index integrations",
    ],
    "metadata": [{"source": "Langchain-docs"}, {"source": "LlamaIndex-docs"}],
    "ids": [42, 2000],
}
IMAGE_EXAMPLE = TEST_DATA_DIR / "image.jpeg"
CROSS_MODEL_EXAMPLE = {**DOCS_EXAMPLE, "images": [IMAGE_EXAMPLE, IMAGE_EXAMPLE]}


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
    dim, dist = local_client._get_text_model_params(embedding_model_name)
    assert dim == 384

    # Use the initialized model to add documents with vector embeddings
    local_client.add(collection_name=collection_name, **DOCS_EXAMPLE)
    assert local_client.count(collection_name).count == 2


@pytest.mark.parametrize("dense_model_name", ["sentence-transformers/all-MiniLM-L6-v2", None])
def test_dense(tmp_path, dense_model_name):
    # tmp_path is a pytest fixture
    local_client = QdrantClient(path=str(tmp_path / "db_test_dense"))
    collection_name = "demo_collection"
    docs = [
        "Qdrant has Langchain integrations",
        "Qdrant also has Llama Index integrations",
    ]

    if not local_client._FASTEMBED_INSTALLED:
        with pytest.raises(ImportError):
            local_client.add(collection_name, docs)
    else:
        if dense_model_name is not None:
            local_client.set_model(embedding_model_name=dense_model_name)
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
        local_client.close()

        # region query without preliminary add
        local_client = QdrantClient(path=str(tmp_path / "db_test_dense"))
        if dense_model_name is not None:
            local_client.set_model(embedding_model_name=dense_model_name)
        local_client.query(collection_name=collection_name, query_text="This is a query document")
        assert len(search_result) > 0
        local_client.close()
        # endregion


def test_image(tmp_path):
    # tmp_path is a pytest fixture
    local_client = QdrantClient(path=str(tmp_path / "db_test_image"))
    collection_name = "demo_collection"
    data = [IMAGE_EXAMPLE, IMAGE_EXAMPLE]

    if not local_client._FASTEMBED_INSTALLED:
        with pytest.raises(ImportError):
            local_client.set_image_model("Qdrant/clip-ViT-B-32-vision")
            local_client.add(collection_name, data)
    else:
        local_client.set_image_model("Qdrant/clip-ViT-B-32-vision")
        local_client.add(collection_name, images=data)
        assert len(local_client.get_collection(collection_name).config.params.vectors) == 1
        assert local_client.count(collection_name).count == 2

        search_result = local_client.query(
            collection_name=collection_name, query_image=IMAGE_EXAMPLE
        )
        assert len(search_result) > 0
        local_client.close()

        # region query without preliminary add
        local_client = QdrantClient(path=str(tmp_path / "db_test_image"))
        local_client.set_image_model("Qdrant/clip-ViT-B-32-vision")
        search_result = local_client.query(
            collection_name=collection_name, query_image=IMAGE_EXAMPLE
        )
        assert len(search_result) > 0
        local_client.close()
        # endregion


@pytest.mark.parametrize("dense_model_name", ["sentence-transformers/all-MiniLM-L6-v2", None])
def test_hybrid(tmp_path, dense_model_name):
    # tmp_path is a pytest fixture
    local_client = QdrantClient(path=str(tmp_path / "db_test_hybrid"))
    collection_name = "hybrid_collection"

    if not local_client._FASTEMBED_INSTALLED:
        pytest.skip("FastEmbed is not installed, skipping test")

    if dense_model_name is not None:
        local_client.set_model(embedding_model_name=dense_model_name)

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
    local_client.close()

    # region query without preliminary add
    local_client = QdrantClient(path=str(tmp_path / "db_test_hybrid"))
    if dense_model_name is not None:
        local_client.set_model(embedding_model_name=dense_model_name)
    local_client.set_sparse_model(embedding_model_name="prithvida/Splade_PP_en_v1")
    hybrid_search_result = local_client.query(
        collection_name=collection_name, query_text="This is a query document"
    )
    assert len(hybrid_search_result) > 0
    local_client.close()
    # endregion


def test_cross_model(tmp_path):
    # tmp_path is a pytest fixture
    local_client = QdrantClient(path=str(tmp_path / "db_test_cross_model"))
    collection_name = "demo_collection"
    if not local_client._FASTEMBED_INSTALLED:
        pytest.skip("FastEmbed is not installed, skipping test")

    local_client.set_model("Qdrant/clip-ViT-B-32-text")
    local_client.set_image_model("Qdrant/clip-ViT-B-32-vision")

    local_client.add(collection_name=collection_name, **CROSS_MODEL_EXAMPLE)
    assert len(local_client.get_collection(collection_name).config.params.vectors) == 2
    assert local_client.count(collection_name).count == 2

    search_result = local_client.query(
        collection_name=collection_name, query_text="This is a query document"
    )

    assert len(search_result) > 0
    assert search_result[0].id == 2000

    search_result = local_client.query(collection_name=collection_name, query_image=IMAGE_EXAMPLE)

    assert len(search_result) > 0
    assert search_result[0].id == 2000

    search_result = local_client.query(
        collection_name=collection_name,
        query_text={local_client.get_image_vector_field_name(): "This is a query document"},
    )

    assert len(search_result) > 0
    assert search_result[0].id == 2000

    search_result = local_client.query(
        collection_name=collection_name,
        query_image={local_client.get_vector_field_name(): IMAGE_EXAMPLE},
    )

    assert len(search_result) > 0
    assert search_result[0].id == 2000

    with pytest.raises(ValueError):
        local_client.query(
            collection_name=collection_name,
            query_text="This is a query document",
            query_image=IMAGE_EXAMPLE,
        )

    local_client.close()
    local_client = QdrantClient(path=str(tmp_path / "db_test_cross_model"))
    local_client.set_model("Qdrant/clip-ViT-B-32-text")
    search_result = local_client.query(
        collection_name=collection_name,
        query_text={
            local_client.vector_field_from_model(
                "Qdrant/clip-ViT-B-32-vision"
            ): "This is a query document"
        },
    )

    assert len(search_result) > 0
    assert search_result[0].id == 2000
    local_client.close()

    local_client = QdrantClient(path=str(tmp_path / "db_test_cross_model"))
    local_client.set_image_model("Qdrant/clip-ViT-B-32-vision")
    search_result = local_client.query(
        collection_name=collection_name,
        query_image={
            local_client.vector_field_from_model("Qdrant/clip-ViT-B-32-text"): IMAGE_EXAMPLE
        },
    )

    assert len(search_result) > 0
    assert search_result[0].id == 2000
    local_client.close()


def test_query_batch(tmp_path):
    # tmp_path is a pytest fixture
    local_client = QdrantClient(path=str(tmp_path / "db_test_query_batch"))

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

    local_client.close()

    # region query without preliminary add
    local_client = QdrantClient(path=str(tmp_path / "db_test_query_batch"))
    dense_search_result = local_client.query_batch(
        collection_name=dense_collection_name, query_texts=query_texts
    )
    assert len(dense_search_result) == len(query_texts)
    assert all(len(result) > 0 for result in dense_search_result)

    local_client.set_sparse_model(embedding_model_name="prithvida/Splade_PP_en_v1")
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
    local_client.close()
    # endregion
