from typing import List

from qdrant_client.client_base import QdrantBase
from qdrant_client.conversions.common_types import NamedSparseVector
from qdrant_client.http.models import models
from tests.congruence_tests.test_common import (
    COLLECTION_NAME,
    compare_client_results,
    generate_sparse_fixtures,
    init_client,
    init_local,
    init_remote,
    sparse_text_vector_size,
)
from tests.fixtures.points import generate_random_sparse_vector

sparse_vectors_idf_config = {
    "sparse-text": models.SparseVectorParams(
        modifier=models.Modifier.IDF,
    ),
}


class TestSimpleSparseSearcher:
    __test__ = False

    def __init__(self):
        self.query_text = generate_random_sparse_vector(sparse_text_vector_size, density=0.1)

    def simple_search_text(self, client: QdrantBase) -> List[models.ScoredPoint]:
        return client.search(
            collection_name=COLLECTION_NAME,
            query_vector=NamedSparseVector(name="sparse-text", vector=self.query_text),
            with_payload=True,
            with_vectors=["sparse-text"],
            limit=10,
        )


def test_simple_search():
    fixture_points = generate_sparse_fixtures(
        vectors_sizes={"sparse-text": sparse_text_vector_size},
        even_sparse=False,
        with_payload=False,
    )

    searcher = TestSimpleSparseSearcher()

    local_client = init_local()
    init_client(
        local_client,
        fixture_points,
        sparse_vectors_config=sparse_vectors_idf_config,
        vectors_config={},
    )

    assert (
        local_client.get_collection(COLLECTION_NAME)
        .config.params.sparse_vectors["sparse-text"]
        .modifier
        == models.Modifier.IDF
    )

    remote_client = init_remote()
    init_client(
        remote_client,
        fixture_points,
        sparse_vectors_config=sparse_vectors_idf_config,
        vectors_config={},
    )

    compare_client_results(local_client, remote_client, searcher.simple_search_text)

    local_client.update_collection(
        collection_name=COLLECTION_NAME,
        sparse_vectors_config={
            "sparse-text": models.SparseVectorParams(
                modifier=models.Modifier.NONE,
            )
        },
    )

    assert (
        local_client.get_collection(COLLECTION_NAME)
        .config.params.sparse_vectors["sparse-text"]
        .modifier
        == models.Modifier.NONE
    )


def test_search_with_persistence():
    import tempfile

    fixture_points = generate_sparse_fixtures(
        vectors_sizes={"sparse-text": sparse_text_vector_size},
        even_sparse=False,
        with_payload=False,
    )
    searcher = TestSimpleSparseSearcher()
    with tempfile.TemporaryDirectory() as tmpdir:
        local_client = init_local(tmpdir)
        init_client(
            local_client,
            fixture_points,
            sparse_vectors_config=sparse_vectors_idf_config,
            vectors_config={},
        )

        del local_client
        local_client_2 = init_local(tmpdir)

        remote_client = init_remote()
        init_client(
            remote_client,
            fixture_points,
            sparse_vectors_config=sparse_vectors_idf_config,
            vectors_config={},
        )

        compare_client_results(local_client_2, remote_client, searcher.simple_search_text)
