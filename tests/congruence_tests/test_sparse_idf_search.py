import tempfile

import pytest

from qdrant_client.client_base import QdrantBase
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
        self.query_text = generate_random_sparse_vector(sparse_text_vector_size, density=0.3)

    def simple_search_text(self, client: QdrantBase) -> list[models.ScoredPoint]:
        return client.query_points(
            collection_name=COLLECTION_NAME,
            using="sparse-text",
            query=self.query_text,
            with_payload=True,
            with_vectors=["sparse-text"],
            limit=10,
        ).points

    def search_with_idf(
        self, client: QdrantBase, idf: models.IdfParams | None = None
    ) -> list[models.ScoredPoint]:
        return client.query_points(
            collection_name=COLLECTION_NAME,
            using="sparse-text",
            query=self.query_text,
            search_params=models.SearchParams(idf=idf) if idf is not None else None,
            with_payload=True,
            limit=10,
        ).points


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


def corpus_filter(digits: list[int]) -> models.Filter:
    return models.Filter(
        must=[models.FieldCondition(key="rand_digit", match=models.MatchAny(any=digits))]
    )


@pytest.mark.parametrize(
    "idf",
    [
        None,
        models.IdfScope.GLOBAL,
        models.IdfCorpusParams(corpus=corpus_filter([0, 1, 2])),
        models.IdfCorpusParams(corpus=corpus_filter([5, 6, 7, 8, 9])),
        # a corpus matching nothing: N and every df collapse to 0
        models.IdfCorpusParams(corpus=corpus_filter([42])),
        # an empty filter selects everything, so it must behave like the global scope
        models.IdfCorpusParams(corpus=models.Filter()),
    ],
    ids=["unset", "global", "corpus_low", "corpus_high", "corpus_empty_match", "corpus_match_all"],
)
def test_idf_scope(idf: models.IdfParams | None):
    fixture_points = generate_sparse_fixtures(
        vectors_sizes={"sparse-text": sparse_text_vector_size},
        even_sparse=False,
    )

    searcher = TestSimpleSparseSearcher()

    local_client = init_local()
    init_client(
        local_client,
        fixture_points,
        sparse_vectors_config=sparse_vectors_idf_config,
        vectors_config={},
    )

    remote_client = init_remote()
    init_client(
        remote_client,
        fixture_points,
        sparse_vectors_config=sparse_vectors_idf_config,
        vectors_config={},
    )

    compare_client_results(local_client, remote_client, searcher.search_with_idf, idf=idf)


def test_idf_scope_narrows_statistics():
    """Narrowing the corpus must change the scores, and change them the same way on both clients.

    Without this, `test_idf_scope` above could pass while both clients ignored `idf` entirely.
    """
    fixture_points = generate_sparse_fixtures(
        vectors_sizes={"sparse-text": sparse_text_vector_size},
        even_sparse=False,
    )

    local_client = init_local()
    init_client(
        local_client,
        fixture_points,
        sparse_vectors_config=sparse_vectors_idf_config,
        vectors_config={},
    )

    remote_client = init_remote()
    init_client(
        remote_client,
        fixture_points,
        sparse_vectors_config=sparse_vectors_idf_config,
        vectors_config={},
    )

    query = generate_random_sparse_vector(sparse_text_vector_size, density=0.3)

    def search(idf: models.IdfParams | None):
        return lambda client: client.query_points(
            COLLECTION_NAME,
            using="sparse-text",
            query=query,
            search_params=models.SearchParams(idf=idf) if idf is not None else None,
            limit=10,
        ).points

    narrowed = models.IdfCorpusParams(corpus=corpus_filter([0, 1]))

    # an unset scope, an explicit global scope and a corpus selecting everything all agree,
    # while a narrower corpus scores differently - on each client and identically across them
    for idf in (None, models.IdfScope.GLOBAL, models.IdfCorpusParams(corpus=models.Filter())):
        compare_client_results(local_client, remote_client, search(idf))
    compare_client_results(local_client, remote_client, search(narrowed))

    for client in (local_client, remote_client):
        globally = [point.score for point in search(None)(client)]
        assert [
            point.score for point in search(narrowed)(client)
        ] != globally, "narrowing the IDF corpus did not change the scores"


def test_idf_ignores_points_without_the_sparse_vector():
    """IDF statistics only count points that actually carry the sparse vector."""
    points = generate_sparse_fixtures(
        vectors_sizes={"sparse-text": sparse_text_vector_size},
        even_sparse=False,
    )
    # half of the points carry no sparse vector at all
    for point in points[::2]:
        point.vector = {}

    local_client = init_local()
    init_client(
        local_client,
        points,
        sparse_vectors_config=sparse_vectors_idf_config,
        vectors_config={},
    )

    remote_client = init_remote()
    init_client(
        remote_client,
        points,
        sparse_vectors_config=sparse_vectors_idf_config,
        vectors_config={},
    )

    searcher = TestSimpleSparseSearcher()
    compare_client_results(local_client, remote_client, searcher.search_with_idf)


def test_idf_scope_in_prefetch():
    fixture_points = generate_sparse_fixtures(
        vectors_sizes={"sparse-text": sparse_text_vector_size},
        even_sparse=False,
    )

    local_client = init_local()
    init_client(
        local_client,
        fixture_points,
        sparse_vectors_config=sparse_vectors_idf_config,
        vectors_config={},
    )

    remote_client = init_remote()
    init_client(
        remote_client,
        fixture_points,
        sparse_vectors_config=sparse_vectors_idf_config,
        vectors_config={},
    )

    query = generate_random_sparse_vector(sparse_text_vector_size, density=0.3)

    for idf in (
        models.IdfScope.GLOBAL,
        models.IdfCorpusParams(corpus=corpus_filter([0, 1, 2, 3])),
    ):
        compare_client_results(
            local_client,
            remote_client,
            lambda c, idf=idf: c.query_points(
                COLLECTION_NAME,
                prefetch=models.Prefetch(
                    query=query,
                    using="sparse-text",
                    limit=20,
                    params=models.SearchParams(idf=idf),
                ),
                query=models.FusionQuery(fusion=models.Fusion.RRF),
                limit=10,
            ).points,
        )


def test_search_with_persistence():
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

        local_client.close()
        local_client_2 = init_local(tmpdir)

        remote_client = init_remote()
        init_client(
            remote_client,
            fixture_points,
            sparse_vectors_config=sparse_vectors_idf_config,
            vectors_config={},
        )

        compare_client_results(local_client_2, remote_client, searcher.simple_search_text)


@pytest.mark.parametrize("operation", ["points", "vectors_upsert", "vectors_update"])
def test_idf_after_deletion(operation: str):
    """Deleted points must leave the IDF corpus, and come back when they are re-added.

    Only the corpus scope is compared. `IdfScope.GLOBAL` is measured over the sparse index on
    the server, which goes on counting deleted points for as long as they sit in that index,
    while local mode drops them at once - the two legitimately disagree there.
    """
    fixture_points = generate_sparse_fixtures(
        vectors_sizes={"sparse-text": sparse_text_vector_size},
        even_sparse=False,
    )

    clients = []
    for client in (init_local(), init_remote()):
        init_client(
            client,
            fixture_points,
            sparse_vectors_config=sparse_vectors_idf_config,
            vectors_config={},
        )
        clients.append(client)
    local_client, remote_client = clients

    query = generate_random_sparse_vector(sparse_text_vector_size, density=0.3)
    match_all = models.IdfCorpusParams(corpus=models.Filter())

    def search(client: QdrantBase) -> list[models.ScoredPoint]:
        return client.query_points(
            COLLECTION_NAME,
            using="sparse-text",
            query=query,
            search_params=models.SearchParams(idf=match_all),
            limit=10,
        ).points

    compare_client_results(local_client, remote_client, search)
    whole_corpus = [point.score for point in search(local_client)]

    doomed = fixture_points[: len(fixture_points) // 2]
    doomed_ids = [point.id for point in doomed]

    # repeating a deletion must not take the same document frequencies out twice
    for _ in range(2):
        for client in clients:
            if operation == "points":
                client.delete(COLLECTION_NAME, points_selector=doomed_ids, wait=True)
            else:
                client.delete_vectors(
                    COLLECTION_NAME, vectors=["sparse-text"], points=doomed_ids, wait=True
                )
        compare_client_results(local_client, remote_client, search)

    assert [
        point.score for point in search(local_client)
    ] != whole_corpus, "deleting half of the corpus did not change the IDF scores"

    # re-adding the vectors puts the points back into the corpus, without counting the ones
    # that were never taken out a second time
    for client in clients:
        if operation == "vectors_update":
            client.update_vectors(
                COLLECTION_NAME,
                [models.PointVectors(id=point.id, vector=point.vector) for point in doomed],
                wait=True,
            )
        else:
            client.upsert(COLLECTION_NAME, doomed, wait=True)

    compare_client_results(local_client, remote_client, search)
    assert [point.score for point in search(local_client)] == pytest.approx(
        whole_corpus
    ), "the corpus did not return to its original size"


@pytest.mark.parametrize("operation", ["update_vectors", "delete_vectors"])
@pytest.mark.parametrize("missing", ["unknown", "deleted"])
def test_vector_update_with_a_missing_point(operation: str, missing: str):
    """A point the collection does not hold does not stop the rest of the request.

    The server writes every other point in it and answers 404 afterwards, naming the first id
    it could not find. A deleted point counts as one of those: it keeps its slot locally so
    that internal ids stay stable, but it is gone as far as the request is concerned - and
    writing to it would put it back into the IDF corpus, and into storage on the next reopen.
    """
    fixture_points = generate_sparse_fixtures(
        num=10,
        vectors_sizes={"sparse-text": sparse_text_vector_size},
        even_sparse=False,
    )
    replacement = generate_random_sparse_vector(sparse_text_vector_size, density=0.3)

    clients = []
    for client in (init_local(), init_remote()):
        init_client(
            client,
            fixture_points,
            sparse_vectors_config=sparse_vectors_idf_config,
            vectors_config={},
        )
        clients.append(client)

    absent = fixture_points[-1].id
    for client in clients:
        client.delete(COLLECTION_NAME, points_selector=[absent], wait=True)
    if missing == "unknown":
        absent = max(point.id for point in fixture_points) + 1_000

    # the missing id sits in the middle, so anything written after it proves the request was
    # not abandoned at the first failure
    targets = [fixture_points[0].id, absent, fixture_points[1].id]

    for client in clients:
        with pytest.raises(Exception):
            if operation == "update_vectors":
                client.update_vectors(
                    COLLECTION_NAME,
                    [
                        models.PointVectors(id=point_id, vector={"sparse-text": replacement})
                        for point_id in targets
                    ],
                    wait=True,
                )
            else:
                client.delete_vectors(
                    COLLECTION_NAME, vectors=["sparse-text"], points=targets, wait=True
                )

    def scroll(client: QdrantBase) -> list[models.Record]:
        return client.scroll(
            COLLECTION_NAME, limit=len(fixture_points), with_vectors=True, with_payload=True
        )[0]

    compare_client_results(*clients, scroll)
    compare_client_results(
        *clients,
        lambda client: client.query_points(
            COLLECTION_NAME,
            using="sparse-text",
            query=replacement,
            search_params=models.SearchParams(idf=models.IdfCorpusParams(corpus=models.Filter())),
            limit=10,
        ).points,
    )
