import math
from dataclasses import dataclass

import numpy as np
import pytest

from qdrant_client.client_base import QdrantBase
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.http.models import models
from qdrant_client.local.multi_distances import calculate_multi_distance_core
from tests.congruence_tests.test_common import (
    COLLECTION_NAME,
    code_vector_size,
    compare_client_results,
    has_near_tie_at_limit,
    image_vector_size,
    init_client,
    init_local,
    init_remote,
    text_vector_size,
    multi_vector_config,
    generate_multivector_fixtures,
)
from tests.fixtures.points import generate_random_multivector


@dataclass
class MmrCase:
    """An MMR query kept as data, so that a failed comparison can be replayed in local mode."""

    query: list[list[float]]
    using: str
    mmr: models.Mmr
    score_threshold: float | None = None
    limit: int = 10

    def run(self, client: QdrantBase) -> models.QueryResponse:
        return client.query_points(
            collection_name=COLLECTION_NAME,
            query=models.NearestQuery(nearest=self.query, mmr=self.mmr),
            using=self.using,
            score_threshold=self.score_threshold,
            limit=self.limit,
        )


class TestSimpleSearcher:
    __test__ = False

    def __init__(self):
        self.query_text = generate_random_multivector(text_vector_size, 10)
        self.query_image = generate_random_multivector(image_vector_size, 10)
        self.query_code = generate_random_multivector(code_vector_size, 10)

    def simple_search_text(self, client: QdrantBase) -> list[models.ScoredPoint]:
        return client.query_points(
            collection_name=COLLECTION_NAME,
            query=self.query_text,
            with_payload=True,
            using="multi-text",
            limit=10,
        ).points

    def simple_search_image(self, client: QdrantBase) -> list[models.ScoredPoint]:
        return client.query_points(
            collection_name=COLLECTION_NAME,
            query=self.query_image,
            using="multi-image",
            with_payload=True,
            limit=10,
        ).points

    def simple_search_code(self, client: QdrantBase) -> list[models.ScoredPoint]:
        return client.query_points(
            collection_name=COLLECTION_NAME,
            query=self.query_code,
            using="multi-code",
            with_payload=True,
            limit=10,
        ).points

    def simple_search_unnamed(self, client: QdrantBase) -> list[models.ScoredPoint]:
        return client.query_points(
            collection_name=COLLECTION_NAME,
            query=self.query_text,
            with_payload=True,
            limit=10,
        ).points

    def default_mmr_query(self) -> MmrCase:
        return MmrCase(query=self.query_text, using="multi-text", mmr=models.Mmr())

    def mmr_query_parametrized(self) -> MmrCase:
        return MmrCase(
            query=self.query_text,
            using="multi-text",
            mmr=models.Mmr(diversity=0.3, candidates_limit=30),
        )

    def mmr_query_parametrized_score_threshold(self) -> MmrCase:
        return MmrCase(
            query=self.query_text,
            using="multi-text",
            mmr=models.Mmr(diversity=0.3, candidates_limit=30),
            score_threshold=7.0,
        )

    def mmr_query_parametrized_dot(self) -> MmrCase:
        return MmrCase(
            query=self.query_image,
            using="multi-image",
            mmr=models.Mmr(diversity=0.3, candidates_limit=30),
        )

    def mmr_query_parametrized_dot_score_threshold(self) -> MmrCase:
        return MmrCase(
            query=self.query_image,
            using="multi-image",
            mmr=models.Mmr(diversity=0.3, candidates_limit=30),
            score_threshold=260.0,
        )

    def mmr_query_parametrized_euclid(self) -> MmrCase:
        return MmrCase(
            query=self.query_code,
            using="multi-code",
            mmr=models.Mmr(diversity=0.3, candidates_limit=30),
        )

    def mmr_query_parametrized_euclid_score_threshold(self) -> MmrCase:
        return MmrCase(
            query=self.query_code,
            using="multi-code",
            mmr=models.Mmr(diversity=0.3, candidates_limit=30),
            score_threshold=10.0,
        )


def replay_mmr_selection(
    client: QdrantBase, case: MmrCase, rel_tol: float = 1e-5
) -> list[models.ExtendedPointId] | None:
    """Replay local mode's MMR selection, return the picked ids, or None at a near-tie.

    Local mode and core don't compute scores bit for bit alike, so two choices closer than
    about 1e-7 relative can go either way on each side: candidates around the candidates limit
    or around `score_threshold`, or the two best MMR scores of a greedy step. MMR picks
    greedily, so one flipped pick reorders every later rank. The replay scores with the
    similarity function local mode's MMR uses.
    """
    mmr = case.mmr
    candidates_limit = mmr.candidates_limit if mmr.candidates_limit is not None else case.limit
    if has_near_tie_at_limit(client, case.query, case.using, candidates_limit):
        return None
    if case.score_threshold is not None:
        nearest = client.query_points(
            COLLECTION_NAME, query=case.query, using=case.using, limit=candidates_limit
        ).points
        if any(
            math.isclose(point.score, case.score_threshold, rel_tol=rel_tol) for point in nearest
        ):
            return None

    candidates = client.query_points(
        COLLECTION_NAME,
        query=case.query,
        using=case.using,
        limit=candidates_limit,
        score_threshold=case.score_threshold,
        with_vectors=[case.using],
    ).points
    if not candidates:
        return []

    distance = multi_vector_config[case.using].distance
    vectors = [np.array(point.vector[case.using]) for point in candidates]
    relevance = calculate_multi_distance_core(np.array(case.query), vectors, distance)
    similarity = [calculate_multi_distance_core(vector, vectors, distance) for vector in vectors]
    scale = max(np.abs(relevance).max(), np.abs(np.array(similarity)).max())
    lambda_ = 1.0 - (mmr.diversity if mmr.diversity is not None else 0.5)

    pending = list(range(len(candidates)))
    selected: list[int] = []
    while pending and len(selected) < case.limit:
        if selected:
            scores = [
                lambda_ * relevance[i] - (1.0 - lambda_) * max(similarity[i][j] for j in selected)
                for i in pending
            ]
        else:  # the first pick is the most relevant candidate
            scores = [relevance[i] for i in pending]
        best, *rest = np.argsort(scores)[::-1]
        if rest and scores[best] - scores[rest[0]] <= rel_tol * scale:
            return None
        selected.append(pending.pop(best))
    return [candidates[i].id for i in selected]


def compare_mmr_results(
    local_client: QdrantBase, remote_client: QdrantBase, case: MmrCase
) -> None:
    try:
        compare_client_results(local_client, remote_client, case.run)
    except AssertionError:
        # at a near-tie either result is right, otherwise the mismatch is real
        if replay_mmr_selection(local_client, case) is not None:
            raise


def test_simple():
    fixture_points = generate_multivector_fixtures(100)

    searcher = TestSimpleSearcher()

    local_client = init_local()
    init_client(local_client, fixture_points, vectors_config=multi_vector_config)

    remote_client = init_remote()
    init_client(remote_client, fixture_points, vectors_config=multi_vector_config)

    compare_client_results(local_client, remote_client, searcher.simple_search_text)
    compare_client_results(local_client, remote_client, searcher.simple_search_image)
    compare_client_results(local_client, remote_client, searcher.simple_search_code)


def test_simple_opt_vectors_search():
    fixture_points = generate_multivector_fixtures(100, skip_vectors=True)

    searcher = TestSimpleSearcher()

    local_client = init_local()
    init_client(local_client, fixture_points, vectors_config=multi_vector_config)

    remote_client = init_remote()
    init_client(remote_client, fixture_points, vectors_config=multi_vector_config)

    compare_client_results(local_client, remote_client, searcher.simple_search_text)
    compare_client_results(local_client, remote_client, searcher.simple_search_image)
    compare_client_results(local_client, remote_client, searcher.simple_search_code)


def test_mmr():
    fixture_points = generate_multivector_fixtures(10)

    searcher = TestSimpleSearcher()

    local_client = init_local()
    init_client(local_client, fixture_points, vectors_config=multi_vector_config)

    remote_client = init_remote()
    init_client(remote_client, fixture_points, vectors_config=multi_vector_config)

    compare_mmr_results(local_client, remote_client, searcher.default_mmr_query())
    compare_mmr_results(local_client, remote_client, searcher.mmr_query_parametrized())
    compare_mmr_results(
        local_client, remote_client, searcher.mmr_query_parametrized_score_threshold()
    )
    compare_mmr_results(local_client, remote_client, searcher.mmr_query_parametrized_dot())
    compare_mmr_results(
        local_client, remote_client, searcher.mmr_query_parametrized_dot_score_threshold()
    )
    compare_mmr_results(local_client, remote_client, searcher.mmr_query_parametrized_euclid())
    compare_mmr_results(
        local_client, remote_client, searcher.mmr_query_parametrized_euclid_score_threshold()
    )


def test_single_vector():
    fixture_points = generate_multivector_fixtures(num=100, vectors_sizes=text_vector_size)

    searcher = TestSimpleSearcher()

    vectors_config = models.VectorParams(
        size=text_vector_size,
        distance=models.Distance.DOT,
        multivector_config=models.MultiVectorConfig(
            comparator=models.MultiVectorComparator.MAX_SIM
        ),
    )

    local_client = init_local()
    init_client(local_client, fixture_points, vectors_config=vectors_config)

    remote_client = init_remote()
    init_client(remote_client, fixture_points, vectors_config=vectors_config)

    compare_client_results(local_client, remote_client, searcher.simple_search_unnamed)


def test_search_with_persistence():
    import tempfile

    fixture_points = generate_multivector_fixtures(100)
    searcher = TestSimpleSearcher()
    with tempfile.TemporaryDirectory() as tmpdir:
        local_client = init_local(tmpdir)
        init_client(local_client, fixture_points, vectors_config=multi_vector_config)

        local_client.close()
        local_client_2 = init_local(tmpdir)

        remote_client = init_remote()
        init_client(remote_client, fixture_points, vectors_config=multi_vector_config)

        compare_client_results(local_client_2, remote_client, searcher.simple_search_text)
        compare_client_results(local_client_2, remote_client, searcher.simple_search_image)
        compare_client_results(local_client_2, remote_client, searcher.simple_search_code)


def test_search_invalid_vector_type():
    fixture_points = generate_multivector_fixtures(100)

    local_client = init_local()
    init_client(local_client, fixture_points, vectors_config=multi_vector_config)

    remote_client = init_remote()
    init_client(remote_client, fixture_points, vectors_config=multi_vector_config)

    vector_invalid_type = {"multi-text": [1, 2, 3, 4]}
    with pytest.raises(ValueError):
        local_client.query_points(
            collection_name=COLLECTION_NAME, query=vector_invalid_type, using="multi-text"
        )

    with pytest.raises(ValueError):
        remote_client.query_points(
            collection_name=COLLECTION_NAME, query=vector_invalid_type, using="multi-text"
        )


def test_query_with_nan():
    fixture_points = generate_multivector_fixtures(10)

    local_client = init_local()
    init_client(local_client, fixture_points, vectors_config=multi_vector_config)

    remote_client = init_remote()
    init_client(remote_client, fixture_points, vectors_config=multi_vector_config)

    vector = generate_random_multivector(text_vector_size, 10)
    vector[0][4] = np.nan
    with pytest.raises(ValueError):
        local_client.query_points(COLLECTION_NAME, query=vector, using="multi-text")
    with pytest.raises(UnexpectedResponse):
        remote_client.query_points(COLLECTION_NAME, query=vector, using="multi-text")

    single_multi_vector_config = models.VectorParams(
        size=text_vector_size,
        distance=models.Distance.COSINE,
        multivector_config=models.MultiVectorConfig(
            comparator=models.MultiVectorComparator.MAX_SIM
        ),
    )

    local_client.delete_collection(COLLECTION_NAME)
    local_client.create_collection(COLLECTION_NAME, vectors_config=single_multi_vector_config)

    remote_client.delete_collection(COLLECTION_NAME)
    remote_client.create_collection(COLLECTION_NAME, vectors_config=single_multi_vector_config)

    fixture_points = generate_multivector_fixtures(vectors_sizes=text_vector_size)
    init_client(local_client, fixture_points, vectors_config=single_multi_vector_config)
    init_client(remote_client, fixture_points, vectors_config=single_multi_vector_config)

    with pytest.raises(ValueError):
        local_client.query_points(COLLECTION_NAME, query=vector)
    with pytest.raises(UnexpectedResponse):
        remote_client.query_points(COLLECTION_NAME, query=vector)
