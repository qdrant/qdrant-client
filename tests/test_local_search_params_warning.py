"""Regression test for #1083.

In local mode there is no ANN index — search is always exact (brute-force) — so
`search_params` (e.g. `exact`, `hnsw_ef`) cannot change results. Previously it was
silently dropped, which surprised users (the reporter saw `exact=True` "have no effect").
The client should warn rather than silently ignore it. Local/in-memory only: no server,
no network, no downloads.
"""
import contextlib
import warnings

from qdrant_client import QdrantClient, models
from qdrant_client.common import client_warnings

_WARN_IDX = "local_query_points_search_params_ignored"


@contextlib.contextmanager
def _fresh_warning(idx: str):
    """Let the once-per-run warning fire, then restore the global cache so this test
    doesn't leave ``idx`` recorded and make later warning assertions order-dependent."""
    had = idx in client_warnings.SEEN_MESSAGES
    client_warnings.SEEN_MESSAGES.discard(idx)
    try:
        yield
    finally:
        if had:
            client_warnings.SEEN_MESSAGES.add(idx)
        else:
            client_warnings.SEEN_MESSAGES.discard(idx)


def _client_with_points() -> QdrantClient:
    client = QdrantClient(location=":memory:")
    client.create_collection(
        "c", vectors_config=models.VectorParams(size=4, distance=models.Distance.COSINE)
    )
    client.upsert(
        "c",
        points=[
            models.PointStruct(id=i, vector=[float(i), 0.0, 0.0, 1.0], payload={"group": i % 2})
            for i in range(5)
        ],
    )
    return client


def test_query_points_warns_when_search_params_passed_in_local_mode():
    client = _client_with_points()
    with _fresh_warning(_WARN_IDX), warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        res = client.query_points(
            "c",
            query=[1.0, 0.0, 0.0, 1.0],
            search_params=models.SearchParams(exact=True),
            limit=3,
        )
    assert any("has no effect" in str(w.message) for w in caught)
    assert len(res.points) == 3  # results are still correct (always exact)


def test_query_batch_points_warns_when_search_params_passed_in_local_mode():
    client = _client_with_points()
    with _fresh_warning(_WARN_IDX), warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        res = client.query_batch_points(
            "c",
            requests=[
                models.QueryRequest(
                    query=[1.0, 0.0, 0.0, 1.0],
                    params=models.SearchParams(exact=True),
                    limit=3,
                )
            ],
        )
    assert any("has no effect" in str(w.message) for w in caught)
    assert len(res) == 1 and len(res[0].points) == 3


def test_query_points_groups_warns_when_search_params_passed_in_local_mode():
    client = _client_with_points()
    with _fresh_warning(_WARN_IDX), warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        res = client.query_points_groups(
            "c",
            group_by="group",
            query=[1.0, 0.0, 0.0, 1.0],
            search_params=models.SearchParams(exact=True),
            limit=2,
            group_size=1,
        )
    assert any("has no effect" in str(w.message) for w in caught)
    assert len(res.groups) == 2


def test_query_points_no_warning_without_search_params():
    client = _client_with_points()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        res = client.query_points("c", query=[1.0, 0.0, 0.0, 1.0], limit=3)
    assert not any("has no effect" in str(w.message) for w in caught)
    assert len(res.points) == 3
