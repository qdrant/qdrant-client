"""Regression test for #1083.

In local mode there is no ANN index — search is always exact (brute-force) — so
`search_params` (e.g. `exact`, `hnsw_ef`) cannot change results. Previously it was
silently dropped, which surprised users (the reporter saw `exact=True` "have no effect").
The client should warn rather than silently ignore it. Local/in-memory only: no server,
no network, no downloads.
"""
import warnings

from qdrant_client import QdrantClient, models
from qdrant_client.common import client_warnings

_WARN_IDX = "local_query_points_search_params_ignored"


def _client_with_points() -> QdrantClient:
    client = QdrantClient(location=":memory:")
    client.create_collection(
        "c", vectors_config=models.VectorParams(size=4, distance=models.Distance.COSINE)
    )
    client.upsert(
        "c",
        points=[models.PointStruct(id=i, vector=[float(i), 0.0, 0.0, 1.0]) for i in range(5)],
    )
    return client


def test_query_points_warns_when_search_params_passed_in_local_mode():
    client_warnings.SEEN_MESSAGES.discard(_WARN_IDX)  # allow the once-per-run warning to fire
    client = _client_with_points()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        res = client.query_points(
            "c",
            query=[1.0, 0.0, 0.0, 1.0],
            search_params=models.SearchParams(exact=True),
            limit=3,
        )
    assert any("has no effect" in str(w.message) for w in caught)
    assert len(res.points) == 3  # results are still correct (always exact)


def test_query_points_no_warning_without_search_params():
    client = _client_with_points()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        res = client.query_points("c", query=[1.0, 0.0, 0.0, 1.0], limit=3)
    assert not any("has no effect" in str(w.message) for w in caught)
    assert len(res.points) == 3
