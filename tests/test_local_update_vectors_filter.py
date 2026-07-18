"""Regression tests for issue #1237.

``LocalCollection.update_vectors`` previously did ``return None`` when a point
failed the ``update_filter`` check, which exited the entire batch loop and
silently skipped every remaining point. The fix is to ``continue`` so each
point is evaluated independently.

Covers:

* The reproducer from issue #1237 — a point that matches ``update_filter`` after
  a non-matching point is still updated.
* All-matching and none-matching boundaries.
* Async mirror: ``AsyncQdrantClient.update_vectors`` delegates to the sync
  collection, so the fix applies there too.

All tests use the in-memory backend (``":memory:"``); no filesystem or network
is touched.

Note: the in-memory backend normalizes vectors on every write under a COSINE
distance config, so expected values are computed from the post-normalization
form. Directionally distinct inputs are used to keep assertions unambiguous.
"""

from __future__ import annotations

import pytest

from qdrant_client import QdrantClient
from qdrant_client.async_qdrant_client import AsyncQdrantClient
from qdrant_client.http import models as rest


def _seeded_client() -> QdrantClient:
    """Three points; ids 1 and 3 carry payload type='a', id 2 carries type='b'."""
    client = QdrantClient(":memory:")
    client.create_collection(
        "t",
        vectors_config=rest.VectorParams(size=4, distance=rest.Distance.COSINE),
    )
    client.upsert(
        "t",
        points=[
            # Distinct unit vectors so per-point assertions are unambiguous.
            rest.PointStruct(id=1, vector=[1.0, 0.0, 0.0, 0.0], payload={"type": "a"}),
            rest.PointStruct(id=2, vector=[0.0, 1.0, 0.0, 0.0], payload={"type": "b"}),
            rest.PointStruct(id=3, vector=[0.0, 0.0, 1.0, 0.0], payload={"type": "a"}),
        ],
        wait=True,
    )
    return client


def _filter_match_type_a() -> rest.Filter:
    return rest.Filter(
        must=[rest.FieldCondition(key="type", match=rest.MatchValue(value="a"))],
    )


def _vectors_by_id(client: QdrantClient, ids: list[int]) -> dict[int, list[float]]:
    return {p.id: list(p.vector) for p in client.retrieve("t", ids=ids, with_vectors=True)}


class TestUpdateVectorsFilterSync:
    def test_later_matching_point_is_not_skipped_after_mismatch(self) -> None:
        """Reproducer from issue #1237: point 2 doesn't match, but point 3 must still update."""
        client = _seeded_client()

        client.update_vectors(
            "t",
            points=[
                # Directionally distinct so each update is identifiable after normalization.
                rest.PointVectors(id=1, vector=[1.0, 1.0, 0.0, 0.0]),  # matches type=a
                rest.PointVectors(id=2, vector=[0.0, 0.0, 1.0, 1.0]),  # doesn't match type=a
                rest.PointVectors(id=3, vector=[0.0, 0.0, 0.0, 1.0]),  # matches type=a
            ],
            update_filter=_filter_match_type_a(),
        )

        vectors = _vectors_by_id(client, [1, 2, 3])
        # Vector at id 1 normalized from [1, 1, 0, 0] -> [0.7071, 0.7071, 0, 0].
        assert vectors[1][0] == pytest.approx(0.7071, abs=1e-3)
        assert vectors[1][1] == pytest.approx(0.7071, abs=1e-3)
        # Non-matching point 2 was left alone; its original unit vector was [0, 1, 0, 0].
        assert vectors[2] == pytest.approx([0.0, 1.0, 0.0, 0.0], abs=1e-6), (
            "non-matching point must be left alone"
        )
        # Matching point 3 after mismatch — this is the bug under test.
        assert vectors[3] == pytest.approx([0.0, 0.0, 0.0, 1.0], abs=1e-6), (
            "matching point after a mismatch must still update (issue #1237)"
        )

    def test_all_points_match(self) -> None:
        client = _seeded_client()

        client.update_vectors(
            "t",
            points=[
                rest.PointVectors(id=1, vector=[2.0, 0.0, 0.0, 0.0]),
                rest.PointVectors(id=2, vector=[0.0, 3.0, 0.0, 0.0]),
                rest.PointVectors(id=3, vector=[0.0, 0.0, 4.0, 0.0]),
            ],
            update_filter=rest.Filter(must=[]),  # matches everything
        )

        vectors = _vectors_by_id(client, [1, 2, 3])
        # After normalization these remain directionally identical to the inputs.
        assert vectors[1] == pytest.approx([1.0, 0.0, 0.0, 0.0], abs=1e-6)
        assert vectors[2] == pytest.approx([0.0, 1.0, 0.0, 0.0], abs=1e-6)
        assert vectors[3] == pytest.approx([0.0, 0.0, 1.0, 0.0], abs=1e-6)

    def test_no_points_match(self) -> None:
        client = _seeded_client()
        before = _vectors_by_id(client, [1, 2, 3])

        client.update_vectors(
            "t",
            points=[
                rest.PointVectors(id=1, vector=[9.0, 0.0, 0.0, 0.0]),
                rest.PointVectors(id=2, vector=[0.0, 9.0, 0.0, 0.0]),
                rest.PointVectors(id=3, vector=[0.0, 0.0, 9.0, 0.0]),
            ],
            update_filter=rest.Filter(
                must=[rest.FieldCondition(key="type", match=rest.MatchValue(value="z"))],
            ),
        )

        vectors = _vectors_by_id(client, [1, 2, 3])
        for pid, expected in before.items():
            assert vectors[pid] == pytest.approx(expected, abs=1e-6), (
                f"point {pid} must be untouched when the filter rejects everything"
            )


@pytest.mark.asyncio
class TestUpdateVectorsFilterAsync:
    async def test_later_matching_point_is_not_skipped_after_mismatch(self) -> None:
        client = AsyncQdrantClient(":memory:")
        await client.create_collection(
            "t",
            vectors_config=rest.VectorParams(size=4, distance=rest.Distance.COSINE),
        )
        await client.upsert(
            "t",
            points=[
                rest.PointStruct(id=1, vector=[1.0, 0.0, 0.0, 0.0], payload={"type": "a"}),
                rest.PointStruct(id=2, vector=[0.0, 1.0, 0.0, 0.0], payload={"type": "b"}),
                rest.PointStruct(id=3, vector=[0.0, 0.0, 1.0, 0.0], payload={"type": "a"}),
            ],
            wait=True,
        )

        await client.update_vectors(
            "t",
            points=[
                rest.PointVectors(id=1, vector=[1.0, 1.0, 0.0, 0.0]),
                rest.PointVectors(id=2, vector=[0.0, 0.0, 1.0, 1.0]),
                rest.PointVectors(id=3, vector=[0.0, 0.0, 0.0, 1.0]),
            ],
            update_filter=_filter_match_type_a(),
        )

        retrieved = await client.retrieve("t", ids=[1, 2, 3], with_vectors=True)
        vectors = {p.id: list(p.vector) for p in retrieved}
        assert vectors[3] == pytest.approx([0.0, 0.0, 0.0, 1.0], abs=1e-6), (
            "async mirror must propagate the fix so point 3 still updates"
        )
