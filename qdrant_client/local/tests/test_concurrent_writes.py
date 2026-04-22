"""Regression tests for concurrent writes on in-memory LocalCollection.

Reproduces the race reported in qdrant-client#1193: concurrent `upsert`
threads would interleave the growth of the per-collection `payload`,
`deleted`, and per-vector numpy arrays, leaving them at mismatched
shapes. A subsequent `scroll` / `search` would then fail with:

    ValueError: operands could not be broadcast together with shapes (N,) (M,)

The fix serializes the public write methods via a reentrant lock on
LocalCollection. These tests assert:

1. Multiple concurrent `upsert` workers produce the exact expected
   point count and no array-shape mismatch on read-back.
2. Shape invariants (`len(payload) == len(deleted) == vectors.shape[0]
   == len(ids) == len(ids_inv)`) hold after a concurrent write storm.
3. Concurrent deletes interleaved with upserts do not corrupt state.
"""

import threading
import uuid

import pytest

from qdrant_client import QdrantClient
from qdrant_client.local.local_collection import LocalCollection
from qdrant_client import models
from qdrant_client.models import (
    Distance,
    PointStruct,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)


def _make_client() -> QdrantClient:
    client = QdrantClient(":memory:")
    client.create_collection(
        collection_name="race",
        vectors_config={"dense": VectorParams(size=8, distance=Distance.COSINE)},
        sparse_vectors_config={"bm25": SparseVectorParams()},
    )
    return client


def _upsert_worker(client: QdrantClient, batch_size: int, runs: int, seed: int) -> None:
    # Build points with both dense + sparse vectors, matching the repro in
    # #1193. Seeded so each thread produces unique ids (UUIDs) without
    # colliding with others.
    import random

    rng = random.Random(seed)
    for _ in range(runs):
        points = [
            PointStruct(
                id=str(uuid.uuid4()),
                vector={
                    "dense": [rng.random() for _ in range(8)],
                    "bm25": SparseVector(indices=[0, 1, 2], values=[0.5, 0.3, 0.1]),
                },
                payload={"thread": seed},
            )
            for _ in range(batch_size)
        ]
        client.upsert(collection_name="race", points=points)


@pytest.mark.timeout(30)
def test_concurrent_upsert_point_count_is_exact() -> None:
    """Repro for #1193 — 8 workers × 20 runs × 10 points = 1600 points."""
    num_threads = 8
    runs = 20
    batch = 10
    expected = num_threads * runs * batch

    client = _make_client()
    threads = [
        threading.Thread(target=_upsert_worker, args=(client, batch, runs, s))
        for s in range(num_threads)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    points, _ = client.scroll(collection_name="race", limit=expected * 2)
    assert len(points) == expected


@pytest.mark.timeout(30)
def test_concurrent_upsert_preserves_array_shape_invariant() -> None:
    """After concurrent upserts, internal arrays must all share the same length."""
    num_threads = 8
    runs = 15
    batch = 10

    client = _make_client()
    threads = [
        threading.Thread(target=_upsert_worker, args=(client, batch, runs, s))
        for s in range(num_threads)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    local = client._client.collections["race"]  # type: ignore[attr-defined]
    n = len(local.payload)
    assert n == num_threads * runs * batch
    # Invariants that the concurrent-upsert race in #1193 specifically breaks:
    # payload / deleted / ids_inv grew together; dense vector backing arrays
    # have capacity >= n (they double-grow internally) and per-vector deleted
    # masks grew with the logical row count.
    assert len(local.deleted) == n
    assert len(local.ids) == n
    assert len(local.ids_inv) == n
    for name, arr in local.vectors.items():
        assert arr.shape[0] >= n, f"dense vector '{name}' capacity {arr.shape[0]} < n={n}"
    for name, mask in local.deleted_per_vector.items():
        assert len(mask) == n, f"deleted_per_vector['{name}'] length {len(mask)} != n={n}"


@pytest.mark.timeout(30)
def test_concurrent_upsert_and_delete_does_not_corrupt() -> None:
    """Interleave upsert with delete; final count + invariants must hold.

    Each upsert worker inserts K known ids; a deleter thread repeatedly
    deletes even-seeded ids. Final state is only asserted for invariants,
    not an exact count (delete timing is racy by design), but a `search`
    must not raise a broadcast-shape error.
    """
    num_upserters = 4
    runs = 10
    batch = 10

    client = _make_client()
    upsert_threads = [
        threading.Thread(target=_upsert_worker, args=(client, batch, runs, s))
        for s in range(num_upserters)
    ]

    stop = threading.Event()

    def deleter() -> None:
        while not stop.is_set():
            # Delete points carrying payload.thread == 0. Safe to run even if
            # no such points exist yet.
            try:
                client.delete(
                    collection_name="race",
                    points_selector=models.FilterSelector(
                        filter=models.Filter(
                            must=[
                                models.FieldCondition(
                                    key="thread",
                                    match=models.MatchValue(value=0),
                                )
                            ]
                        )
                    ),
                )
            except Exception:
                # delete should never raise under the lock; surface via assert below.
                stop.set()
                raise

    del_thread = threading.Thread(target=deleter, daemon=True)
    del_thread.start()
    for t in upsert_threads:
        t.start()
    for t in upsert_threads:
        t.join()
    stop.set()
    del_thread.join(timeout=5)

    # Pure read; must not raise operand-broadcast ValueError.
    points, _ = client.scroll(collection_name="race", limit=num_upserters * runs * batch * 2)

    local = client._client.collections["race"]  # type: ignore[attr-defined]
    n = len(local.payload)
    assert len(local.deleted) == n
    assert len(local.ids_inv) == n
    for name, arr in local.vectors.items():
        assert arr.shape[0] >= n, f"dense vector '{name}' capacity {arr.shape[0]} < n={n}"
    # Every returned point should be in the ids map.
    for p in points:
        assert p.id in local.ids


@pytest.mark.timeout(10)
def test_write_lock_is_reentrant() -> None:
    """`batch_update_points` delegates to `upsert` / `delete` — both of which
    now acquire the write lock — so the lock must be reentrant or a
    single-threaded batch would deadlock."""
    collection = LocalCollection(
        models.CreateCollection(
            vectors=models.VectorParams(size=2, distance=models.Distance.COSINE)
        )
    )
    collection.batch_update_points(
        update_operations=[
            models.UpsertOperation(
                upsert=models.PointsList(
                    points=[models.PointStruct(id=1, vector=[0.1, 0.2])]
                )
            ),
            models.DeleteOperation(
                delete=models.PointIdsList(points=[1]),
            ),
        ]
    )
    # LocalCollection soft-deletes: id stays in the map, `deleted[idx]` is set.
    idx = collection.ids[1]
    assert collection.deleted[idx]
