"""Tests for the optional progress_callback parameter on
QdrantRemote._upload_collection and the public QdrantClient uploaders.

The progress callback only fires when the underlying client is a
QdrantRemote (i.e. when uploads go through _upload_collection).
QdrantLocal handles uploads directly and does not invoke the callback.
"""

from typing import Any, Iterable

import numpy as np
import pytest

from qdrant_client import QdrantClient, models
from qdrant_client.qdrant_remote import QdrantRemote
from qdrant_client.uploader.uploader import UploadProgress


def _make_remote_client(prefer_grpc: bool = False) -> QdrantClient:
    """Construct a QdrantClient whose _client is a QdrantRemote aimed at
    a non-existent host. Used to exercise the callback plumbing without
    actually sending any network traffic.
    """
    client = QdrantClient(
        host="127.0.0.1",
        port=1,
        grpc_port=1,
        prefer_grpc=prefer_grpc,
        check_compatibility=False,
    )
    client._client = QdrantRemote(host="127.0.0.1", port=1, prefer_grpc=prefer_grpc)
    return client


class _FakeUploaderCls:
    """Stateful stand-in for GrpcBatchUploader / RestBatchUploader.

    The class is patched in for QdrantRemote._updater_class via
    monkeypatch. Each test stages a `results` queue; the queue is
    consumed (in order) by every batch the fake uploader emits.
    """

    results: list[bool] = []

    @classmethod
    def start(cls, **_kwargs: Any) -> "_FakeUploaderInst":
        return _FakeUploaderInst()


class _FakeUploaderInst:
    def process(self, items: Iterable[Any]) -> Iterable[bool]:
        # Yield the staged results in order; if more items than staged,
        # default to True so callers don't see surprising drops.
        staged = list(_FakeUploaderCls.results) or [True]
        yield from staged


class _FakePool:
    def __init__(self, num_workers, worker_class, start_method):
        pass

    def unordered_map(self, stream, **_kwargs):
        updater_cls = _FakeUploaderCls
        updater = updater_cls.start()
        for result in updater.process(stream):
            yield result


def _patch_uploader(monkeypatch, results: list[bool]):
    """Replace QdrantRemote._updater_class with _FakeUploaderCls and
    optionally swap in _FakePool for parallel mode.
    """
    _FakeUploaderCls.results = list(results)
    # _updater_class is a @property: assign a new property to the class.
    monkeypatch.setattr(
        QdrantRemote,
        "_updater_class",
        property(lambda _self: _FakeUploaderCls),
    )


def _make_batches(sizes: list[int]) -> list[tuple]:
    """Build batches of the (ids_batch, vectors_batch, payload_batch)
    shape consumed by _upload_collection.
    """
    batches: list[tuple] = []
    for size in sizes:
        batches.append(
            (
                list(range(size)),
                [[float(i)] * 4 for i in range(size)],
                [{} for _ in range(size)],
            )
        )
    return batches


def test_upload_progress_dataclass_is_frozen_and_hashable():
    p = UploadProgress(total_uploaded=10, batch_count=1)
    assert p.total_uploaded == 10
    assert p.batch_count == 1
    assert hash(p)  # frozen + slots => hashable
    with pytest.raises(Exception):
        p.total_uploaded = 5  # type: ignore[misc]


def test_progress_callback_default_none_does_not_break_signature():
    client = _make_remote_client()
    client.upload_points(
        collection_name="noop",
        points=[],
        progress_callback=None,
    )


def test_progress_callback_fires_once_per_batch(monkeypatch):
    _patch_uploader(monkeypatch, [True, True, True, True])
    client = _make_remote_client()
    batches = _make_batches([64, 64, 64, 58])  # 4 batches

    progress: list[UploadProgress] = []
    client._client._upload_collection(
        batches_iterator=batches,
        collection_name="test",
        max_retries=1,
        progress_callback=lambda p: progress.append(p),
    )

    assert len(progress) == 4, f"expected 4 callbacks, got {len(progress)}"
    assert [p.batch_count for p in progress] == [1, 2, 3, 4]
    assert [p.total_uploaded for p in progress] == [64, 128, 192, 250]


def test_progress_callback_skips_failed_batches(monkeypatch):
    _patch_uploader(monkeypatch, [True, False, True, True])
    client = _make_remote_client()
    batches = _make_batches([10, 10, 10, 10])

    progress: list[UploadProgress] = []
    client._client._upload_collection(
        batches_iterator=batches,
        collection_name="test",
        max_retries=1,
        progress_callback=lambda p: progress.append(p),
    )

    assert [p.batch_count for p in progress] == [1, 2, 3]
    assert [p.total_uploaded for p in progress] == [10, 20, 30]  # failed batch at idx 1 is skipped


def test_progress_callback_parallel_path(monkeypatch):
    _patch_uploader(monkeypatch, [True, True, True])
    monkeypatch.setattr(
        "qdrant_client.qdrant_remote.ParallelWorkerPool", _FakePool
    )
    client = _make_remote_client(prefer_grpc=True)
    batches = _make_batches([5, 5, 5])

    progress: list[UploadProgress] = []
    client._client._upload_collection(
        batches_iterator=batches,
        collection_name="test",
        max_retries=1,
        parallel=2,
        progress_callback=lambda p: progress.append(p),
    )

    assert len(progress) == 3
    assert [p.batch_count for p in progress] == [1, 2, 3]
    assert [p.total_uploaded for p in progress] == [5, 10, 15]


def test_qdrant_client_local_mode_accepts_kwarg_without_crash():
    """Local mode does not invoke the callback (it doesn't go through
    _upload_collection) but must accept the kwarg without a TypeError.
    """
    client = QdrantClient(":memory:")
    received: list[UploadProgress] = []

    def cb(p: UploadProgress) -> None:
        received.append(p)

    client.create_collection(
        collection_name="noop",
        vectors_config=models.VectorParams(size=4, distance=models.Distance.COSINE),
    )
    client.upload_points(
        collection_name="noop",
        points=[
            models.PointStruct(id=i, vector=[0.1, 0.2, 0.3, 0.4])
            for i in range(3)
        ],
        batch_size=2,
        progress_callback=cb,
    )

    assert received == []  # local mode does not invoke the callback


def test_signature_accepts_progress_callback_kwarg():
    """Both upload_points and upload_collection on the public facade must
    accept progress_callback as a keyword argument.
    """
    client = _make_remote_client()
    client.upload_points(
        collection_name="t",
        points=[],
        progress_callback=lambda p: None,
    )
    client.upload_collection(
        collection_name="t",
        vectors=np.zeros((0, 4), dtype=np.float32),
        progress_callback=lambda p: None,
    )
