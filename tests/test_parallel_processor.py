import os
from typing import Any, Iterable

import pytest

from qdrant_client import parallel_processor
from qdrant_client.parallel_processor import ParallelWorkerPool, Worker


class CrashingWorker(Worker):
    @classmethod
    def start(cls, *args: Any, **kwargs: Any) -> "CrashingWorker":
        return cls()

    def process(self, items: Iterable[Any]) -> Iterable[Any]:
        for _ in items:
            os._exit(3)
            yield


def test_unordered_map_detects_dead_worker_while_draining(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(parallel_processor, "processing_timeout", 3)

    pool = ParallelWorkerPool(num_workers=1, worker=CrashingWorker)

    with pytest.raises(RuntimeError, match="terminated unexpectedly with code 3"):
        list(pool.unordered_map([1]))
