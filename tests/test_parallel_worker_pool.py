import os
import threading
from typing import Any, Iterable

import pytest

from qdrant_client.parallel_processor import ParallelWorkerPool, Worker


class _Echo(Worker):
    """A worker that appends a marker, so we can tell the pool really ran."""

    @classmethod
    def start(cls, *args: Any, **kwargs: Any) -> "_Echo":
        return cls(*args, **kwargs)

    def __init__(self, marker: str = "", **_kwargs: Any) -> None:
        self.marker = marker

    def process(self, items: Iterable[Any]) -> Iterable[Any]:
        for item in items:
            yield f"{item}{self.marker}"


def _drain(num_workers: int, timeout: float = 60.0) -> list[Any]:
    """Run a pool to completion, or report that it never finished."""
    results: list[Any] = []

    def runner() -> None:
        try:
            for item in ParallelWorkerPool(num_workers, _Echo).unordered_map(
                [[1], [2], [3]], marker="!"
            ):
                results.append(item)
        except Exception as exc:  # pragma: no cover - surfaced through the timeout check
            results.append(f"ERROR {type(exc).__name__}")

    thread = threading.Thread(target=runner, daemon=True)
    thread.start()
    thread.join(timeout=timeout)
    return results


def test_zero_workers_does_not_hang() -> None:
    """`parallel=0` used to deadlock instead of running.

    A pool with no workers never drains its input queue, so `unordered_map`
    blocked forever on `output_queue.get`. This runs the pool on a thread with a
    timeout so a regression fails the test instead of hanging CI.
    """
    results = _drain(0)

    assert results, "parallel=0 hung: the pool never produced a result"
    assert not any(isinstance(r, str) and r.startswith("ERROR") for r in results), (
        f"parallel=0 failed instead of running: {results}"
    )


def test_zero_workers_is_treated_as_all_cores() -> None:
    """0 means "every core", the same reading `ModelEmbedder` already applies."""
    pool = ParallelWorkerPool(0, _Echo)

    assert pool.num_workers == (os.cpu_count() or 1)
    assert pool.queue_size > 0, "a zero-size queue is what caused the deadlock"


def test_negative_workers_is_treated_as_all_cores() -> None:
    """A negative count cannot mean anything useful either."""
    assert ParallelWorkerPool(-3, _Echo).num_workers == (os.cpu_count() or 1)


@pytest.mark.parametrize("num_workers", [1, 2])
def test_positive_worker_counts_still_run(num_workers: int) -> None:
    """The guard must not disturb the normal path."""
    results = _drain(num_workers)

    assert sorted(results) == ["[1]!", "[2]!", "[3]!"]
