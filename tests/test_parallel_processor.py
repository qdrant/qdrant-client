from collections import deque
from multiprocessing import get_all_start_methods
from queue import Empty
from unittest.mock import MagicMock

import pytest

from qdrant_client.parallel_processor import ParallelWorkerPool, QueueSignals, Worker


class EchoWorker(Worker):
    @classmethod
    def start(cls, **kwargs):
        return cls()

    def process(self, items):
        yield from items


@pytest.mark.parametrize("window", [4, 8, 64])
def test_ordered_map_bounds_results_behind_slow_first_batch(window):
    # Complete every batch immediately except batch zero. Only a blocking read
    # completes zero, simulating a slow embedding request without timing sleeps.
    pool = ParallelWorkerPool(1, EchoWorker, max_internal_batch_size=window)
    ready = deque()
    first = []
    submitted = []
    submitted_when_first_completed = []
    pool.input_queue = MagicMock()
    pool.output_queue = MagicMock()
    pool.start = MagicMock()
    pool.check_worker_health = MagicMock()

    def put(item, **kwargs):
        if item == QueueSignals.stop:
            return
        submitted.append(item)
        if item[0] == 0:
            first.append(item)
        else:
            ready.append(item)

    def get_nowait():
        if ready:
            return ready.popleft()
        raise Empty

    def blocking_read():
        if first:
            submitted_when_first_completed.append(len(submitted))
            return first.pop()
        return get_nowait()

    pool.input_queue.put.side_effect = put
    pool.output_queue.get_nowait.side_effect = get_nowait
    pool._get_output = blocking_read
    assert list(pool.ordered_map(range(100))) == list(range(100))
    assert submitted_when_first_completed == [window]


@pytest.mark.timeout(30)
@pytest.mark.parametrize("method", get_all_start_methods())
@pytest.mark.parametrize("mode", ["ordered_map", "semi_ordered_map", "unordered_map"])
def test_pool_maps_complete_with_small_window(method, mode):
    pool = ParallelWorkerPool(2, EchoWorker, start_method=method, max_internal_batch_size=1)
    result = list(getattr(pool, mode)(range(30)))
    if mode == "ordered_map":
        assert result == list(range(30))
    elif mode == "semi_ordered_map":
        assert sorted(result) == list(enumerate(range(30)))
    else:
        assert sorted(result) == list(range(30))
    assert not pool.processes


class GatedWorker(Worker):
    @classmethod
    def start(cls, release, later_completed):
        worker = cls()
        worker.release = release
        worker.later_completed = later_completed
        return worker

    def process(self, items):
        for idx, item in items:
            if idx == 0:
                assert self.release.wait(10), "test did not release first batch"
            elif idx == 7:
                self.later_completed.set()
            yield idx, item


@pytest.mark.timeout(30)
def test_real_workers_do_not_exhaust_lazy_input_behind_slow_first_batch():
    from concurrent.futures import ThreadPoolExecutor, TimeoutError

    pool = ParallelWorkerPool(2, GatedWorker, start_method="spawn", max_internal_batch_size=4)
    release = pool.ctx.Event()
    later_completed = pool.ctx.Event()

    def stream():
        for idx in range(100):
            # The map reads one item before making room for it in its window.
            assert idx <= pool.queue_size or release.is_set(), "unbounded input read-ahead"
            yield idx

    with ThreadPoolExecutor(1) as executor:
        future = executor.submit(
            lambda: list(
                pool.ordered_map(stream(), release=release, later_completed=later_completed)
            )
        )
        try:
            assert later_completed.wait(10)
            with pytest.raises(TimeoutError):
                future.result(timeout=0.2)
        finally:
            release.set()
        assert future.result(timeout=10) == list(range(100))
    assert not pool.processes


@pytest.mark.timeout(30)
def test_ordered_map_closes_workers_when_consumer_stops():
    pool = ParallelWorkerPool(2, EchoWorker, start_method="spawn", max_internal_batch_size=2)
    result = pool.ordered_map(range(100))
    assert next(result) == 0
    processes = list(pool.processes)
    result.close()
    assert not pool.processes
    assert all(not process.is_alive() for process in processes)


class FailingWorker(EchoWorker):
    def process(self, items):
        for idx, item in items:
            if idx == 3:
                raise ValueError("test worker failure")
            yield idx, item


@pytest.mark.timeout(30)
def test_ordered_map_cleans_up_after_worker_failure():
    pool = ParallelWorkerPool(2, FailingWorker, start_method="spawn", max_internal_batch_size=2)
    with pytest.raises(RuntimeError, match="unexpectedly terminated"):
        list(pool.ordered_map(range(100)))
    assert not pool.processes


@pytest.mark.timeout(30)
def test_ordered_map_cleans_up_after_input_failure():
    def stream():
        yield 0
        raise ValueError("test input failure")

    pool = ParallelWorkerPool(2, EchoWorker, start_method="spawn", max_internal_batch_size=2)
    with pytest.raises(ValueError, match="test input failure"):
        list(pool.ordered_map(stream()))
    assert not pool.processes
