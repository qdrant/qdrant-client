import sqlite3
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from qdrant_client.local.persistence import CollectionPersistence


def read_storage(storage: CollectionPersistence) -> None:
    storage.storage.execute("SELECT COUNT(*) FROM points").fetchone()


@pytest.mark.parametrize("cached_check", [True, None])
def test_thread_check_override_is_per_connection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cached_check: bool | None
) -> None:
    # Simulate SQLite's multi-thread mode, where connections must not be shared.
    monkeypatch.setattr(sqlite3, "threadsafety", 1)
    monkeypatch.setattr(CollectionPersistence, "CHECK_SAME_THREAD", cached_check)

    overridden = CollectionPersistence(
        str(tmp_path / "overridden"), force_disable_check_same_thread=True
    )
    default = CollectionPersistence(str(tmp_path / "default"))
    try:
        with ThreadPoolExecutor(max_workers=1) as worker:
            worker.submit(read_storage, overridden).result()
            with pytest.raises(sqlite3.ProgrammingError, match="same thread"):
                worker.submit(read_storage, default).result()
    finally:
        default.close()
        overridden.close()
