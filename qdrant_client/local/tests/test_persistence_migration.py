"""Legacy DBM migration must not skip backends that use sidecar files.

`dbm.dumb` creates `storage.dbm.dat`/`.dir` (no unsuffixed file), same class
as macOS `dbm.ndbm` (`storage.dbm.db`). Detection via `Path.exists()` misses
both, so points disappear after upgrade. See #1456.
"""

import dbm
import dbm.dumb
import pickle
from pathlib import Path

import pytest

from qdrant_client.http import models
from qdrant_client.local.persistence import (
    CollectionPersistence,
    STORAGE_FILE_NAME_OLD,
    try_migrate_to_sqlite,
)


def test_migrate_dumb_dbm_sidecars(tmp_path: Path) -> None:
    """Dumb DBM sidecars are detected and migrated to SQLite."""
    location = tmp_path / "legacy"
    location.mkdir()
    dbm_path = location / STORAGE_FILE_NAME_OLD

    point = models.PointStruct(id=1, vector=[1.0, 2.0], payload={"source": "v1.1.4"})
    db = dbm.dumb.open(str(dbm_path), "c")
    db[pickle.dumps(point.id)] = pickle.dumps(point)
    db.close()

    try_migrate_to_sqlite(str(location))

    persistence = CollectionPersistence(str(location))
    loaded = list(persistence.load())
    assert loaded == [point]
    persistence.close()


def test_migrates_dumb_dbm_without_removing_other_backend_files(tmp_path: Path) -> None:
    """Dumb DBM migration preserves files owned by other backends."""
    location = tmp_path / "legacy"
    location.mkdir()
    dbm_path = location / STORAGE_FILE_NAME_OLD
    point = models.PointStruct(id=1, vector=[1.0, 2.0], payload={"source": "legacy"})

    with dbm.dumb.open(str(dbm_path), "c") as storage:
        storage[pickle.dumps(point.id)] = pickle.dumps(point)

    unrelated_file = location / f"{STORAGE_FILE_NAME_OLD}.db"
    unrelated_file.write_text("not part of the dumb DBM store")
    assert dbm.whichdb(str(dbm_path)) == "dbm.dumb"

    persistence = CollectionPersistence(str(location))

    assert list(persistence.load()) == [point]
    assert unrelated_file.read_text() == "not part of the dumb DBM store"
    assert not any(
        location.joinpath(f"{STORAGE_FILE_NAME_OLD}{suffix}").exists()
        for suffix in (".dat", ".dir", ".bak")
    )
    persistence.close()


def test_migrates_ndbm_without_removing_unsuffixed_file(tmp_path: Path) -> None:
    """NDBM migration preserves an unrelated unsuffixed file."""
    ndbm = pytest.importorskip("dbm.ndbm")
    location = tmp_path / "legacy"
    location.mkdir()
    dbm_path = location / STORAGE_FILE_NAME_OLD
    point = models.PointStruct(id=1, vector=[1.0, 2.0], payload={"source": "legacy"})

    with ndbm.open(str(dbm_path), "c") as storage:
        storage[pickle.dumps(point.id)] = pickle.dumps(point)

    backend_files = list(location.iterdir())
    dbm_path.write_text("not part of the NDBM store")
    assert dbm.whichdb(str(dbm_path)) == "dbm.ndbm"

    persistence = CollectionPersistence(str(location))

    assert list(persistence.load()) == [point]
    assert dbm_path.read_text() == "not part of the NDBM store"
    assert not any(path.exists() for path in backend_files)
    persistence.close()


def test_preserves_ambiguous_ndbm_sidecars(tmp_path: Path) -> None:
    """Ambiguous NDBM layouts are preserved conservatively."""
    ndbm = pytest.importorskip("dbm.ndbm")
    location = tmp_path / "legacy"
    location.mkdir()
    dbm_path = location / STORAGE_FILE_NAME_OLD
    point = models.PointStruct(id=1, vector=[1.0, 2.0], payload={"source": "legacy"})

    with ndbm.open(str(dbm_path), "c") as storage:
        storage[pickle.dumps(point.id)] = pickle.dumps(point)

    backend_files = list(location.iterdir())
    backend_suffixes = {path.suffix for path in backend_files}
    if ".db" in backend_suffixes:
        unrelated_files = [
            location / f"{STORAGE_FILE_NAME_OLD}.dir",
            location / f"{STORAGE_FILE_NAME_OLD}.pag",
        ]
    elif {".dir", ".pag"} <= backend_suffixes:
        unrelated_files = [location / f"{STORAGE_FILE_NAME_OLD}.db"]
    else:
        pytest.skip(f"Unsupported NDBM layout: {backend_suffixes}")

    for path in unrelated_files:
        path.write_text("not part of the active NDBM store")
    assert dbm.whichdb(str(dbm_path)) == "dbm.ndbm"

    persistence = CollectionPersistence(str(location))

    assert list(persistence.load()) == [point]
    assert all(path.exists() for path in backend_files)
    assert all(path.read_text() == "not part of the active NDBM store" for path in unrelated_files)
    persistence.close()


def test_ignores_unrecognized_legacy_file(tmp_path: Path) -> None:
    """Unrecognized legacy files are left alone and not migrated."""
    legacy_file = tmp_path / STORAGE_FILE_NAME_OLD
    legacy_file.write_text("not a DBM store")
    assert dbm.whichdb(str(legacy_file)) == ""

    persistence = CollectionPersistence(str(tmp_path))

    assert list(persistence.load()) == []
    assert legacy_file.read_text() == "not a DBM store"
    persistence.close()
