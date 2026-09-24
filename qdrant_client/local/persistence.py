import base64
import dbm
import logging
import os
import pickle
import sqlite3
import tempfile
from pathlib import Path
from typing import Iterable

from qdrant_client.http import models

STORAGE_FILE_NAME_OLD = "storage.dbm"
STORAGE_FILE_NAME = "storage.sqlite"

_DBM_FILE_SUFFIXES = {
    "dbm.dumb": (".dat", ".dir", ".bak"),
    "dbm.gnu": ("",),
    "dbm.sqlite3": ("",),
}


def _dbm_file_suffixes(dbm_path: Path, backend: str) -> tuple[str, ...]:
    """Return sidecar suffixes owned by the detected DBM backend.

    Only files belonging to the detected backend may be removed after
    migration. Unknown backends return an empty tuple so unrelated files
    are preserved. Ambiguous NDBM layouts are also preserved.
    """
    if backend != "dbm.ndbm":
        return _DBM_FILE_SUFFIXES.get(backend, ())

    layouts = [
        suffixes
        for suffixes in ((".db",), (".dir", ".pag"))
        if all(dbm_path.with_name(f"{dbm_path.name}{suffix}").is_file() for suffix in suffixes)
    ]
    return layouts[0] if len(layouts) == 1 else ()


def _has_conflicting_dbm_sidecars(dbm_path: Path) -> bool:
    """Detect a dumb/NDBM sidecar conflict that whichdb misclassifies.

    whichdb checks `.dir`+`.pag` (NDBM) before `.dat`+`.dir` (dumb), so
    `.dat`+`.dir`+`.pag` is reported as `dbm.ndbm` even though a dumb
    store is present. Migrating as NDBM could delete `.dir`/`.pag` and
    orphan `.dat`. Return True so the caller raises and preserves everything
    without creating an empty storage.sqlite.
    """
    return all(
        dbm_path.with_name(f"{dbm_path.name}{suffix}").is_file()
        for suffix in (".dat", ".dir", ".pag")
    )


def try_migrate_to_sqlite(location: str) -> None:
    """Migrate legacy DBM storage to SQLite, if present.

    Detection is backend-aware via dbm.whichdb so sidecar backends
    (ndbm .db, dumb .dat/.dir/.bak) are not skipped. Cleanup removes
    only sidecars owned by the detected backend after the SQLite
    commit succeeds, preserving unrelated files. Conflicting sidecar
    layouts raise without creating storage.sqlite so a later retry
    cannot be blocked by an empty database. Migration copies into a
    private temp file and publishes atomically, so a failed migration
    never deletes another instance's database and stays retryable.
    """
    dbm_path = Path(location) / STORAGE_FILE_NAME_OLD
    sql_path = Path(location) / STORAGE_FILE_NAME

    if sql_path.exists():
        return

    if _has_conflicting_dbm_sidecars(dbm_path):
        raise RuntimeError(
            f"Conflicting DBM sidecar files for {dbm_path} (.dat/.dir/.pag): "
            "whichdb misreports this layout as ndbm. Resolve manually before migration."
        )

    backend = dbm.whichdb(str(dbm_path))
    if not backend:
        return

    con: sqlite3.Connection | None = None
    dbm_storage = None
    tmp_sql_path: Path | None = None
    migration_succeeded = False
    try:
        fd, tmp_name = tempfile.mkstemp(
            dir=str(Path(location)),
            prefix=f"{STORAGE_FILE_NAME}.migrating-",
            suffix=".tmp",
        )
        os.close(fd)
        tmp_sql_path = Path(tmp_name)

        dbm_storage = dbm.open(str(dbm_path), "r")

        con = sqlite3.connect(str(tmp_sql_path))
        cur = con.cursor()

        # Create table
        cur.execute("CREATE TABLE IF NOT EXISTS points (id TEXT PRIMARY KEY, point BLOB)")

        for key in dbm_storage.keys():
            value = dbm_storage[key]
            if isinstance(key, str):
                key = key.encode("utf-8")
            key = pickle.loads(key)
            sqlite_key = CollectionPersistence.encode_key(key)
            # Insert a row of data
            cur.execute(
                "INSERT INTO points VALUES (?, ?)",
                (
                    sqlite_key,
                    sqlite3.Binary(value),
                ),
            )
        con.commit()
        migration_succeeded = True
    except Exception as e:
        logging.error("Failed to migrate dbm to sqlite: %s", e)
        logging.error(
            "Please try to use previous version of qdrant-client or re-create collection"
        )
        raise
    finally:
        if dbm_storage is not None:
            try:
                dbm_storage.close()
            except Exception:
                pass
        if con is not None:
            try:
                con.close()
            except Exception:
                pass
        if not migration_succeeded and tmp_sql_path is not None and tmp_sql_path.is_file():
            try:
                tmp_sql_path.unlink()
            except OSError:
                pass

    if migration_succeeded and tmp_sql_path is not None:
        if sql_path.exists():
            # Another instance published first; discard ours without touching it.
            try:
                tmp_sql_path.unlink()
            except OSError:
                pass
            return
        try:
            tmp_sql_path.replace(sql_path)
        except OSError:
            try:
                tmp_sql_path.unlink()
            except OSError:
                pass
            raise
        for suffix in _dbm_file_suffixes(dbm_path, backend):
            sidecar = dbm_path.with_name(dbm_path.name + suffix)
            if sidecar.is_file():
                sidecar.unlink()


class CollectionPersistence:
    CHECK_SAME_THREAD: bool | None = None

    @classmethod
    def encode_key(cls, key: models.ExtendedPointId) -> str:
        return base64.b64encode(pickle.dumps(key)).decode("utf-8")

    def __init__(self, location: str, force_disable_check_same_thread: bool = False):
        """
        Create or load a collection from the local storage.
        Args:
            location: path to the collection directory.
        """

        try_migrate_to_sqlite(location)

        self.location = Path(location) / STORAGE_FILE_NAME
        self.location.parent.mkdir(exist_ok=True, parents=True)

        if self.CHECK_SAME_THREAD is None and force_disable_check_same_thread is False:
            with sqlite3.connect(":memory:") as tmp_conn:
                # it is unsafe to use `sqlite3.threadsafety` until python3.11 since it was hardcoded to 1, thus we
                # need to fetch threadsafe with a query
                # THREADSAFE = 0: Threads may not share the module
                # THREADSAFE = 1: Threads may share the module, connections and cursors. Default for Linux.
                # THREADSAFE = 2: Threads may share the module, but not connections. Default for macOS.
                threadsafe = tmp_conn.execute(
                    "select * from pragma_compile_options where compile_options like 'THREADSAFE=%'"
                ).fetchone()[0]
                self.__class__.CHECK_SAME_THREAD = threadsafe != "THREADSAFE=1"

        if force_disable_check_same_thread:
            self.__class__.CHECK_SAME_THREAD = False

        self.storage = sqlite3.connect(
            str(self.location),
            check_same_thread=self.CHECK_SAME_THREAD,  # type: ignore
        )

        self._ensure_table()

    def close(self) -> None:
        self.storage.close()

    def _ensure_table(self) -> None:
        cursor = self.storage.cursor()
        cursor.execute("CREATE TABLE IF NOT EXISTS points (id TEXT PRIMARY KEY, point BLOB)")
        self.storage.commit()

    def persist(self, point: models.PointStruct) -> None:
        """
        Persist a point in the local storage.
        Args:
            point: point to persist
        """
        key = self.encode_key(point.id)
        value = pickle.dumps(point)

        cursor = self.storage.cursor()
        # Insert or update by key
        cursor.execute(
            "INSERT OR REPLACE INTO points VALUES (?, ?)",
            (
                key,
                sqlite3.Binary(value),
            ),
        )

        self.storage.commit()

    def delete(self, point_id: models.ExtendedPointId) -> None:
        """
        Delete a point from the local storage.
        Args:
            point_id: id of the point to delete
        """
        key = self.encode_key(point_id)
        cursor = self.storage.cursor()
        cursor.execute(
            "DELETE FROM points WHERE id = ?",
            (key,),
        )
        self.storage.commit()

    def load(self) -> Iterable[models.PointStruct]:
        """
        Load a point from the local storage.
        Returns:
            point: loaded point
        """
        cursor = self.storage.cursor()
        cursor.execute("SELECT point FROM points")
        for row in cursor.fetchall():
            yield pickle.loads(row[0])


def test_persistence() -> None:
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        persistence = CollectionPersistence(tmpdir)
        point = models.PointStruct(id=1, vector=[1.0, 2.0, 3.0], payload={"a": 1})
        persistence.persist(point)
        for loaded_point in persistence.load():
            assert loaded_point == point
            break

        del persistence
        persistence = CollectionPersistence(tmpdir)
        for loaded_point in persistence.load():
            assert loaded_point == point
            break

        persistence.delete(point.id)
        persistence.delete(point.id)
        for _ in persistence.load():
            assert False, "Should not load anything"
