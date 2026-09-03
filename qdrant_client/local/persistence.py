import base64
import dbm
import logging
import pickle
import sqlite3
from pathlib import Path
from typing import Iterable

from qdrant_client.http import models

STORAGE_FILE_NAME_OLD = "storage.dbm"
STORAGE_FILE_NAME = "storage.sqlite"


def try_migrate_to_sqlite(location: str) -> None:
    dbm_path = Path(location) / STORAGE_FILE_NAME_OLD
    sql_path = Path(location) / STORAGE_FILE_NAME

    if sql_path.exists():
        return

    if not dbm_path.exists():
        return

    try:
        dbm_storage = dbm.open(str(dbm_path), "c")

        con = sqlite3.connect(str(sql_path))
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
        con.close()
        dbm_storage.close()
        dbm_path.unlink()
    except Exception as e:
        logging.error("Failed to migrate dbm to sqlite:", e)
        logging.error(
            "Please try to use previous version of qdrant-client or re-create collection"
        )
        raise e


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
            yield load_point_compat(row[0])


def load_point_compat(blob: bytes) -> models.PointStruct:
    """Unpickle a persisted point, including pydantic v1 -> v2 migrations.

    Points stored with pydantic<2 put ``__fields_set__`` in the pickle state.
    pydantic>=2 expects ``__pydantic_fields_set__`` and raises ``KeyError`` on
    load otherwise (https://github.com/qdrant/qdrant-client/issues/481).
    """
    try:
        point = pickle.loads(blob)
    except KeyError as exc:
        if exc.args != ("__pydantic_fields_set__",):
            raise
        point = _load_pydantic_v1_point(blob)

    if isinstance(point, models.PointStruct):
        return point
    if isinstance(point, dict):
        return models.PointStruct.model_validate(point)
    raise TypeError(f"Unexpected persisted point type: {type(point)!r}")


def _load_pydantic_v1_point(blob: bytes) -> models.PointStruct:
    """Best-effort recovery for PointStruct blobs pickled under pydantic v1."""
    import io

    class _LegacyPoint:
        def __setstate__(self, state):  # type: ignore[no-untyped-def]
            if not isinstance(state, dict):
                raise TypeError(f"Unexpected pickle state type: {type(state)!r}")
            payload = state.get("__dict__", state)
            if not isinstance(payload, dict):
                raise TypeError(f"Unexpected payload type: {type(payload)!r}")
            self.__dict__.update(payload)

    class _CompatUnpickler(pickle.Unpickler):
        def find_class(self, module: str, name: str):  # type: ignore[override]
            if name == "PointStruct" and "qdrant" in module:
                return _LegacyPoint
            return super().find_class(module, name)

    obj = _CompatUnpickler(io.BytesIO(blob)).load()
    data = getattr(obj, "__dict__", None)
    if not isinstance(data, dict):
        raise TypeError(f"Recovered point has no dict state: {type(obj)!r}")

    payload = {
        "id": data.get("id"),
        "vector": data.get("vector", data.get("vectors")),
        "payload": data.get("payload"),
    }
    return models.PointStruct.model_validate(
        {key: value for key, value in payload.items() if value is not None or key == "id"}
    )




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
