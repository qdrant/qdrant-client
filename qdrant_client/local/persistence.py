import base64
import dbm
import pickle
import sqlite3
from pathlib import Path
from typing import Iterable

from qdrant_client.http import models
from qdrant_client._pydantic_compat import construct
from qdrant_client.common.client_warnings import show_warning
from qdrant_client.local.utils import normalize_point_id

STORAGE_FILE_NAME_OLD = "storage.dbm"
STORAGE_FILE_NAME = "storage.sqlite"


def check_legacy_storage(location: str) -> None:
    dbm_path = Path(location) / STORAGE_FILE_NAME_OLD
    sql_path = Path(location) / STORAGE_FILE_NAME

    if not sql_path.exists() and (dbm_path.exists() or dbm.whichdb(str(dbm_path))):
        raise ValueError(
            f"Collection at {location} uses the legacy dbm storage format (qdrant-client < 1.1.7), "
            "which is no longer supported. Open it once with an older version of qdrant-client to "
            "migrate it to sqlite, or re-create the collection."
        )


class CollectionPersistence:
    CHECK_SAME_THREAD: bool | None = None
    # pinned, since the default protocol changed from 4 to 5 in Python 3.14, and keys with it
    KEY_PICKLE_PROTOCOL = 4

    @classmethod
    def encode_key(cls, key: models.ExtendedPointId) -> str:
        return base64.b64encode(pickle.dumps(key, protocol=cls.KEY_PICKLE_PROTOCOL)).decode(
            "utf-8"
        )

    def __init__(self, location: str, force_disable_check_same_thread: bool = False):
        """
        Create or load a collection from the local storage.
        Args:
            location: path to the collection directory.
        """

        check_legacy_storage(location)

        self.location = Path(location) / STORAGE_FILE_NAME
        self.location.parent.mkdir(exist_ok=True, parents=True)

        if self.CHECK_SAME_THREAD is None and force_disable_check_same_thread is False:
            # `sqlite3.threadsafety` maps SQLite's THREADSAFE compile option to a DB-API level:
            # THREADSAFE = 0 -> 0: Threads may not share the module
            # THREADSAFE = 1 -> 3: Threads may share the module, connections and cursors. Linux default.
            # THREADSAFE = 2 -> 1: Threads may share the module, but not connections. macOS default.
            self.__class__.CHECK_SAME_THREAD = sqlite3.threadsafety != 3

        self.storage = sqlite3.connect(
            str(self.location),
            check_same_thread=False if force_disable_check_same_thread else self.CHECK_SAME_THREAD,  # type: ignore
        )

        self._ensure_table()
        # stored keys which differ from `encode_key(point_id)`
        self._legacy_keys: dict[models.ExtendedPointId, str] = {}

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
        point_id = normalize_point_id(point.id)
        point = construct(
            models.PointStruct, id=point_id, vector=point.vector, payload=point.payload
        )
        key = self.encode_key(point_id)
        value = pickle.dumps(point)

        # The connection commits on leaving the block, and rolls back if the commit fails,
        # so a failed write can't be committed later along with another one.
        with self.storage:
            legacy_key = self._legacy_keys.get(point_id)
            if legacy_key is not None:
                self.storage.execute("DELETE FROM points WHERE id = ?", (legacy_key,))
            # Insert or update by key
            self.storage.execute(
                "INSERT OR REPLACE INTO points VALUES (?, ?)", (key, sqlite3.Binary(value))
            )
        self._legacy_keys.pop(point_id, None)

    def delete(self, point_id: models.ExtendedPointId) -> None:
        """
        Delete a point from the local storage.
        Args:
            point_id: id of the point to delete
        """
        point_id = normalize_point_id(point_id)
        key = self._legacy_keys.get(point_id, self.encode_key(point_id))
        with self.storage:
            self.storage.execute(
                "DELETE FROM points WHERE id = ?",
                (key,),
            )
        self._legacy_keys.pop(point_id, None)

    def load(self) -> Iterable[models.PointStruct]:
        """
        Load a point from the local storage.

        Older versions could store a point in several rows (per UUID spelling or key pickle
        protocol). The highest rowid is the latest write, the other rows are removed.

        Returns:
            point: loaded point
        """
        cursor = self.storage.cursor()
        cursor.execute("SELECT rowid, id, point FROM points ORDER BY rowid DESC")
        self._legacy_keys = {}
        loaded = set()
        outdated = []
        for rowid, key, value in cursor.fetchall():
            point = pickle.loads(value)
            point_id = normalize_point_id(point.id)
            if point_id in loaded:
                outdated.append(rowid)
                continue
            loaded.add(point_id)
            if key != self.encode_key(point_id):
                self._legacy_keys[point_id] = key
            if point_id != point.id:
                point = construct(
                    models.PointStruct, id=point_id, vector=point.vector, payload=point.payload
                )
            yield point

        if outdated:
            self._remove_outdated(outdated)

    def _remove_outdated(self, rowids: list[int]) -> None:
        with self.storage:
            self.storage.executemany(
                "DELETE FROM points WHERE rowid = ?", [(rowid,) for rowid in rowids]
            )
        show_warning(
            f"Removed {len(rowids)} outdated copies of points from {self.location}, keeping "
            "the latest copy of each point. They were written by a qdrant-client version "
            "which stored each spelling of a UUID as a separate point, or which ran under "
            "another Python version and failed to replace the existing copy.",
            category=UserWarning,
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
