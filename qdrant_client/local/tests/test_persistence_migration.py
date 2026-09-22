"""Legacy DBM migration must not skip backends that use sidecar files.

`dbm.dumb` creates `storage.dbm.dat`/`.dir` (no unsuffixed file), same class
as macOS `dbm.ndbm` (`storage.dbm.db`). Detection via `Path.exists()` misses
both, so points disappear after upgrade. See #1456.
"""

import dbm.dumb
import pickle

from qdrant_client.http import models
from qdrant_client.local.persistence import (
    CollectionPersistence,
    STORAGE_FILE_NAME_OLD,
    try_migrate_to_sqlite,
)


def test_migrate_dumb_dbm_sidecars(tmp_path) -> None:
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
