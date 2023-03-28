import dbm
import pickle
from pathlib import Path
from typing import Iterable

from qdrant_client.http import models

STORAGE_FILE_NAME = "storage.dbm"


class CollectionPersistence:
    def __init__(self, location: str):
        """
        Create or load a collection from the local storage.
        Args:
            location: path to the collection directory.
        """

        self.location = Path(location) / STORAGE_FILE_NAME
        self.location.parent.mkdir(exist_ok=True, parents=True)
        self.storage = dbm.open(str(self.location), "c")

    def persist(self, point: models.PointStruct) -> None:
        """
        Persist a point in the local storage.
        Args:
            point: point to persist
        """
        key = pickle.dumps(point.id)
        value = pickle.dumps(point)
        self.storage[key] = value

        if hasattr(self.storage, "sync"):
            self.storage.sync()

    def delete(self, point_id: models.ExtendedPointId) -> None:
        """
        Delete a point from the local storage.
        Args:
            point_id: id of the point to delete
        """
        key = pickle.dumps(point_id)
        if key in self.storage:
            del self.storage[key]

    def load(self) -> Iterable[models.PointStruct]:
        """
        Load a point from the local storage.
        Returns:
            point: loaded point
        """
        for key in self.storage.keys():
            value = self.storage[key]
            yield pickle.loads(value)


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
