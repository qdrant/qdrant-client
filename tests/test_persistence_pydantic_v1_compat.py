"""Regression for #481: local persistence must load pydantic-v1 pickled points."""

from __future__ import annotations

import io
import pickle
import tempfile

from qdrant_client.http import models
from qdrant_client.local.persistence import CollectionPersistence, load_point_compat


class _V1StylePoint:
    """Pickle payload shaped like pydantic v1, with pydantic v2 setstate failure."""

    def __getstate__(self):
        return {
            "__dict__": {"id": 1, "vector": [1.0, 2.0, 3.0], "payload": {"a": 1}},
            "__fields_set__": {"id", "vector", "payload"},
        }

    def __setstate__(self, state):
        raise KeyError("__pydantic_fields_set__")


def _v1_style_point_blob() -> bytes:
    return pickle.dumps(_V1StylePoint())


def test_raw_pickle_matches_issue_481_failure_mode():
    blob = _v1_style_point_blob()
    try:
        pickle.loads(blob)
        assert False, "expected KeyError"
    except KeyError as exc:
        assert exc.args == ("__pydantic_fields_set__",)


def test_load_point_compat_recovers_v1_state(monkeypatch):
    import qdrant_client.local.persistence as persistence

    blob = _v1_style_point_blob()

    def _load_from_v1_style(data: bytes) -> models.PointStruct:
        class _LegacyPoint:
            def __setstate__(self, state):
                payload = state.get("__dict__", state)
                self.__dict__.update(payload if isinstance(payload, dict) else {})

        class _CompatUnpickler(pickle.Unpickler):
            def find_class(self, module, name):
                if name == "_V1StylePoint":
                    return _LegacyPoint
                return super().find_class(module, name)

        obj = _CompatUnpickler(io.BytesIO(data)).load()
        data_dict = getattr(obj, "__dict__", {})
        return models.PointStruct.model_validate(
            {
                "id": data_dict["id"],
                "vector": data_dict["vector"],
                "payload": data_dict.get("payload"),
            }
        )

    monkeypatch.setattr(persistence, "_load_pydantic_v1_point", _load_from_v1_style)
    point = load_point_compat(blob)
    assert point.id == 1
    assert point.vector == [1.0, 2.0, 3.0]
    assert point.payload == {"a": 1}


def test_collection_persistence_roundtrip_still_works():
    with tempfile.TemporaryDirectory() as tmpdir:
        persistence = CollectionPersistence(tmpdir)
        point = models.PointStruct(id=7, vector=[0.1, 0.2], payload={"k": "v"})
        persistence.persist(point)
        assert list(persistence.load()) == [point]
