from typing import Any

import numpy as np
import pytest
from pydantic import BaseModel, ValidationError

from qdrant_client import models
from qdrant_client._numpy_vectors import NumpyVectorsToLists, _vectors_to_lists
from qdrant_client._pydantic_compat import PYDANTIC_V2

pytestmark = pytest.mark.skipif(not PYDANTIC_V2, reason="the fast path is for pydantic 2 only")

rng = np.random.default_rng(42)
dense = rng.standard_normal(384).astype(np.float32)
multi = rng.standard_normal((3, 16)).astype(np.float32)
# multivectors of different points usually have different numbers of rows
ragged = [rng.standard_normal((rows, 16)).astype(np.float32) for rows in (3, 5)]
sparse = models.SparseVector(indices=[1, 7], values=[0.5, 0.25])


@pytest.mark.parametrize(
    "vector, expected",
    [
        (dense, dense.tolist()),
        (dense.astype(np.float64), dense.astype(np.float64).tolist()),
        (multi, multi.tolist()),
        (list(multi), multi.tolist()),
        (
            {"dense": dense, "multi": multi, "sparse": sparse},
            {"dense": dense.tolist(), "multi": multi.tolist(), "sparse": sparse},
        ),
    ],
    ids=["dense", "dense float64", "multi", "multi as rows", "named"],
)
@pytest.mark.parametrize("model", [models.PointStruct, models.PointVectors])
def test_point_vectors(model: type[BaseModel], vector: Any, expected: Any) -> None:
    assert model(id=1, vector=vector).vector == expected
    assert model.model_validate({"id": 1, "vector": vector}).vector == expected


@pytest.mark.parametrize(
    "vectors, expected",
    [
        (np.stack([dense, dense]), [dense.tolist()] * 2),
        ([dense, dense], [dense.tolist()] * 2),
        (ragged, [rows.tolist() for rows in ragged]),
        (
            {"dense": np.stack([dense, dense]), "multi": ragged, "sparse": [sparse, sparse]},
            {
                "dense": [dense.tolist()] * 2,
                "multi": [rows.tolist() for rows in ragged],
                "sparse": [sparse, sparse],
            },
        ),
    ],
    ids=["dense", "dense as rows", "ragged multi", "named"],
)
def test_batch_vectors(vectors: Any, expected: Any) -> None:
    assert models.Batch(ids=[1, 2], vectors=vectors).vectors == expected
    assert models.PointsBatch(batch={"ids": [1, 2], "vectors": vectors}).batch.vectors == expected


def test_memmap(tmp_path: Any) -> None:
    np.save(tmp_path / "vectors.npy", np.stack([dense, dense]))
    vectors = np.load(tmp_path / "vectors.npy", mmap_mode="r")

    assert models.Batch(ids=[1, 2], vectors=vectors).vectors == [dense.tolist()] * 2


def test_vectors_to_lists() -> None:
    lists = dense.tolist()

    assert _vectors_to_lists(dense) == lists
    assert _vectors_to_lists([dense, dense]) == [lists, lists]
    assert _vectors_to_lists({"dense": dense, "sparse": sparse}) == {
        "dense": lists,
        "sparse": sparse,
    }


def test_float32_values_are_widened_not_rounded() -> None:
    assert models.PointStruct(id=1, vector=np.float32([0.1])).vector == [0.10000000149011612]


def test_invalid_input_still_fails() -> None:
    with pytest.raises(ValidationError, match=r"vector\.dict\[str,.*\]\.1\.\[key\]"):
        models.PointStruct(id=1, vector={"dense": dense, 1: dense})


@pytest.mark.parametrize(
    "model, field",
    [(models.PointStruct, "vector"), (models.PointVectors, "vector"), (models.Batch, "vectors")],
    ids=["PointStruct", "PointVectors", "Batch"],
)
def test_generated_fields_convert_numpy_vectors(model: type[BaseModel], field: str) -> None:
    # pydantic_openapi_v3 annotates VectorStruct and BatchVectorStruct fields
    assert NumpyVectorsToLists in model.model_fields[field].metadata
