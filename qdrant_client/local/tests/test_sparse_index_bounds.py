from collections.abc import Iterator
from pathlib import Path

import pytest

from qdrant_client import QdrantClient, models
from qdrant_client.conversions.conversion import RestToGrpc


@pytest.fixture(params=["memory", "disk"])
def sparse_client(request: pytest.FixtureRequest, tmp_path: Path) -> Iterator[QdrantClient]:
    client = (
        QdrantClient(":memory:") if request.param == "memory" else QdrantClient(path=str(tmp_path))
    )
    client.create_collection(
        "sparse", vectors_config={}, sparse_vectors_config={"s": models.SparseVectorParams()}
    )
    client.upsert(
        "sparse",
        points=[
            models.PointStruct(
                id=i,
                vector={"s": models.SparseVector(indices=[0], values=[float(i)])},
                payload={"original": i},
            )
            for i in [1, 2]
        ],
    )
    yield client
    client.close()


@pytest.mark.parametrize("index", [-1, 2**32])
@pytest.mark.parametrize(
    "operation", ["upsert-list", "upsert-batch", "update-vectors", "query", "batch-update"]
)
def test_sparse_index_out_of_range_is_rejected(
    sparse_client: QdrantClient, index: int, operation: str
) -> None:
    invalid = models.SparseVector(indices=[index], values=[1.0])
    valid = models.SparseVector(indices=[1], values=[3.0])
    before = sparse_client.retrieve("sparse", [1, 2], with_vectors=True)

    # The actual protobuf encoder defines the same bounds as the local validator.
    with pytest.raises(ValueError, match="out of range"):
        RestToGrpc.convert_sparse_vector(invalid)

    with pytest.raises(ValueError, match="Indices must be between 0 and 4294967295"):
        if operation == "upsert-list":
            sparse_client.upsert(
                "sparse",
                points=[
                    models.PointStruct(id=1, vector={"s": valid}, payload={"changed": True}),
                    models.PointStruct(id=2, vector={"s": invalid}),
                ],
            )
        elif operation == "upsert-batch":
            sparse_client.upsert(
                "sparse",
                points=models.Batch(
                    ids=[1, 2], vectors={"s": [valid, invalid]}, payloads=[{"changed": True}, {}]
                ),
            )
        elif operation == "update-vectors":
            sparse_client.update_vectors(
                "sparse",
                points=[
                    models.PointVectors(id=1, vector={"s": valid}),
                    models.PointVectors(id=2, vector={"s": invalid}),
                ],
            )
        elif operation == "query":
            sparse_client.query_points("sparse", query=invalid, using="s")
        else:
            sparse_client.batch_update_points(
                "sparse",
                update_operations=[
                    models.SetPayloadOperation(
                        set_payload=models.SetPayload(payload={"changed": True}, points=[1])
                    ),
                    models.UpsertOperation(
                        upsert=models.PointsList(
                            points=[models.PointStruct(id=2, vector={"s": invalid})]
                        )
                    ),
                ],
            )

    assert sparse_client.retrieve("sparse", [1, 2], with_vectors=True) == before
    assert sparse_client.count("sparse").count == 2


@pytest.mark.parametrize("indices", [[], [0], [2**32 - 1], [2**32 - 1, 0]])
def test_sparse_index_bounds_and_unsorted_vectors_are_valid(
    sparse_client: QdrantClient, indices: list[int]
) -> None:
    vector = models.SparseVector(
        indices=indices, values=[float(i + 1) for i in range(len(indices))]
    )
    wire = RestToGrpc.convert_sparse_vector(vector)
    assert list(type(wire).FromString(wire.SerializeToString()).indices) == indices

    sparse_client.upsert("sparse", points=[models.PointStruct(id=3, vector={"s": vector})])
    sparse_client.update_vectors(
        "sparse", points=[models.PointVectors(id=3, vector={"s": vector})]
    )
    sparse_client.query_points("sparse", query=vector, using="s")
    stored = sparse_client.retrieve("sparse", [3], with_vectors=True)[0].vector
    assert isinstance(stored, dict)
    assert stored["s"] == models.SparseVector(
        indices=sorted(indices),
        values=[value for _, value in sorted(zip(vector.indices, vector.values))],
    )
