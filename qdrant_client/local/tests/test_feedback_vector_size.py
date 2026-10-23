import pytest

from qdrant_client import models
from qdrant_client.local.local_collection import LocalCollection


@pytest.fixture(params=list(models.Distance))
def collection(request: pytest.FixtureRequest) -> LocalCollection:
    collection = LocalCollection(
        models.CreateCollection(
            vectors={
                "multi": models.VectorParams(
                    size=3,
                    distance=request.param,
                    multivector_config=models.MultiVectorConfig(
                        comparator=models.MultiVectorComparator.MAX_SIM
                    ),
                )
            }
        )
    )
    collection.upsert([models.PointStruct(id=1, vector={"multi": [[1.0, 0.0, 0.0]]})])
    return collection


def feedback_query(
    vectors: list[list[list[float]]], equal_scores: bool = False
) -> models.RelevanceFeedbackQuery:
    return models.RelevanceFeedbackQuery(
        relevance_feedback=models.RelevanceFeedbackInput(
            target=vectors[0],
            feedback=[
                models.FeedbackItem(example=vectors[1], score=0.9),
                models.FeedbackItem(example=vectors[2], score=0.9 if equal_scores else 0.1),
            ],
            strategy=models.NaiveFeedbackStrategy(
                naive=models.NaiveFeedbackStrategyParams(a=1.0, b=1.0, c=1.0)
            ),
        )
    )


@pytest.mark.parametrize("dimension", [1, 4])
@pytest.mark.parametrize(
    "invalid_vector", [0, 1, 2], ids=["target", "first-example", "last-example"]
)
@pytest.mark.parametrize("equal_scores", [False, True])
def test_multivector_feedback_rejects_wrong_size(
    collection: LocalCollection, dimension: int, invalid_vector: int, equal_scores: bool
) -> None:
    vectors = [[[1.0, 0.0, 0.0]], [[0.0, 1.0, 0.0]], [[0.0, 0.0, 1.0]]]
    vectors[invalid_vector] = [[0.5] * dimension]
    query = feedback_query(vectors, equal_scores)

    with pytest.raises(
        ValueError, match=f"Vector dimension error: expected dim: 3, got {dimension}"
    ):
        collection.query_points(query=query, using="multi")


def test_multivector_feedback_accepts_different_row_counts(collection: LocalCollection) -> None:
    query = feedback_query(
        [
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            [[1.0, 0.0, 0.0]],
            [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]],
        ]
    )
    points = collection.query_points(query=query, using="multi").points
    assert [point.id for point in points] == [1]
    distance = collection.get_vector_params("multi").distance
    expected_score = (
        1.2 if distance in (models.Distance.EUCLID, models.Distance.MANHATTAN) else 1.0
    )
    assert points[0].score == pytest.approx(expected_score)
