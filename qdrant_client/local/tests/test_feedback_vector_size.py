import pytest

from qdrant_client import models
from qdrant_client.local.local_collection import LocalCollection


def test_multivector_feedback_wrong_vector_size():
    multivector_config = models.MultiVectorConfig(comparator=models.MultiVectorComparator.MAX_SIM)
    collection = LocalCollection(
        models.CreateCollection(
            vectors=models.VectorParams(
                size=3, distance=models.Distance.EUCLID, multivector_config=multivector_config
            )
        )
    )
    collection.upsert([models.PointStruct(id=1, vector=[[1.0, 2.0, 3.0]])])

    # euclid broadcast a one-element target over all dimensions and returned a score
    query = models.RelevanceFeedbackQuery(
        relevance_feedback=models.RelevanceFeedbackInput(
            target=[[0.5]],
            feedback=[models.FeedbackItem(example=[[0.0, 1.0, 0.0]], score=0.9)],
            strategy=models.NaiveFeedbackStrategy(
                naive=models.NaiveFeedbackStrategyParams(a=1.0, b=1.0, c=1.0)
            ),
        )
    )
    with pytest.raises(ValueError, match="expected dim: 3, got 1"):
        collection.query_points(query=query)
