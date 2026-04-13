from qdrant_client import models
from tests.congruence_tests.test_common import (
    COLLECTION_NAME,
    compare_client_results,
    init_client,
    init_local,
    init_remote,
)


def test_values_count():
    vectors_config = models.VectorParams(size=2, distance=models.Distance.COSINE)
    points = [
        # dict: count == 1
        models.PointStruct(id=1, vector=[0.1, 0.2], payload={"field": {"a": 1, "b": 2}}),
        # list of 2: count == 2
        models.PointStruct(id=2, vector=[0.2, 0.3], payload={"field": ["x", "y"]}),
        # scalar int: count == 1
        models.PointStruct(id=3, vector=[0.3, 0.4], payload={"field": 42}),
        # scalar string: count == 1
        models.PointStruct(id=4, vector=[0.4, 0.5], payload={"field": "hello"}),
        # empty list: count == 0
        models.PointStruct(id=5, vector=[0.5, 0.6], payload={"field": []}),
        # null: count == 0
        models.PointStruct(id=6, vector=[0.6, 0.7], payload={"field": None}),
        # missing key
        models.PointStruct(id=7, vector=[0.7, 0.8], payload={}),
    ]

    local_client = init_local()
    init_client(local_client, points, vectors_config=vectors_config)

    remote_client = init_remote()
    init_client(remote_client, points, vectors_config=vectors_config)

    filters = [
        models.Filter(
            must=[models.FieldCondition(key="field", values_count=models.ValuesCount(gt=1))]
        ),
        models.Filter(
            must=[models.FieldCondition(key="field", values_count=models.ValuesCount(gte=2))]
        ),
        models.Filter(
            must=[models.FieldCondition(key="field", values_count=models.ValuesCount(lt=2))]
        ),
        models.Filter(
            must=[models.FieldCondition(key="field", values_count=models.ValuesCount(lte=1))]
        ),
        models.Filter(
            must=[models.FieldCondition(key="field", values_count=models.ValuesCount(gt=0))]
        ),
        models.Filter(
            must=[models.FieldCondition(key="field", values_count=models.ValuesCount(gte=1))]
        ),
        models.Filter(
            must=[models.FieldCondition(key="field", values_count=models.ValuesCount(lt=1))]
        ),
        models.Filter(
            must=[models.FieldCondition(key="field", values_count=models.ValuesCount(lte=0))]
        ),
        models.Filter(
            must=[
                models.FieldCondition(key="field", values_count=models.ValuesCount(gte=1, lte=2))
            ]
        ),
        models.Filter(
            must=[models.FieldCondition(key="field", values_count=models.ValuesCount(gt=0, lt=3))]
        ),
    ]

    for flt in filters:
        compare_client_results(
            local_client,
            remote_client,
            lambda c, f=flt: c.scroll(
                COLLECTION_NAME,
                scroll_filter=f,
                limit=100,
                with_payload=False,
            ),
        )
