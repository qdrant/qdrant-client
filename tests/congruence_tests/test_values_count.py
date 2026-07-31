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


def test_values_count_multivalue():
    """A multi-value path must satisfy every bound with the *same* count."""
    vectors_config = models.VectorParams(size=2, distance=models.Distance.COSINE)
    points = [
        # counts == [1, 10]: neither count is inside (2, 9)
        models.PointStruct(
            id=1,
            vector=[0.1, 0.2],
            payload={"nested": [{"field": [1]}, {"field": list(range(10))}]},
        ),
        # counts == [3]
        models.PointStruct(id=2, vector=[0.2, 0.3], payload={"nested": [{"field": [1, 2, 3]}]}),
        # counts == [1, 5]: 5 is inside (2, 9)
        models.PointStruct(
            id=3,
            vector=[0.3, 0.4],
            payload={"nested": [{"field": [1]}, {"field": [1, 2, 3, 4, 5]}]},
        ),
    ]

    local_client = init_local()
    init_client(local_client, points, vectors_config=vectors_config)

    remote_client = init_remote()
    init_client(remote_client, points, vectors_config=vectors_config)

    filters = [
        models.Filter(
            must=[
                models.FieldCondition(
                    key="nested[].field", values_count=models.ValuesCount(gt=2, lt=9)
                )
            ]
        ),
        models.Filter(
            must=[
                models.FieldCondition(
                    key="nested[].field", values_count=models.ValuesCount(gte=3, lte=8)
                )
            ]
        ),
        models.Filter(
            must=[
                models.FieldCondition(
                    key="nested[].field", values_count=models.ValuesCount(gte=2, lte=4)
                )
            ]
        ),
        models.Filter(
            must=[
                models.FieldCondition(key="nested[].field", values_count=models.ValuesCount(gt=2))
            ]
        ),
        models.Filter(
            must=[
                models.FieldCondition(key="nested[].field", values_count=models.ValuesCount(lt=9))
            ]
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
