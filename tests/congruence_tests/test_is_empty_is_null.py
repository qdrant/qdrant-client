from qdrant_client import models
from tests.congruence_tests.test_common import (
    COLLECTION_NAME,
    compare_client_results,
    init_client,
    init_local,
    init_remote,
)


def test_field_condition_is_empty_is_null():
    """`FieldCondition.is_empty` / `is_null` are the shorthand syntax for
    `IsEmptyCondition` / `IsNullCondition`."""
    vectors_config = models.VectorParams(size=2, distance=models.Distance.COSINE)
    points = [
        # non-empty list: neither empty nor null
        models.PointStruct(id=1, vector=[0.1, 0.2], payload={"field": [1, 2]}),
        # empty list: empty, not null
        models.PointStruct(id=2, vector=[0.2, 0.3], payload={"field": []}),
        # null: both empty and null
        models.PointStruct(id=3, vector=[0.3, 0.4], payload={"field": None}),
        # missing key: empty, but not null
        models.PointStruct(id=4, vector=[0.4, 0.5], payload={}),
        # list containing a null: not empty, but null
        models.PointStruct(id=5, vector=[0.5, 0.6], payload={"field": [None, 1]}),
        # scalars and objects are neither empty nor null
        models.PointStruct(id=6, vector=[0.6, 0.7], payload={"field": 0}),
        models.PointStruct(id=7, vector=[0.7, 0.8], payload={"field": ""}),
        models.PointStruct(id=8, vector=[0.8, 0.9], payload={"field": {"a": 1}}),
        # the same shapes under a nested key, and under a key resolving to several values
        models.PointStruct(id=9, vector=[0.9, 1.0], payload={"nested": {"field": None}}),
        models.PointStruct(id=10, vector=[1.0, 1.1], payload={"nested": {"field": []}}),
        models.PointStruct(id=11, vector=[1.1, 1.2], payload={"nested": {}}),
        models.PointStruct(id=12, vector=[1.2, 1.3], payload={"nested": {"field": [1, 2]}}),
        models.PointStruct(
            id=13, vector=[1.3, 1.4], payload={"array": [{"field": 1}, {"field": None}]}
        ),
        models.PointStruct(
            id=14, vector=[1.4, 1.5], payload={"array": [{"field": 1}, {"field": 2}]}
        ),
        models.PointStruct(id=15, vector=[1.5, 1.6], payload={"array": []}),
        models.PointStruct(
            id=16, vector=[1.6, 1.7], payload={"array": [{"field": []}, {"field": 1}]}
        ),
    ]

    local_client = init_local()
    init_client(local_client, points, vectors_config=vectors_config)

    remote_client = init_remote()
    init_client(remote_client, points, vectors_config=vectors_config)

    filters = [
        models.Filter(must=[models.FieldCondition(key="field", is_empty=True)]),
        models.Filter(must=[models.FieldCondition(key="field", is_empty=False)]),
        models.Filter(must=[models.FieldCondition(key="field", is_null=True)]),
        models.Filter(must=[models.FieldCondition(key="field", is_null=False)]),
        models.Filter(must_not=[models.FieldCondition(key="field", is_empty=True)]),
        models.Filter(must_not=[models.FieldCondition(key="field", is_null=True)]),
        # the verbose conditions the shorthand abbreviates
        models.Filter(must=[models.IsEmptyCondition(is_empty=models.PayloadField(key="field"))]),
        models.Filter(must=[models.IsNullCondition(is_null=models.PayloadField(key="field"))]),
        # nested keys, and keys resolving to several values
        models.Filter(must=[models.FieldCondition(key="nested.field", is_empty=True)]),
        models.Filter(must=[models.FieldCondition(key="nested.field", is_empty=False)]),
        models.Filter(must=[models.FieldCondition(key="nested.field", is_null=True)]),
        models.Filter(must=[models.FieldCondition(key="nested.field", is_null=False)]),
        models.Filter(must=[models.FieldCondition(key="array[].field", is_empty=True)]),
        models.Filter(must=[models.FieldCondition(key="array[].field", is_empty=False)]),
        models.Filter(must=[models.FieldCondition(key="array[].field", is_null=True)]),
        models.Filter(must=[models.FieldCondition(key="array[].field", is_null=False)]),
        models.Filter(must_not=[models.FieldCondition(key="array[].field", is_empty=True)]),
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
        compare_client_results(
            local_client,
            remote_client,
            lambda c, f=flt: c.count(COLLECTION_NAME, count_filter=f).count,
        )
