import copy
import inspect
import math

import pytest

from qdrant_client import AsyncQdrantClient, QdrantClient, models


GLOBAL_IDF = (math.log(4 / 1.5), math.log(4 / 2.5))
SCOPED_IDF = (math.log(2 / 1.5), math.log(2 / 0.5))
SCOPED_PARAMS = models.SearchParams(
    idf=models.IdfCorpusParams(
        corpus=models.Filter(
            must=[models.FieldCondition(key="tenant", match=models.MatchValue(value="a"))]
        )
    )
)


@pytest.mark.asyncio
@pytest.mark.parametrize("client_class", [QdrantClient, AsyncQdrantClient])
@pytest.mark.parametrize("use_prefetch", [False, True])
@pytest.mark.parametrize(
    "search_params, expected_idf, expected_hits",
    [
        pytest.param(None, GLOBAL_IDF, [0, 1], id="unset"),
        pytest.param(
            models.SearchParams(idf=models.IdfScope.GLOBAL), GLOBAL_IDF, [0, 1], id="global"
        ),
        pytest.param(
            models.SearchParams(idf=models.IdfCorpusParams(corpus=models.Filter())),
            GLOBAL_IDF,
            [0, 1],
            id="all-points",
        ),
        pytest.param(SCOPED_PARAMS, SCOPED_IDF, [1, 2], id="tenant"),
        pytest.param(
            {"idf": {"corpus": {"must": [{"key": "tenant", "match": {"value": "a"}}]}}},
            SCOPED_IDF,
            [1, 2],
            id="tenant-dict",
        ),
        pytest.param(
            models.SearchParams(
                idf=models.IdfCorpusParams(
                    corpus=models.Filter(must=[models.HasIdCondition(has_id=[])])
                )
            ),
            (math.log(2), math.log(2)),
            [1, 0],
            id="empty-corpus",
        ),
    ],
)
async def test_query_points_groups_idf_scope(
    client_class, use_prefetch, search_params, expected_idf, expected_hits
):
    client = client_class(":memory:")
    points = [
        models.PointStruct(
            id=point_id,
            vector={"text": models.SparseVector(indices=[index], values=[value])},
            payload={"document": point_id % 2, "tenant": tenant},
        )
        for point_id, (index, value, tenant) in enumerate(
            [(0, 0.75, "a"), (1, 1.0, "b"), (1, 0.5, "b")]
        )
    ]
    query = models.SparseVector(indices=[0, 1], values=[1.0, 1.0])
    prefetch = (
        models.Prefetch(
            query=query,
            using="text",
            limit=3,
            params=models.SearchParams(idf=models.IdfScope.GLOBAL),
        )
        if use_prefetch
        else None
    )
    original_params = copy.deepcopy(search_params)
    original_prefetch = copy.deepcopy(prefetch)

    try:
        created = client.create_collection(
            "docs",
            vectors_config={},
            sparse_vectors_config={
                "text": models.SparseVectorParams(modifier=models.Modifier.IDF)
            },
        )
        if inspect.isawaitable(created):
            await created
        upserted = client.upsert("docs", points=points)
        if inspect.isawaitable(upserted):
            await upserted

        ordinary = client.query_points(
            "docs",
            query=query,
            using="text",
            search_params=search_params,
            prefetch=prefetch,
            with_payload=True,
        )
        if inspect.isawaitable(ordinary):
            ordinary = await ordinary
        grouped = client.query_points_groups(
            "docs",
            query=query,
            using="text",
            search_params=search_params,
            prefetch=prefetch,
            group_by="document",
            group_size=1,
            with_payload=True,
        )
        if inspect.isawaitable(grouped):
            grouped = await grouped

        expected_scores = {0: 0.75 * expected_idf[0], 1: expected_idf[1], 2: 0.5 * expected_idf[1]}
        assert [point.id for point in ordinary.points] == sorted(
            expected_scores, key=expected_scores.__getitem__, reverse=True
        )
        assert {point.id: point.score for point in ordinary.points} == pytest.approx(
            expected_scores
        )
        assert [group.id for group in grouped.groups] == [
            point_id % 2 for point_id in expected_hits
        ]
        assert [[hit.id for hit in group.hits] for group in grouped.groups] == [
            [point_id] for point_id in expected_hits
        ]
        assert [group.hits[0].score for group in grouped.groups] == pytest.approx(
            [expected_scores[point_id] for point_id in expected_hits]
        )
        assert [group.hits[0].payload for group in grouped.groups] == [
            points[point_id].payload for point_id in expected_hits
        ]
        assert search_params == original_params
        assert prefetch == original_prefetch
    finally:
        closed = client.close()
        if inspect.isawaitable(closed):
            await closed
