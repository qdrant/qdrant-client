import inspect
from copy import deepcopy

import pytest

from qdrant_client import AsyncQdrantClient, QdrantClient, models


async def call(client, method, *args, **kwargs):
    result = getattr(client, method)(*args, **kwargs)
    return await result if inspect.isawaitable(result) else result


def feedback_query(target, examples):
    return models.RelevanceFeedbackQuery(
        relevance_feedback=models.RelevanceFeedbackInput(
            target=target,
            feedback=[
                models.FeedbackItem(example=examples[0], score=1.0),
                models.FeedbackItem(example=examples[1], score=0.0),
            ],
            strategy=models.NaiveFeedbackStrategy(
                naive=models.NaiveFeedbackStrategyParams(a=1.0, b=1.0, c=0.5)
            ),
        )
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("client_type", [QdrantClient, AsyncQdrantClient])
@pytest.mark.parametrize(
    "lookup_from",
    [
        None,
        models.LookupLocation(collection="search"),
        models.LookupLocation(collection="search", vector="source"),
        models.LookupLocation(collection="lookup"),
        models.LookupLocation(collection="lookup", vector="source"),
    ],
)
@pytest.mark.parametrize("reference", ["target", "examples", "both", "inline"])
async def test_feedback_lookup_matches_inline_vectors(client_type, lookup_from, reference):
    client = client_type(":memory:")
    try:
        vectors = {
            "search": {
                "dense": [[3.0, 0.0], [0.0, 2.0], [1.0, 1.0], [2.0, 1.0], [0.0, 1.0]],
                "source": [[0.2, 2.0], [2.0, 0.3], [0.5, 1.0], [1.0, 0.0], [0.0, 1.0]],
            },
            "lookup": {
                "dense": [[0.1, 1.0], [1.0, 0.2], [0.3, 0.5], [1.0, 0.0], [0.0, 1.0]],
                "source": [[0.4, 3.0], [2.0, 0.1], [0.1, 2.0], [1.0, 0.0], [0.0, 1.0]],
            },
        }
        # Both collections deliberately reuse IDs, but not their vectors. Cross-collection
        # lookup must neither read the search vectors nor exclude the colliding search IDs.
        for name, named_vectors in vectors.items():
            await call(
                client,
                "create_collection",
                name,
                vectors_config={
                    key: models.VectorParams(size=2, distance=models.Distance.DOT)
                    for key in named_vectors
                },
            )
            await call(
                client,
                "upsert",
                name,
                [
                    models.PointStruct(
                        id=i + 1,
                        vector={key: values[i] for key, values in named_vectors.items()},
                        payload={"group": i + 1},
                    )
                    for i in range(5)
                ],
            )

        lookup_collection = lookup_from.collection if lookup_from is not None else "search"
        lookup_vector = lookup_from.vector if lookup_from is not None else None
        source = vectors[lookup_collection][lookup_vector or "dense"]
        target = 1 if reference in ("target", "both") else source[0]
        examples = [2, 3] if reference in ("examples", "both") else source[1:3]
        query = feedback_query(target, examples)
        original = deepcopy(query)

        excluded = []
        if lookup_collection == "search":
            if reference in ("target", "both"):
                excluded.append(1)
            if reference in ("examples", "both"):
                excluded.extend([2, 3])
        query_filter = models.Filter(must_not=[models.HasIdCondition(has_id=excluded)])
        expected = await call(
            client,
            "query_points",
            "search",
            query=feedback_query(source[0], source[1:3]),
            using="dense",
            query_filter=query_filter,
            with_vectors=True,
        )
        actual = await call(
            client,
            "query_points",
            "search",
            query=query,
            using="dense",
            lookup_from=lookup_from,
            with_vectors=True,
        )

        assert len(expected.points) == 5 - len(excluded)
        assert [point.id for point in actual.points] == [point.id for point in expected.points]
        assert [point.score for point in actual.points] == pytest.approx(
            [point.score for point in expected.points]
        )
        assert [point.payload for point in actual.points] == [
            point.payload for point in expected.points
        ]
        assert [point.vector for point in actual.points] == [
            point.vector for point in expected.points
        ]
        assert query == original
    finally:
        await call(client, "close")


@pytest.mark.asyncio
@pytest.mark.parametrize("client_type", [QdrantClient, AsyncQdrantClient])
@pytest.mark.parametrize("missing", ["target", "example", "vector"])
async def test_feedback_lookup_does_not_fall_back_to_search(client_type, missing):
    client = client_type(":memory:")
    try:
        for name, ids in [("search", [1, 2]), ("lookup", [2])]:
            await call(
                client,
                "create_collection",
                name,
                vectors_config={
                    "dense": models.VectorParams(size=2, distance=models.Distance.DOT)
                },
            )
            await call(
                client,
                "upsert",
                name,
                [models.PointStruct(id=i, vector={"dense": [1.0, 0.0]}) for i in ids],
            )
        target = 1 if missing == "target" else 2 if missing == "vector" else [1.0, 0.0]
        examples = [1, [0.0, 1.0]] if missing == "example" else [[1.0, 0.0], [0.0, 1.0]]
        lookup = models.LookupLocation(
            collection="lookup", vector="missing" if missing == "vector" else None
        )
        expected_error = "Vector missing not found" if missing == "vector" else "Point 1"
        with pytest.raises(ValueError, match=expected_error):
            await call(
                client,
                "query_points",
                "search",
                query=feedback_query(target, examples),
                using="dense",
                lookup_from=lookup,
            )
    finally:
        await call(client, "close")


@pytest.mark.asyncio
@pytest.mark.parametrize("client_type", [QdrantClient, AsyncQdrantClient])
@pytest.mark.parametrize("entrypoint", ["query", "batch", "groups", "prefetch"])
@pytest.mark.parametrize("reference", ["target", "examples", "both"])
async def test_feedback_looks_up_ids_absent_from_search(client_type, entrypoint, reference):
    client = client_type(":memory:")
    try:
        for name, vector_name in [("search", "dense"), ("lookup", "source")]:
            await call(
                client,
                "create_collection",
                name,
                vectors_config={
                    vector_name: models.VectorParams(size=2, distance=models.Distance.DOT)
                },
            )
        await call(
            client,
            "upsert",
            "search",
            [
                models.PointStruct(id=1, vector={"dense": [3.0, 0.0]}, payload={"group": 1}),
                models.PointStruct(id=2, vector={"dense": [0.0, 2.0]}, payload={"group": 2}),
            ],
        )
        source = [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]]
        await call(
            client,
            "upsert",
            "lookup",
            [models.PointStruct(id=10 + i, vector={"source": v}) for i, v in enumerate(source)],
        )
        target = 10 if reference in ("target", "both") else source[0]
        examples = [11, 12] if reference in ("examples", "both") else source[1:3]
        query = feedback_query(target, examples)
        original = deepcopy(query)
        lookup = models.LookupLocation(collection="lookup", vector="source")
        kwargs = {"query": query, "using": "dense", "lookup_from": lookup}
        if entrypoint == "query":
            points = (await call(client, "query_points", "search", **kwargs)).points
        elif entrypoint == "batch":
            responses = await call(
                client, "query_batch_points", "search", [models.QueryRequest(**kwargs)]
            )
            points = responses[0].points
        elif entrypoint == "groups":
            response = await call(
                client, "query_points_groups", "search", group_by="group", **kwargs
            )
            points = [group.hits[0] for group in response.groups]
        else:
            response = await call(
                client,
                "query_points",
                "search",
                prefetch=models.Prefetch(**kwargs),
                query=models.FormulaQuery(formula="$score"),
            )
            points = response.points

        # target + 0.5 * (positive - negative) = [0.5, 0.5]
        assert [point.id for point in points] == [1, 2]
        assert [point.score for point in points] == pytest.approx([1.5, 1.0])
        assert query == original
    finally:
        await call(client, "close")
