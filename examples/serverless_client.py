"""
Example of using the Qdrant Serverless client.

**In development — do not use yet.** This client is experimental and unstable;
it may change without notice and is not ready for production or general use.

Collection management uses the simplified serverless API; point operations
(query, upsert, ...) work exactly like in the regular client.
"""

from qdrant_client.serverless import QdrantServerless
from qdrant_client.serverless.models import (
    Bm25Params,
    Bm25Scoring,
    DenseVectorConfig,
    Distance,
    FieldCondition,
    Filter,
    FloatIndex,
    KeywordIndex,
    MatchValue,
    Nested,
    NestedCondition,
    PointStruct,
    Range,
    TextIndex,
    TextQuery,
    TextQueryInput,
    TextScoringParams,
    TextScoringType,
    TokenizerType,
)


def main() -> None:
    client = QdrantServerless(
        url="https://serverless.plush-volt.aws.development-cloud.qdrant.io",
        api_key="<your api key>",
    )

    # make the example rerunnable: creating an existing collection raises ALREADY_EXISTS
    if client.collection_exists("my-collection"):
        client.delete_collection("my-collection")

    # serverless-specific collection management: no quantization, wal,
    # segment number etc. - the serverless manager decides those
    print(
        client.create_collection(
            "my-collection",
            dense_vectors=DenseVectorConfig(size=4, distance=Distance.COSINE),
            payload_indexes={
                "color": KeywordIndex(),
                # full-text index with BM25 scoring: enables `TextQuery` on this field.
                # `scoring=True` is the same as `TextScoringParams(type=TextScoringType.BM25)`
                "description": TextIndex(scoring=True),
                # the `code` tokenizer splits identifiers into their camelCase / snake_case
                # subwords, so "parseHttpRequest" is found by "http" or "request"
                "snippet": TextIndex(
                    tokenizer=TokenizerType.CODE,
                    scoring=TextScoringParams(type=TextScoringType.BM25),
                ),
                # nested indexes: every element of the `offers` array is indexed as a
                # separate entry, so conditions on `offers[].*` match within one element
                "offers[].shop": KeywordIndex(nested="offers"),
                "offers[].price": FloatIndex(nested="offers"),
            },
        )
    )

    # point operations, same as in the regular client
    client.upsert(
        "my-collection",
        points=[
            PointStruct(
                id=1,
                vector=[0.1, 0.2, 0.3, 0.4],
                payload={
                    "color": "red",
                    "description": "A red wool sweater, warm and soft",
                    "snippet": "def parseHttpRequest(raw_bytes): ...",
                    "offers": [
                        {"shop": "alpha", "price": 49.0},
                        {"shop": "beta", "price": 120.0},
                    ],
                },
            ),
            PointStruct(
                id=2,
                vector=[0.4, 0.3, 0.2, 0.1],
                payload={
                    "color": "blue",
                    "description": "A blue cotton t-shirt, light and breathable",
                    "snippet": "class HttpResponseWriter: ...",
                    "offers": [
                        {"shop": "alpha", "price": 150.0},
                        {"shop": "beta", "price": 15.0},
                    ],
                },
            ),
        ],
    )

    # collection stats: `status` (green / yellow), estimated `size_bytes`,
    # approximate `indexed_vectors_count` and `updates` (applied / pending).
    # They are eventually consistent and may be `None` right after creation.
    print(client.get_collections())  # each summary includes `size_bytes`
    info = client.get_collection("my-collection")
    print(
        info.status,
        info.point_count,
        info.size_bytes,
        info.indexed_vectors_count,
        info.updates,
    )

    # vector search
    print(client.query_points("my-collection", query=[0.1, 0.2, 0.3, 0.4]))

    # BM25 ranking over a text index with `scoring`: `using` names the payload field
    print(
        client.query_points(
            "my-collection", query=TextQuery(text="warm sweater"), using="description"
        )
    )

    # BM25 parameters `k` and `b` are set per query
    print(
        client.query_points(
            "my-collection",
            query=TextQuery(
                text=TextQueryInput(
                    query="http", scoring=Bm25Scoring(bm25=Bm25Params(k=1.5, b=0.5))
                )
            ),
            using="snippet",
        )
    )

    # nested filter: matches only points where a single offer is from `beta` AND under 100,
    # i.e. point 2 (beta at 15.0), but not point 1 (beta at 120.0, the cheap offer is from alpha)
    print(
        client.query_points(
            "my-collection",
            query=[0.1, 0.2, 0.3, 0.4],
            query_filter=Filter(
                must=[
                    NestedCondition(
                        nested=Nested(
                            key="offers",
                            filter=Filter(
                                must=[
                                    FieldCondition(
                                        key="shop", match=MatchValue(value="beta")
                                    ),
                                    FieldCondition(key="price", range=Range(lt=100.0)),
                                ]
                            ),
                        )
                    )
                ]
            ),
        )
    )

    client.delete_collection("my-collection")
    client.close()


if __name__ == "__main__":
    main()
