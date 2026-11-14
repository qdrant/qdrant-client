"""
Example of using Voyage AI embeddings, contextualized embeddings and reranking with Qdrant.

Qdrant Client does not call Voyage AI itself: the vectors are computed with the official
`voyageai` package and stored as regular dense vectors, so this works with any Qdrant
deployment (the example uses the in-memory local mode).

    pip install qdrant-client "voyageai>=0.6.0"
    export VOYAGE_API_KEY=...        # read from the environment only
    python examples/voyage_embeddings.py \\
        --embedding-model voyage-4 \\
        --contextualized-model voyage-context-4 \\
        --rerank-model rerank-3

The models are always passed explicitly, there are no defaults. `--contextualized-model` and
`--rerank-model` are optional and enable the corresponding steps.
"""

import argparse
import os

import voyageai

from qdrant_client import QdrantClient, models

COLLECTION = "voyage-example"

DOCUMENTS = [
    "Qdrant is a vector database and vector search engine.",
    "Voyage AI provides embedding and reranking models.",
    "Paris is the capital of France.",
    "Matryoshka embeddings can be truncated to smaller dimensions.",
]
QUERY = "What is a vector search engine?"

# Voyage limits a single request to 1000 inputs and to a per-model total number of tokens
# (see https://docs.voyageai.com/docs/embeddings). Longer inputs have to be split into batches.
MAX_INPUTS_PER_REQUEST = 1000


def embed_in_batches(
    client: voyageai.Client,
    texts: list[str],
    model: str,
    input_type: str,
    max_tokens_per_request: int,
    output_dimension: int | None = None,
) -> list[list[float]]:
    """Embed texts in as few requests as the limits allow, keeping the input order.

    Tokens are counted with the Voyage tokenizer (`Client.count_tokens`), not estimated.
    Pass the token limit of the model from the Voyage documentation.
    """
    batches: list[list[str]] = []
    batch: list[str] = []
    batch_tokens = 0
    token_counts = [client.count_tokens([text], model=model) for text in texts]
    for text, tokens in zip(texts, token_counts):
        if tokens > max_tokens_per_request:
            raise ValueError(
                f"A single input has {tokens} tokens, the request limit is {max_tokens_per_request}"
            )
        if batch and (
            len(batch) == MAX_INPUTS_PER_REQUEST or batch_tokens + tokens > max_tokens_per_request
        ):
            batches.append(batch)
            batch, batch_tokens = [], 0
        batch.append(text)
        batch_tokens += tokens
    if batch:
        batches.append(batch)

    vectors: list[list[float]] = []
    for batch in batches:
        result = client.embed(
            batch, model=model, input_type=input_type, output_dimension=output_dimension
        )
        vectors.extend(result.embeddings)
    return vectors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding-model", required=True, help="e.g. voyage-4")
    parser.add_argument("--contextualized-model", help="e.g. voyage-context-4")
    parser.add_argument("--rerank-model", help="e.g. rerank-3")
    parser.add_argument(
        "--output-dimension",
        type=int,
        help="Matryoshka models only: 256, 512, 1024 or 2048",
    )
    args = parser.parse_args()

    voyage = voyageai.Client(api_key=os.environ["VOYAGE_API_KEY"])
    qdrant = QdrantClient(":memory:")

    # --- embeddings: documents and queries use different `input_type` values ---
    doc_vectors = embed_in_batches(
        voyage,
        DOCUMENTS,
        model=args.embedding_model,
        input_type="document",
        # 120_000 is the lowest per-request limit among the text models, check the docs for yours
        max_tokens_per_request=120_000,
        output_dimension=args.output_dimension,
    )
    query_vector = voyage.embed(
        [QUERY],
        model=args.embedding_model,
        input_type="query",
        output_dimension=args.output_dimension,
    ).embeddings[0]

    qdrant.create_collection(
        COLLECTION,
        vectors_config=models.VectorParams(
            size=len(doc_vectors[0]), distance=models.Distance.COSINE
        ),
    )
    qdrant.upsert(
        COLLECTION,
        points=[
            models.PointStruct(id=i, vector=vector, payload={"text": text})
            for i, (text, vector) in enumerate(zip(DOCUMENTS, doc_vectors))
        ],
    )
    hits = qdrant.query_points(COLLECTION, query=query_vector, limit=3, with_payload=True).points
    for hit in hits:
        print(f"{hit.score:.3f}  {hit.payload['text']}")

    # --- reranking: reorder the candidates found by Qdrant with a Voyage reranker ---
    if args.rerank_model:
        candidates = [hit.payload["text"] for hit in hits]
        reranked = voyage.rerank(QUERY, candidates, model=args.rerank_model, top_k=2)
        for item in reranked.results:
            print(f"rerank {item.relevance_score:.3f}  {item.document}")

    # --- contextualized embeddings: chunks are embedded together with their document ---
    if args.contextualized_model:
        # Auto-chunking works in document mode only and takes one string per document.
        contextualized = voyage.contextualized_embed(
            [" ".join(DOCUMENTS)],
            model=args.contextualized_model,
            input_type="document",
            output_dimension=args.output_dimension,
            enable_auto_chunking=True,
            chunk_size=32_000,
        )
        result = contextualized.results[0]
        # `chunk_texts` holds the text of every chunk, so that it can go to the payload
        if result.chunk_texts is None:
            raise RuntimeError("The API did not return the chunk texts")
        chunks = result.chunk_texts
        qdrant.create_collection(
            f"{COLLECTION}-contextualized",
            vectors_config=models.VectorParams(
                size=len(result.embeddings[0]), distance=models.Distance.COSINE
            ),
        )
        qdrant.upsert(
            f"{COLLECTION}-contextualized",
            points=[
                models.PointStruct(id=i, vector=vector, payload={"text": chunk})
                for i, (chunk, vector) in enumerate(zip(chunks, result.embeddings))
            ],
        )
        # Queries are not chunked: pass them as a single-item list of lists.
        query_result = voyage.contextualized_embed(
            [[QUERY]],
            model=args.contextualized_model,
            input_type="query",
            output_dimension=args.output_dimension,
        )
        found = qdrant.query_points(
            f"{COLLECTION}-contextualized",
            query=query_result.results[0].embeddings[0],
            limit=1,
            with_payload=True,
        ).points
        print("contextualized:", found[0].payload["text"][:80])

    qdrant.close()


if __name__ == "__main__":
    main()
