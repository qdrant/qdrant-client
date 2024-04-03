from qdrant_client import QdrantClient, models
from tests.fixtures.points import random_sparse_vectors

if __name__ == "__main__":
    client = QdrantClient(":memory:")
    collection_name = "sparse-recommend"
    vector_name = "sparse-image"
    client.recreate_collection(
        collection_name,
        vectors_config={},
        sparse_vectors_config={vector_name: models.SparseVectorParams()},
    )
    sparse_vectors = [random_sparse_vectors({vector_name: 100}) for i in range(100)]
    client.upload_collection(
        collection_name=collection_name,
        vectors=sparse_vectors,
        ids=range(100),
    )
    rs, _ = client.scroll(collection_name=collection_name, limit=2)
    for r in rs:
        print(r)

    pos = sparse_vectors[0][vector_name]
    pos_2 = sparse_vectors[2][vector_name]
    neg = sparse_vectors[1][vector_name]
    neg_2 = sparse_vectors[3][vector_name]
    target = sparse_vectors[4][vector_name]

    r = client.discover(
        collection_name=collection_name,
        context=[models.ContextExamplePair(positive=pos, negative=neg)],
        with_payload=True,
        limit=10,
        using="sparse-image",
    )
    print(r)

    r = client.discover(
        collection_name=collection_name,
        context=[
            models.ContextExamplePair(positive=pos, negative=neg),
            models.ContextExamplePair(positive=pos_2, negative=neg_2),
        ],
        with_payload=True,
        limit=10,
        using="sparse-image",
    )
    print(r)

    r = client.discover(
        collection_name=collection_name,
        target=target,
        context=[
            models.ContextExamplePair(positive=pos, negative=neg),
        ],
        with_payload=True,
        limit=10,
        using="sparse-image",
    )
    print(r)

    r = client.discover(
        collection_name=collection_name,
        target=target,
        context=[
            models.ContextExamplePair(positive=pos, negative=neg),
            models.ContextExamplePair(positive=pos_2, negative=neg_2),
        ],
        with_payload=True,
        limit=10,
        using="sparse-image",
    )
    print(r)

    pos_id = 0
    pos_id_2 = 1
    neg_id = 2
    neg_id_2 = 3
    target_id = 4

    r = client.discover(
        collection_name=collection_name,
        context=[models.ContextExamplePair(positive=pos_id, negative=neg_id)],
        with_payload=True,
        limit=10,
        using="sparse-image",
    )
    print(r)

    r = client.discover(
        collection_name=collection_name,
        context=[
            models.ContextExamplePair(positive=pos_id, negative=neg_id),
            models.ContextExamplePair(positive=pos_id_2, negative=neg_id_2),
        ],
        with_payload=True,
        limit=10,
        using="sparse-image",
    )
    print(r)

    r = client.discover(
        collection_name=collection_name,
        target=target_id,
        context=[
            models.ContextExamplePair(positive=pos_id, negative=neg_id),
        ],
        with_payload=True,
        limit=10,
        using="sparse-image",
    )
    print(r)

    r = client.discover(
        collection_name=collection_name,
        target=target,
        context=[
            models.ContextExamplePair(positive=pos_id, negative=neg_id),
            models.ContextExamplePair(positive=pos_id_2, negative=neg_id_2),
        ],
        with_payload=True,
        limit=10,
        using="sparse-image",
    )
    print(r)
