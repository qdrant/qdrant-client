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
    neg = sparse_vectors[1][vector_name]
    # res = client.recommend(
    #     collection_name=collection_name,
    #     positive=[pos],
    #     negative=[neg],
    #     strategy=models.RecommendStrategy.AVERAGE_VECTOR,
    #     limit=2,
    #     using=vector_name,
    #     with_vectors=True
    # )
    # print(res)
    res = client.recommend(
        collection_name=collection_name,
        positive=[pos],
        negative=[neg],
        strategy=models.RecommendStrategy.BEST_SCORE,
        limit=2,
        using=vector_name,
        with_vectors=True,
    )
    print(res)
    res = client.recommend(
        collection_name=collection_name,
        positive=[pos],
        negative=[],
        strategy=models.RecommendStrategy.AVERAGE_VECTOR,
        limit=2,
        using=vector_name,
        with_vectors=True,
    )
    print(res)
    res = client.recommend(
        collection_name=collection_name,
        positive=[pos],
        negative=[],
        strategy=models.RecommendStrategy.BEST_SCORE,
        limit=2,
        using=vector_name,
        with_vectors=True,
    )
    print(res)
    res = client.recommend(
        collection_name=collection_name,
        positive=[neg],
        negative=[],
        strategy=models.RecommendStrategy.AVERAGE_VECTOR,
        limit=2,
        using=vector_name,
        with_vectors=True,
    )
    print(res)
    res = client.recommend(
        collection_name=collection_name,
        positive=[neg],
        negative=[],
        strategy=models.RecommendStrategy.BEST_SCORE,
        limit=2,
        using=vector_name,
        with_vectors=True,
    )
    print(res)
