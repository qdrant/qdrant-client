"""
Example of using the Qdrant Serverless client.

**In development — do not use yet.** This client is experimental and unstable;
it may change without notice and is not ready for production or general use.

Collection management uses the simplified serverless API; point operations
(query, upsert, ...) work exactly like in the regular client.
"""

from qdrant_client.models import PointStruct
from qdrant_client.serverless.models import DenseVectorConfig, Distance, KeywordIndex
from qdrant_client.serverless import QdrantServerless


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
    print(client.create_collection(
        "my-collection",
        dense_vectors=DenseVectorConfig(size=4, distance=Distance.COSINE),
        payload_indexes={"color": KeywordIndex()},
    ))

    print(client.get_collections())
    print(client.get_collection("my-collection"))

    # point operations, same as in the regular client
    client.upsert(
        "my-collection",
        points=[
            PointStruct(id=1, vector=[0.1, 0.2, 0.3, 0.4], payload={"color": "red"}),
            PointStruct(id=2, vector=[0.4, 0.3, 0.2, 0.1], payload={"color": "blue"}),
        ],
    )
    print(client.query_points("my-collection", query=[0.1, 0.2, 0.3, 0.4]))

    client.delete_collection("my-collection")
    client.close()


if __name__ == "__main__":
    main()
