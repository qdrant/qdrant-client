import pytest
import os
import httpx

from qdrant_client import QdrantClient, models
from qdrant_client.embed.utils import read_base64


QDRANT_URL: str = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY: str = os.getenv("QDRANT_API_KEY", "")


# This test requires configured remote inference server, so it is disabled by default and
# expected to be used manually.
@pytest.mark.skip(reason="Requires configured remote inference server")
def test_remove_inference_image():

    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, cloud_inference=True)
    collection_name = "image_embeddings"
    model_name = "Qdrant/clip-ViT-B-32-vision"

    image_url = "https://qdrant.tech/example.png"

    # Compare inference of image exposed via url and local file
    # So download image to local file and compare results
    image_path = "/tmp/example.png"

    with httpx.Client() as httpx_client:
        response = httpx_client.get(image_url)
        with open(image_path, "wb") as f:
            f.write(response.content)

    if client.collection_exists(collection_name):
        client.delete_collection(collection_name)

    client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(
            size=client.get_embedding_size(model_name),
            distance=models.Distance.COSINE
        ),
    )

    try:
        client.upsert(
            collection_name=collection_name,
            points=[
                models.PointStruct(
                    id=1,
                    vector=models.Image(
                        image=image_url,
                        model=model_name,
                    ),
                ),
                # models.PointStruct(
                #     id=2,
                #     vector=models.Image(
                #         image=read_base64(image_path),
                #         model=model_name,
                #     ),
                # ),
            ]
        )
    except Exception as e:
        print(e)
        raise e
