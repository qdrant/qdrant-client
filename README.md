# Python Qdrant client library 

Client library for the [Qdrant](https://github.com/qdrant/qdrant) vector search engine.

Library contains type definitions for all Qdrant API and allows to make both Sync and Async requests.

`Pydantic` is used for describing request models and `httpx` for handling http queries.

Client allows calls for all [Qdrant API methods](https://qdrant.github.io/qdrant/redoc/index.html) directly.
It also provides some additional helper methods for frequently required operations, e.g. initial collection uploading.

## Installation

ToDo

## Examples

Instance a client
```python
from qdrant_client import QdrantClient

client = QdrantClient(host="localhost", port=6333)
```

Create a new collection
```python
client.recreate_collection(
    collection_name="my_collection",
    vector_size=100
)
```

Get info about created collection
```python
my_collection_info = client.http.collections_api.get_collection("my_collection")
print(my_collection_info.dict())
```

Search for similar vectors

```python
query_vector = np.random.rand(100)
hits = client.search(
    collection_name="my_collection",
    query_vector=query_vector,
    query_filter=None,  # Don't use any filters for now, search across all indexed points
    append_payload=True,  # Also return a stored payload for found points
    top=5  # Return 5 closest points
)
```
