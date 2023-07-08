

<p align="center">
  <img height="100" src="https://github.com/qdrant/qdrant/raw/master/docs/logo.svg" alt="Qdrant">
</p>

<p align="center">
    <b>Python Client library for the <a href="https://github.com/qdrant/qdrant">Qdrant</a> vector search engine.</b>
</p>


<p align=center>
    <a href="https://pypi.org/project/qdrant-client/"><img src="https://badge.fury.io/py/qdrant-client.svg" alt="PyPI version" height="18"></a>
    <a href="https://qdrant.github.io/qdrant/redoc/index.html"><img src="https://img.shields.io/badge/Docs-OpenAPI%203.0-success" alt="OpenAPI Docs"></a>
    <a href="https://github.com/qdrant/qdrant-client/blob/master/LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-success" alt="Apache 2.0 License"></a>
    <a href="https://qdrant.to/discord"><img src="https://img.shields.io/badge/Discord-Qdrant-5865F2.svg?logo=discord" alt="Discord"></a>
    <a href="https://qdrant.to/roadmap"><img src="https://img.shields.io/badge/Roadmap-2023-bc1439.svg" alt="Roadmap 2023"></a>
</p>

# Python Qdrant Client

Client library and SDK for the [Qdrant](https://github.com/qdrant/qdrant) vector search engine.

Library contains type definitions for all Qdrant API and allows to make both Sync and Async requests.

Client allows calls for all [Qdrant API methods](https://qdrant.github.io/qdrant/redoc/index.html) directly.
It also provides some additional helper methods for frequently required operations, e.g. initial collection uploading.

See [QuickStart](https://qdrant.tech/documentation/quick-start/#create-collection) for more details!

## Installation

```
pip install qdrant-client
```

## Features

- Type hints for all API methods
- Local mode - use same API without running server
- REST and gRPC support
- Minimal dependencies

## Local mode

<p align="center">
  <!--- https://github.com/qdrant/qdrant-client/raw/master -->
  <img max-height="180" src="https://github.com/qdrant/qdrant-client/raw/master/docs/images/try-develop-deploy.png" alt="Qdrant">
</p>

Python client allows you to run same code in local mode without running Qdrant server.

Simply initialize client like this:

```python
from qdrant_client import QdrantClient

client = QdrantClient(":memory:")
# or
client = QdrantClient(path="path/to/db")  # Persists changes to disk
```

Local mode is useful for development, prototyping and testing.

- You can use it to run tests in your CI/CD pipeline.
- Run it in Colab or Jupyter Notebook, no extra dependencies required. See an [example](https://colab.research.google.com/drive/1Bz8RSVHwnNDaNtDwotfPj0w7AYzsdXZ-?usp=sharing)
- When you need to scale, simply switch to server mode.

### How it works?

We just implemented Qdrant API in pure Python.
We covered it with tests extensively to be sure it works the same as the server version.

## Connect to Qdrant server

To connect to Qdrant server, simply specify host and port:

```python
from qdrant_client import QdrantClient

client = QdrantClient(host="localhost", port=6333)
# or
client = QdrantClient(url="http://localhost:6333")
```

You can run Qdrant server locally with docker:

```bash
docker run -p 6333:6333 qdrant/qdrant:latest
```

See more launch options in [Qdrant repository](https://github.com/qdrant/qdrant#usage).


## Connect to Qdrant cloud

You can register and use [Qdrant Cloud](https://cloud.qdrant.io/) to get a free tier account with 1GB RAM.

Once you have your cluster and API key, you can connect to it like this:

```python
from qdrant_client import QdrantClient

qdrant_client = QdrantClient(
    url="https://xxxxxx-xxxxx-xxxxx-xxxx-xxxxxxxxx.us-east.aws.cloud.qdrant.io:6333",
    api_key="<your-api-key>",
)
```

## Examples


Create a new collection
```python
from qdrant_client.models import Distance, VectorParams

client.recreate_collection(
    collection_name="my_collection",
    vectors_config=VectorParams(size=100, distance=Distance.COSINE),
)
```

Insert vectors into a collection

```python
import numpy as np
from qdrant_client.models import PointStruct

vectors = np.random.rand(100, 100)
client.upsert(
    collection_name="my_collection",
    points=[
        PointStruct(
            id=idx,
            vector=vector.tolist(),
            payload={"color": "red", "rand_number": idx % 10}
        )
        for idx, vector in enumerate(vectors)
    ]
)
```

Search for similar vectors

```python
query_vector = np.random.rand(100)
hits = client.search(
    collection_name="my_collection",
    query_vector=query_vector,
    limit=5  # Return 5 closest points
)
```

Search for similar vectors with filtering condition

```python
from qdrant_client.models import Filter, FieldCondition, Range

hits = client.search(
    collection_name="my_collection",
    query_vector=query_vector,
    query_filter=Filter(
        must=[  # These conditions are required for search results
            FieldCondition(
                key='rand_number',  # Condition based on values of `rand_number` field.
                range=Range(
                    gte=3  # Select only those results where `rand_number` >= 3
                )
            )
        ]
    ),
    limit=5  # Return 5 closest points
)
```

See more examples in our [Documentation](https://qdrant.tech/documentation/)!

### gRPC

To enable (typically, much faster) collection uploading with gRPC, use the following initialization:

```python
from qdrant_client import QdrantClient

client = QdrantClient(host="localhost", grpc_port=6334, prefer_grpc=True)
```


## Async client

Async methods are available in raw autogenerated clients.
Usually, you don't need to use them directly, but if you need extra performance, you can access them directly.

### Async gRPC

Example of using raw async gRPC client:

```python
from qdrant_client import QdrantClient, grpc

client = QdrantClient(prefer_grpc=True, timeout=3.0)

grpc_collections = client.async_grpc_collections

res = await grpc_collections.List(grpc.ListCollectionsRequest(), timeout=1.0)
```

More examples can be found [here](./tests/test_async_qdrant_client.py).

### Development

This project uses git hooks to run code formatters.

Install `pre-commit` with `pip3 install pre-commit` and set up hooks with `pre-commit install`.

> pre-commit requires python>=3.8
