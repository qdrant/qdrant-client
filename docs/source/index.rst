.. You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Qdrant Python Client Documentation
==================================

Client library for the `Qdrant <https://github.com/qdrant/qdrant>`_ vector search engine.

Library contains type definitions for all Qdrant API and allows to make both Sync and Async requests.

``Pydantic`` is used for describing request models and ``httpx`` for handling http queries.

Client allows calls for all `Qdrant API methods <https://qdrant.github.io/qdrant/redoc/index.html>`_ directly. It also provides some additional helper methods for frequently required operations, e.g. initial collection uploading.

Installation
============

.. code-block:: bash

   pip install qdrant-client

Examples
========

Instance a client

.. code-block:: python

   from qdrant_client import QdrantClient

   client = QdrantClient(host="localhost", port=6333)

Create a new collection

.. code-block:: python

   from qdrant_client.models import VectorParams, Distance

   if not client.collection_exists("my_collection"):
      client.create_collection(
         collection_name="my_collection",
         vectors_config=VectorParams(size=100, distance=Distance.COSINE),
      )

Insert vectors into a collection

.. code-block:: python

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

Search for similar vectors

.. code-block:: python

   query_vector = np.random.rand(100)
   hits = client.search(
      collection_name="my_collection",
      query_vector=query_vector,
      limit=5  # Return 5 closest points
   )

Search for similar vectors with filtering condition

.. code-block:: python

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

Async Client
============

Starting from version 1.6.1, all python client methods are available in async version.

.. code-block:: python

   from qdrant_client import AsyncQdrantClient, models
   import numpy as np
   import asyncio

   async def main():
      # Your async code using QdrantClient might be put here
      client = AsyncQdrantClient(url="http://localhost:6333")

      if not await client.collection_exists("my_collection"):
         await client.create_collection(
            collection_name="my_collection",
            vectors_config=models.VectorParams(size=10, distance=models.Distance.COSINE),
         )

      await client.upsert(
         collection_name="my_collection",
         points=[
               models.PointStruct(
                  id=i,
                  vector=np.random.rand(10).tolist(),
               )
               for i in range(100)
         ],
      )

      res = await client.search(
         collection_name="my_collection",
         query_vector=np.random.rand(10).tolist(),  # type: ignore
         limit=10,
      )

      print(res)

   asyncio.run(main())


Both, gRPC and REST API are supported in async mode.

Highlighted Classes
===================

- :class:`qdrant_client.http.models.models.PointStruct`
- :class:`qdrant_client.http.models.models.Filter`
- :class:`qdrant_client.http.models.models.VectorParams`
- :class:`qdrant_client.http.models.models.BinaryQuantization`

.. toctree::
   :maxdepth: 2
   :caption: PointStruct Reference

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

.. toctree::
   :maxdepth: 2
   :caption: Examples

   quickstart.ipynb

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   Models <qdrant_client.http.models.models>
   Exceptions <qdrant_client.http.exceptions>
   QdrantClient <qdrant_client.qdrant_client>
   AsyncQdrantClient <qdrant_client.async_qdrant_client>
   FastEmbed Mixin <qdrant_client.qdrant_fastembed>

.. toctree::
   :maxdepth: 1
   :caption: Complete Docs

   Complete Client API Docs <qdrant_client>
