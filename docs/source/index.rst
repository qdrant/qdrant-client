.. You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Python Qdrant client library
=============================

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

   client.recreate_collection(
       collection_name="my_collection",
       vector_size=100
   )

Get info about created collection

.. code-block:: python

   from qdrant_client._pydantic_compat import to_dict
   my_collection_info = client.http.collections_api.get_collection("my_collection")
   print(to_dict(my_collection_info))

Insert vectors into a collection

.. code-block:: python

   from qdrant_client.http.models import PointStruct

   vectors = np.random.rand(100, 100)
   client.upsert(
       collection_name="my_collection",
       points=[
           PointStruct(
               id=idx,
               vector=vector,
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
      query_filter=None,  # Don't use any filters for now, search across all indexed points
      with_payload=True, # Also return a stored payload for found points, true by default
   )

Search for similar vectors with filtering condition

.. code-block:: python

   from qdrant_client.http.models import Filter, FieldCondition, Range

   hits = client.search(
       collection_name="my_collection",
       query_vector=query_vector,
       query_filter=Filter(
           must=[  # These conditions are required for search results
               FieldCondition(
                   key='rand_number',  # Condition based on values of `rand_number` field.
                   range=Range(
                       gte=0.5  # Select only those results where `rand_number` >= 0.5
                  )
               )
           ]
       ),
       with_payload=True, # Return payload, true by default
   )

Check out `full example code <https://github.com/qdrant/qdrant-client/blob/master/tests/test_qdrant_client.py>`_

gRPC
====

gRPC support in Qdrant client is under active development. Basic classes could be found `here <https://github.com/qdrant/qdrant-client/blob/master/qdrant_client/grpc/__init__.py>`_.

To enable (much faster) collection uploading with gRPC, use the following initialization:

.. code-block:: python

   from qdrant_client import QdrantClient

   client = QdrantClient(host="localhost", grpc_port=6334, prefer_grpc=True)

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

   examples/quickstart.ipynb

.. toctree::
   :maxdepth: 2
   :caption: Selected API Reference

   DataTypes aka Models <qdrant_client.http.models.models>
   Exceptions <qdrant_client.http.exceptions>
   
.. toctree::
   :maxdepth: 1
   :caption: Complete Docs

   Complete Client API Docs <qdrant_client.qdrant_client>
