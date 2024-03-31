

import numpy as np
import pytest
from qdrant_client.http import models as rest_models
from qdrant_client.local.local_collection import LocalCollection
from qdrant_client.local.qdrant_local import QdrantLocal
from qdrant_client.local.distances import calculate_distance
from qdrant_client.local.sparse import SparseVector, calculate_distance_sparse
from qdrant_client.http.models import CreateCollection, VectorParams, Distance



class TestComprehensiveNaNCheck:
    @pytest.fixture
    def setup(self):
        # Create a config object for LocalCollection
        config = CreateCollection(
            vectors=VectorParams(size=128, distance=Distance.COSINE)  # Example configuration
        )
        self.local_collection = LocalCollection(config=config)
        # Initialize QdrantLocal with in-memory storage for testing
        self.qdrant_local = QdrantLocal(location=":memory:")
        # Example vector with NaN
        self.vector_with_nan = [0.1, np.nan, 0.3]
        # Sparse vector representation with NaN
        self.sparse_vector_with_nan = SparseVector(indices=[0, 1], values=[1.0, np.nan])
    def test_local_collection_add_with_nan(self):
        # Create a VectorParams object with the desired configuration
        vector_params = VectorParams(size=128, distance=Distance.COSINE)
        
        # Create a CreateCollection object with the VectorParams
        config = CreateCollection(vectors=vector_params)
        
        # Initialize LocalCollection with the config
        local_collection = LocalCollection(config=config)
        
        vector_with_nan = [np.nan, 0.5, 0.8]  # Explicitly define a vector with NaN

        with pytest.raises(ValueError):
            # Create a PointStruct with the vector containing NaN
            point = rest_models.PointStruct(id="test_id", vector=vector_with_nan, payload={})
            # Attempt to upsert the point into the LocalCollection
            local_collection.upsert([point])


    def test_qdrant_local_validate_vector_with_nan(self, setup):
        with pytest.raises(ValueError):
            # Assuming QdrantLocal has a method to validate vectors. Adjust as necessary.
            self.qdrant_local._validate_vector(self.vector_with_nan)
    def test_distance_calculation_with_nan(self, setup):
        with pytest.raises(ValueError):
            calculate_distance(np.array(self.vector_with_nan), np.array([0.0, 1.0, 0.0]), Distance.COSINE)

    def test_sparse_vector_distance_with_nan(self, setup):
        with pytest.raises(ValueError):
            calculate_distance_sparse(self.sparse_vector_with_nan, [SparseVector(indices=[0, 1], values=[0.0, 1.0])])