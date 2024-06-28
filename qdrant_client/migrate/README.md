## How to run migrate_different_name.py

### Example
```python
from qdrant_client import QdrantClient
from migrate_with_different_name import migrate_with_different_name

# Create Qdrant client(s) with (existing collections)
source_client = QdrantClient('http://qdrant-client', prefer_grpc=True, api_key='...')

# Destination client (where new collections will be created)
dest_client = QdrantClient('http://qdrant-client', prefer_grpc=True, api_key='...')

# set below if you want to migrate in the same qdrant client.
# source_client = dest_client

# Define your collection mapping
collection_mapping = {
    "old_name": "new_name",
    # Add more mappings as needed
}

# print time
import time
print("Start time: ", time.ctime())

# Call the migrate function
migrate_with_different_name(
    source_client=source_client,
    dest_client=dest_client,
    collection_mapping=collection_mapping,
    recreate_on_collision=True,
    batch_size=100
)

print("Migration completed successfully!")
print("End time: ", time.ctime())
```
