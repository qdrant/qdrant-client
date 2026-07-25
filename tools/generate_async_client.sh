#!/bin/bash

set -e

RELATIVE_PROJECT_ROOT="$(dirname "$0")/.."
cd $RELATIVE_PROJECT_ROOT
ABSOLUTE_PROJECT_ROOT=$(pwd)

python3 -m tools.async_client_generator.base_client_generator
python3 -m tools.async_client_generator.fastembed_generator
python3 -m tools.async_client_generator.client_generator
python3 -m tools.async_client_generator.remote_generator
python3 -m tools.async_client_generator.local_generator

cd $ABSOLUTE_PROJECT_ROOT/tools/async_client_generator

mv async_client_base.py $ABSOLUTE_PROJECT_ROOT/qdrant_client/async_client_base.py
mv async_qdrant_client.py $ABSOLUTE_PROJECT_ROOT/qdrant_client/async_qdrant_client.py
mv async_qdrant_fastembed.py $ABSOLUTE_PROJECT_ROOT/qdrant_client/async_qdrant_fastembed.py
mv async_qdrant_remote.py $ABSOLUTE_PROJECT_ROOT/qdrant_client/async_qdrant_remote.py
mv async_qdrant_local.py $ABSOLUTE_PROJECT_ROOT/qdrant_client/async_qdrant_local.py

cd $ABSOLUTE_PROJECT_ROOT/qdrant_client

ls -1 async*.py | autoflake --recursive --imports qdrant_client --remove-unused-variables --in-place async*.py

# The AST transformer copies parameter annotations verbatim from the sync
# source, so sync-only types like httpx.Client leak into the async
# signatures. Remap the known cases here so regen output matches the
# manually-edited files.
sed -i.bak 's/http_client: httpx.Client /http_client: httpx.AsyncClient /g' \
    async_qdrant_remote.py async_qdrant_client.py
rm -f async_qdrant_remote.py.bak async_qdrant_client.py.bak

ls -1 async*.py | xargs -I {} ruff format --line-length 99 {}

mv async_qdrant_local.py local/async_qdrant_local.py
