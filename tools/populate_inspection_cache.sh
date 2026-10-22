#!/bin/bash

set -xe

RELATIVE_PROJECT_ROOT="$(dirname "$0")/.."
cd $RELATIVE_PROJECT_ROOT
ABSOLUTE_PROJECT_ROOT=$(pwd)

rm -f $ABSOLUTE_PROJECT_ROOT/qdrant_client/embed/_inspection_cache.py

python3 -m tools.populate_inspection_cache

cd $ABSOLUTE_PROJECT_ROOT/qdrant_client/embed
ruff check --fix-only --unsafe-fixes --select F401,F841 _inspection_cache.py
ruff format --line-length 99 _inspection_cache.py
