#!/bin/bash

set -xe

RELATIVE_PROJECT_ROOT="$(dirname "$0")/.."
cd $RELATIVE_PROJECT_ROOT
ABSOLUTE_PROJECT_ROOT=$(pwd)

python3 -m tools.populate_inspection_cache

cd $ABSOLUTE_PROJECT_ROOT/qdrant_client/embed

autoflake --recursive --imports qdrant_client --remove-unused-variables --in-place _inspection_cache.py
ruff format --line-length 99 _inspection_cache.py
