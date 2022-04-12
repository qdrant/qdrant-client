#!/usr/bin/env sh

set -xe

# Ensure current path is project root
cd "$(dirname "$0")/../"

pip install --no-deps qdrant-client

pip install sphinx>=4.4.0
pip install "git+https://github.com/qdrant/qdrant_sphinx_theme.git@master#egg=qdrant-sphinx-theme"

sphinx-build docs/source docs/html
