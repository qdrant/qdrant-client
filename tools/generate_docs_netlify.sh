#!/usr/bin/env sh

set -xe

# Ensure current path is project root
cd "$(dirname "$0")/../"

pip install sphinx>=4.4.0
pip install "https://github.com/qdrant/qdrant_sphinx_theme.git#egg=qdrant-sphinx-theme"

sphinx-build docs/source docs/html
