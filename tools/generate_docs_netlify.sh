#!/usr/bin/env sh

set -xe

# Ensure current path is project root
cd "$(dirname "$0")/../"

pip install qdrant-client

pip install sphinx==4.5.0
pip install "git+https://github.com/qdrant/qdrant_sphinx_theme.git@master#egg=qdrant-sphinx-theme"
# sudo apt-get install -y pandoc


sphinx-build docs/source docs/html