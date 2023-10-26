#!/usr/bin/env sh

set -e

# Ensure current path is project root
cd "$(dirname "$0")/../"

# sudo apt-get install -y pandoc
poetry run sphinx-apidoc --force --separate --no-toc -o docs/source qdrant_client
poetry run sphinx-build docs/source docs/html