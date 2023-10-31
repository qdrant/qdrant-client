#!/usr/bin/env sh

set -xe

# Ensure current path is project root
cd "$(dirname "$0")/../"

brew install pandoc

curl -sSL https://install.python-poetry.org | python3 -
export PATH="/opt/buildhome/.local/bin:$PATH"
poetry install

sphinx-build docs/source docs/html