#!/usr/bin/env sh

set -xe

# Ensure current path is project root
cd "$(dirname "$0")/../"

python3 -m venv venv
source venv/bin/activate

brew install pandoc

export PATH="/opt/buildhome/.local/bin:$PATH"
pip3 install poetry
poetry install

sphinx-build docs/source docs/html
