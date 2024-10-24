#!/usr/bin/env sh

set -xe

# Ensure current path is project root
cd "$(dirname "$0")/../"

brew install pandoc

pip install --upgrade "virtualenv>=20.26.6"
pip3 freeze | grep virtualenv

curl -sSL https://install.python-poetry.org | python3 -
export PATH="/opt/buildhome/.local/bin:$PATH"
poetry install

sphinx-build docs/source docs/html
