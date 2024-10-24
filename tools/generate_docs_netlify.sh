#!/usr/bin/env sh

set -x
# Ensure current path is project root
cd "$(dirname "$0")/../"

brew install pandoc

pip install --upgrade "virtualenv==20.26.6"
pip freeze | grep virtualenv
curl -sSL https://install.python-poetry.org | python3 -
error_log=$(find /opt/build/repo/ -name 'poetry-installer-error-*.log' | head -n 1)
cat "$error_log"

export PATH="/opt/buildhome/.local/bin:$PATH"
poetry install

sphinx-build docs/source docs/html
