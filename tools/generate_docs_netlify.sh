#!/usr/bin/env sh

set -x

# Ensure current path is project root
cd "$(dirname "$0")/../"

pip install --upgrade "virtualenv==20.26.6"
python3 -m venv venv
source venv/bin/activate

brew install pandoc

export PATH="/opt/buildhome/.local/bin:$PATH"
#curl -sSL https://install.python-poetry.org | python3 -
#error_log=$(find /opt/build/repo/ -name 'poetry-installer-error-*.log' | head -n 1)
#cat "$error_log"
pip3 install poetry
poetry install

sphinx-build docs/source docs/html
