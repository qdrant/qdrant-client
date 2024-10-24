#!/usr/bin/env sh

# Ensure current path is project root
cd "$(dirname "$0")/../"

brew install pandoc

pip install --upgrade "virtualenv==20.26.6"
pip freeze | grep virtualenv
curl -sSL https://install.python-poetry.org | python3 - 2> curl_error.log

# Display the contents of the error log file, if any
cat curl_error.log

export PATH="/opt/buildhome/.local/bin:$PATH"
poetry install

sphinx-build docs/source docs/html
