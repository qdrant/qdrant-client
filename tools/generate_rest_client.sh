#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TEMP_ENV=$(mktemp -d)
QDRANT_PATH=$(mktemp -d)
GENERATOR_PATH=$(mktemp -d)
VENV_DIR="$TEMP_ENV/rest_generator_venv"

PYTHON_BIN=""

if [[ "$(python --version 2>&1 | awk '{print $2}')" == "3.10.10" ]]; then
    PYTHON_BIN="python"
elif [[ "$(python3 --version 2>&1 | awk '{print $2}')" == "3.10.10" ]]; then
    PYTHON_BIN="python3"
elif [[ "$(python3.10 --version 2>&1 | awk '{print $2}')" == "3.10.10" ]]; then
    PYTHON_BIN="python3.10"
fi

if [[ -z "$PYTHON_BIN" ]]; then
    echo "Error: No suitable Python 3.10.10 installation found looked among {python, python3, python3.10}" >&2
    exit 1
fi

trap 'rm -rf "$TEMP_ENV"; rm -rf "$QDRANT_PATH"; rm -rf "$GENERATOR_PATH"' EXIT

"$PYTHON_BIN" -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

if ! command -v poetry &>/dev/null; then
    echo "Installing poetry in virtualenv..."
    "$VENV_DIR/bin/python" -m pip install --upgrade pip
    pip install poetry
fi

cd "$PROJECT_ROOT"
poetry install

cd "$QDRANT_PATH"
git clone --sparse --filter=blob:none --depth=1 -b dev git@github.com:qdrant/qdrant.git
cd qdrant
git sparse-checkout add docs/redoc/master

OPENAPI_PATH="$(pwd)/docs/redoc/master/openapi.json"

if [ ! -f "$OPENAPI_PATH" ]; then
  echo "Failed to get OpenAPI spec from Qdrant"
  exit 1
fi

cd "$GENERATOR_PATH"
git clone git@github.com:qdrant/pydantic_openapi_v3.git
cd pydantic_openapi_v3

poetry install

cp "$OPENAPI_PATH" openapi-qdrant.yaml

INPUT_YAML="openapi-qdrant.yaml" ROOT_PACKAGE_NAME="qdrant_client" IMPORT_NAME="qdrant_client.http" PACKAGE_NAME=qdrant_openapi_client bash -x scripts/model_data_generator.sh

rm -rf "${PROJECT_ROOT}/qdrant_client/http/"
mv scripts/output/qdrant_openapi_client "${PROJECT_ROOT}/qdrant_client/http/"

deactivate
