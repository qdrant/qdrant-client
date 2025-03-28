#!/usr/bin/env bash
set -e

PROJECT_ROOT="$(pwd)/$(dirname "$0")/../"

TMP_QDRANT=$(mktemp -d)
TMP_PYDANTIC=$(mktemp -d)
TMP_VENV=$(mktemp -d)

cleanup() {
  echo "Cleaning up temporary directories..."
  rm -rf "$TMP_QDRANT" "$TMP_PYDANTIC" "$TMP_VENV"
}
trap cleanup EXIT

cd "$TMP_QDRANT"
git clone --sparse --filter=blob:none --depth=1 -b dev git@github.com:qdrant/qdrant.git
cd qdrant
git sparse-checkout add docs/redoc/master

OPENAPI_PATH="$(pwd)/docs/redoc/master/openapi.json"
if [ ! -f "$OPENAPI_PATH" ]; then
  echo "Failed to generate inference structures"
  exit 1
fi

cd "$TMP_PYDANTIC"
git clone git@github.com:qdrant/pydantic_openapi_v3.git
cd pydantic_openapi_v3

# Set up venv instead of poetry
python3.10 -m venv "$TMP_VENV"
source "$TMP_VENV/bin/activate"
pip install --upgrade pip
pip install .

cp "$OPENAPI_PATH" openapi-qdrant.yaml

PATH_TO_QDRANT_CLIENT="$PROJECT_ROOT"
INPUT_YAML=openapi-qdrant.yaml \
ROOT_PACKAGE_NAME="qdrant_client" \
IMPORT_NAME="qdrant_client.http" \
PACKAGE_NAME=qdrant_openapi_client \
bash -x scripts/model_data_generator.sh

rm -rf "${PATH_TO_QDRANT_CLIENT}/qdrant_client/http/"
mv scripts/output/qdrant_openapi_client "${PATH_TO_QDRANT_CLIENT}/qdrant_client/http/"

deactivate
