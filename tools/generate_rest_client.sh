#!/usr/bin/env bash

set -e

PROJECT_ROOT="$(pwd)/$(dirname "$0")/../"

cd $(mktemp -d)

git clone --sparse --filter=blob:none --depth=1 -b dev git@github.com:qdrant/qdrant.git

cd qdrant
git sparse-checkout add docs/redoc/master

CLEAN_OPENAPI_PATH="$(pwd)/docs/redoc/master/openapi.json"
OPENAPI_PATH="$(pwd)/extended-openapi.json"
STRUCTURES_PATH="$PROJECT_ROOT/tools/structures.json"

python3 $PROJECT_ROOT/tools/merge_openapi.py "$CLEAN_OPENAPI_PATH" "$STRUCTURES_PATH" $OPENAPI_PATH

if [ $? -ne 0 ]; then
  echo "Failed to generate inference structures"
  exit 1
fi

cd $(mktemp -d)

git clone git@github.com:qdrant/pydantic_openapi_v3.git
cd pydantic_openapi_v3

poetry install

cp $OPENAPI_PATH openapi-qdrant.yaml

PATH_TO_QDRANT_CLIENT=$PROJECT_ROOT

INPUT_YAML=openapi-qdrant.yaml IMPORT_NAME="qdrant_client.http" PACKAGE_NAME=qdrant_openapi_client bash -x scripts/model_data_generator.sh

rm -rf ${PATH_TO_QDRANT_CLIENT}/qdrant_client/http/
mv scripts/output/qdrant_openapi_client ${PATH_TO_QDRANT_CLIENT}/qdrant_client/http/
