#!/usr/bin/env bash
set -e

# Define the project root relative to this script.
PROJECT_ROOT="$(pwd)/$(dirname "$0")/../"

# Create temporary directories for the qdrant clone, pydantic_openapi_v3 clone,
# and a separate temporary directory for poetry's virtual environment.
TMP_QDRANT=$(mktemp -d)
TMP_PYDANTIC=$(mktemp -d)
TMP_POETRY_ENV=$(mktemp -d)

# Function to cleanup temporary directories on exit.
cleanup() {
  echo "Cleaning up temporary directories..."
  rm -rf "$TMP_QDRANT" "$TMP_PYDANTIC" "$TMP_POETRY_ENV"
}
trap cleanup EXIT

# --- Clone qdrant repository and extract the openapi.json ---
cd "$TMP_QDRANT"
git clone --sparse --filter=blob:none --depth=1 -b dev git@github.com:qdrant/qdrant.git
cd qdrant
git sparse-checkout add docs/redoc/master

OPENAPI_PATH="$(pwd)/docs/redoc/master/openapi.json"
if [ ! -f "$OPENAPI_PATH" ]; then
  echo "Failed to generate inference structures"
  exit 1
fi

# --- Clone pydantic_openapi_v3 repository and set up a separated poetry environment ---
cd "$TMP_PYDANTIC"
git clone git@github.com:qdrant/pydantic_openapi_v3.git
cd pydantic_openapi_v3

# Set the poetry virtual environment path to our temporary directory.
export POETRY_VIRTUALENVS_PATH="$TMP_POETRY_ENV"
# Force poetry to use python3.10 for this environment.
poetry env use python3.10
poetry install

# Copy the openapi spec to the working directory.
cp "$OPENAPI_PATH" openapi-qdrant.yaml

# Run the model data generator script with the proper environment variables.
PATH_TO_QDRANT_CLIENT="$PROJECT_ROOT"
INPUT_YAML=openapi-qdrant.yaml \
ROOT_PACKAGE_NAME="qdrant_client" \
IMPORT_NAME="qdrant_client.http" \
PACKAGE_NAME=qdrant_openapi_client \
bash -x scripts/model_data_generator.sh

# Replace the qdrant client's http module with the generated one.
rm -rf "${PATH_TO_QDRANT_CLIENT}/qdrant_client/http/"
mv scripts/output/qdrant_openapi_client "${PATH_TO_QDRANT_CLIENT}/qdrant_client/http/"
