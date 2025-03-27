#!/usr/bin/env bash
set -e

PROJECT_ROOT="$(pwd)/$(dirname "$0")/../"

TMP_QDRANT=$(mktemp -d)
TMP_POETRY=$(mktemp -d)
TMP_POETRY_PROJECT=$(mktemp -d)

cleanup() {
  echo "Cleaning up temporary directories..."
  rm -rf "$TMP_QDRANT" "$TMP_POETRY" "$TMP_POETRY_PROJECT"
}
trap cleanup EXIT

export POETRY_VIRTUALENVS_PATH="$TMP_POETRY"
cd "$TMP_POETRY_PROJECT"
poetry init --no-interaction --name temp-proj
# Force Poetry to use Python 3.10, won't work if python3.10 is not available at PATH
poetry env use python3.10

PYTHON_VERSION=$(poetry run python --version)
echo "Using Python version: $PYTHON_VERSION"

poetry run pip install grpcio==1.48.2 grpcio-tools==1.48.2

cd "$TMP_QDRANT"
git clone --sparse --filter=blob:none --depth=1 -b dev git@github.com:qdrant/qdrant.git
cd qdrant
git sparse-checkout add lib/api/src/grpc/proto

PROTO_DIR="$(pwd)/lib/api/src/grpc/proto"

cd "$PROJECT_ROOT"

CLIENT_DIR="qdrant_client/proto"

cp "$PROTO_DIR"/*.proto "$CLIENT_DIR/"

rm "$CLIENT_DIR"/collections_internal_service.proto \
   "$CLIENT_DIR"/points_internal_service.proto \
   "$CLIENT_DIR"/qdrant_internal_service.proto \
   "$CLIENT_DIR"/shard_snapshots_service.proto \
   "$CLIENT_DIR"/raft_service.proto \
   "$CLIENT_DIR"/health_check.proto

grep -v 'collections_internal_service.proto' "$CLIENT_DIR/qdrant.proto" \
  | grep -v 'points_internal_service.proto' \
  | grep -v 'qdrant_internal_service.proto' \
  | grep -v 'shard_snapshots_service.proto' \
  | grep -v 'raft_service.proto' \
  | grep -v 'health_check.proto' \
  > "$CLIENT_DIR/qdrant_tmp.proto"
mv "$CLIENT_DIR/qdrant_tmp.proto" "$CLIENT_DIR/qdrant.proto"

poetry run python -m grpc_tools.protoc --proto_path=qdrant_client/proto/ \
  -I ./qdrant_client/grpc \
  ./qdrant_client/proto/*.proto \
  --python_out=./qdrant_client/grpc \
  --grpc_python_out=./qdrant_client/grpc

sed -i -re 's/^import (\w*)_pb2/from . import \1_pb2/g' ./qdrant_client/grpc/*.py
