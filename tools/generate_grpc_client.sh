#!/usr/bin/env bash
set -e

PROJECT_ROOT="$(pwd)/$(dirname "$0")/../"

TMP_QDRANT=$(mktemp -d)
TMP_VENV=$(mktemp -d)
TMP_CLIENT=$(mktemp -d)

cleanup() {
  echo "Cleaning up temporary directories..."
  rm -rf "$TMP_QDRANT" "$TMP_VENV" "$TMP_CLIENT"
}
trap cleanup EXIT

python3.10 -m venv "$TMP_VENV"
source "$TMP_VENV/bin/activate"

PYTHON_VERSION=$(python --version)
echo "Using Python version: $PYTHON_VERSION"

pip install --upgrade pip
pip install grpcio==1.48.2 grpcio-tools==1.48.2

cd "$TMP_QDRANT"
git clone --sparse --filter=blob:none --depth=1 -b dev git@github.com:qdrant/qdrant.git
cd qdrant
git sparse-checkout add lib/api/src/grpc/proto

PROTO_DIR="$(pwd)/lib/api/src/grpc/proto"

cd "$PROJECT_ROOT"

CLIENT_DIR="qdrant_client/proto"
GRPC_OUT_DIR="qdrant_client/grpc"

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

python -m grpc_tools.protoc --proto_path="$CLIENT_DIR" \
  -I ./"$GRPC_OUT_DIR" \
  "$CLIENT_DIR"/*.proto \
  --python_out=./"$GRPC_OUT_DIR" \
  --grpc_python_out=./"$GRPC_OUT_DIR"

# Fix imports
sed -i -re 's/^import (\w*)_pb2/from . import \1_pb2/g' ./"$GRPC_OUT_DIR"/*.py

deactivate
