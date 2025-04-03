#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TEMP_ENV=$(mktemp -d)
VENV_DIR="$TEMP_ENV/grpc_generator_venv"
QDRANT_PATH=$(mktemp -d)

trap "rm -rf \"$TEMP_ENV\"; rm -rf \"$QDRANT_PATH\"" EXIT

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

"$PYTHON_BIN" -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

pip install --upgrade pip
pip install "grpcio==1.48.2"
pip install "grpcio-tools==1.48.2"

cd "$QDRANT_PATH"
git clone --sparse --filter=blob:none --depth=1 -b dev git@github.com:qdrant/qdrant.git
cd qdrant
git sparse-checkout add lib/api/src/grpc/proto
PROTO_DIR="$(pwd)/lib/api/src/grpc/proto"

cd "$PROJECT_ROOT"
CLIENT_DIR="qdrant_client/proto"
cp "$PROTO_DIR"/*.proto "$CLIENT_DIR/"

rm -f $CLIENT_DIR/collections_internal_service.proto
rm -f $CLIENT_DIR/points_internal_service.proto
rm -f $CLIENT_DIR/qdrant_internal_service.proto
rm -f $CLIENT_DIR/shard_snapshots_service.proto
rm -f $CLIENT_DIR/raft_service.proto
rm -f $CLIENT_DIR/health_check.proto

# Clean qdrant.proto references to those removed files
grep -vE 'collections_internal_service.proto|points_internal_service.proto|qdrant_internal_service.proto|shard_snapshots_service.proto|raft_service.proto|health_check.proto' \
  "$CLIENT_DIR/qdrant.proto" > "$CLIENT_DIR/qdrant_tmp.proto"
mv "$CLIENT_DIR/qdrant_tmp.proto" "$CLIENT_DIR/qdrant.proto"

"$VENV_DIR/bin/python" -m grpc_tools.protoc \
  --proto_path=qdrant_client/proto/ \
  -I ./qdrant_client/grpc \
  ./qdrant_client/proto/*.proto \
  --python_out=./qdrant_client/grpc \
  --grpc_python_out=./qdrant_client/grpc

sed -i -re 's/^import (\w*)_pb2/from . import \1_pb2/g' ./qdrant_client/grpc/*.py

deactivate
