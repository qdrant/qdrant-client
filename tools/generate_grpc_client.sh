#!/usr/bin/env bash

cd $(mktemp -d)

git clone --sparse --filter=blob:none --depth=1 git@github.com:qdrant/qdrant.git
cd qdrant
git sparse-checkout add src/tonic/proto

PROTO_DIR="$(pwd)/src/tonic/proto"

# Ensure current path is project root
cd "$(dirname "$0")/../"

CLIENT_DIR="qdrant_client/grpc/"

cp $PROTO_DIR/*.proto $CLIENT_DIR

python -m grpc_tools.protoc -I . --python_betterproto_out=$CLIENT_DIR $CLIENT_DIR/collections.proto $CLIENT_DIR/points.proto

mv $CLIENT_DIR/qdrant/__init__.py $CLIENT_DIR/__init__.py

rmdir $CLIENT_DIR/qdrant/
