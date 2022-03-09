#!/usr/bin/env bash

set -e

PROJECT_ROOT="$(pwd)/$(dirname "$0")/../"

cd $(mktemp -d)

git clone --sparse --filter=blob:none --depth=1 git@github.com:qdrant/qdrant.git
cd qdrant
git sparse-checkout add src/tonic/proto

PROTO_DIR="$(pwd)/src/tonic/proto"

# Ensure current path is project root
cd $PROJECT_ROOT

CLIENT_DIR="qdrant_client/grpc"

cp $PROTO_DIR/*.proto $CLIENT_DIR/

python -m grpc_tools.protoc -I $CLIENT_DIR --python_betterproto_out=$CLIENT_DIR $CLIENT_DIR/*.proto

mv $CLIENT_DIR/qdrant/__init__.py $CLIENT_DIR/__init__.py

rmdir $CLIENT_DIR/qdrant/
