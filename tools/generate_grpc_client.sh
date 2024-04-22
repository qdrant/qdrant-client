#!/usr/bin/env bash

set -e

PROJECT_ROOT="$(pwd)/$(dirname "$0")/../"

pip install grpcio==1.48.2
pip install grpcio-tools==1.48.2

cd $(mktemp -d)

git clone --sparse --filter=blob:none --depth=1 git@github.com:qdrant/qdrant.git
cd qdrant
git sparse-checkout add lib/api/src/grpc/proto

PROTO_DIR="$(pwd)/lib/api/src/grpc/proto"

# Ensure current path is project root
cd $PROJECT_ROOT

CLIENT_DIR="qdrant_client/proto"

cp $PROTO_DIR/*.proto $CLIENT_DIR/

# Remove internal services *.proto
rm $CLIENT_DIR/collections_internal_service.proto
rm $CLIENT_DIR/points_internal_service.proto
rm $CLIENT_DIR/qdrant_internal_service.proto
rm $CLIENT_DIR/shard_snapshots_service.proto
rm $CLIENT_DIR/raft_service.proto
rm $CLIENT_DIR/health_check.proto

cat $CLIENT_DIR/qdrant.proto \
 | grep -v 'collections_internal_service.proto' \
 | grep -v 'points_internal_service.proto' \
 | grep -v 'qdrant_internal_service.proto' \
 | grep -v 'shard_snapshots_service.proto' \
 | grep -v 'raft_service.proto' \
 | grep -v 'health_check.proto' \
  > $CLIENT_DIR/qdrant_tmp.proto
mv $CLIENT_DIR/qdrant_tmp.proto $CLIENT_DIR/qdrant.proto

python -m grpc_tools.protoc --proto_path=qdrant_client/proto/ -I ./qdrant_client/grpc ./qdrant_client/proto/*.proto --python_out=./qdrant_client/grpc --grpc_python_out=./qdrant_client/grpc

# maybe I'll remove this crutch when google makes normal imports for issue from 2016
# https://github.com/protocolbuffers/protobuf/issues/1491

sed -i -re 's/^import (\w*)_pb2/from . import \1_pb2/g' ./qdrant_client/grpc/*.py
