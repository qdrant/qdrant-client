#!/usr/bin/env bash

set -e

PROJECT_ROOT="$(pwd)/$(dirname "$0")/../"

cd $(mktemp -d)

git clone --sparse --filter=blob:none --depth=1 git@github.com:qdrant/qdrant.git
cd qdrant
git sparse-checkout add lib/api/src/grpc/proto

PROTO_DIR="$(pwd)/lib/api/src/grpc/proto"

# Ensure current path is project root
cd $PROJECT_ROOT

CLIENT_DIR="qdrant_client/grpc"

cp $PROTO_DIR/*.proto $CLIENT_DIR/

# Remove internal services *.proto
rm $CLIENT_DIR/points_internal_service.proto
rm $CLIENT_DIR/collections_internal_service.proto
rm $CLIENT_DIR/raft_service.proto
cat $CLIENT_DIR/qdrant.proto | grep -v 'collections_internal_service.proto' > $CLIENT_DIR/qdrant_tmp.proto
cat $CLIENT_DIR/qdrant.proto | grep -v 'points_internal_service.proto' > $CLIENT_DIR/qdrant_tmp.proto
cat $CLIENT_DIR/qdrant.proto | grep -v 'raft_service.proto' > $CLIENT_DIR/qdrant_tmp.proto
mv $CLIENT_DIR/qdrant_tmp.proto $CLIENT_DIR/qdrant.proto

python -m grpc_tools.protoc -I $CLIENT_DIR --python_betterproto_out=$CLIENT_DIR $CLIENT_DIR/*.proto

mv $CLIENT_DIR/qdrant/__init__.py $CLIENT_DIR/__init__.py

rmdir $CLIENT_DIR/qdrant/
