#!/bin/bash

set -ex

function stop_docker()
{
  echo "stopping qdrant_test"
  docker stop qdrant_test
}

# Ensure current path is project root
cd "$(dirname "$0")/../"

QDRANT_LATEST="v1.4.0"
QDRANT_VERSION=${QDRANT_VERSION:-"$QDRANT_LATEST"}
REST_PORT="6333"
GRPC_PORT="6334"

QDRANT_HOST=localhost:${REST_PORT}

docker run -d --rm \
           -p ${REST_PORT}:${REST_PORT} \
           -p ${GRPC_PORT}:${GRPC_PORT} \
           -e QDRANT__SERVICE__GRPC_PORT=${GRPC_PORT} \
           --name qdrant_test qdrant/qdrant:${QDRANT_VERSION}

trap stop_docker SIGINT
trap stop_docker ERR

until $(curl --output /dev/null --silent --get --fail http://$QDRANT_HOST/collections); do
  printf 'waiting for server to start...'
  sleep 5
done

# If running backwards compatibility tests, skip local compatibility tests
# Backwards compatibility tests are enabled by setting QDRANT_VERSION to a version that is not the latest

if [ "$QDRANT_VERSION" != "$QDRANT_LATEST" ]; then
  pytest --ignore=tests/congruence_tests
else
  pytest
fi


echo "Ok, that is enough"

stop_docker
