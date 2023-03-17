#!/bin/bash

set -ex

function stop_docker()
{
  echo "stopping qdrant_test"
  docker stop qdrant_test
}

# Ensure current path is project root
cd "$(dirname "$0")/../"

QDRANT_VERSION=${QDRANT_VERSION:-'v1.1.0'}

QDRANT_HOST='localhost:6333'

docker run -d --rm \
           --network=host \
           -e QDRANT__SERVICE__GRPC_PORT="6334" \
           --name qdrant_test qdrant/qdrant:${QDRANT_VERSION}

trap stop_docker SIGINT
trap stop_docker ERR

until $(curl --output /dev/null --silent --get --fail http://$QDRANT_HOST/collections); do
  printf 'waiting for server to start...'
  sleep 5
done

pytest

echo "Ok, that is enough"

stop_docker
