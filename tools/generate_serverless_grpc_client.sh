#!/usr/bin/env bash
# Regenerates qdrant_client/serverless/grpc from the public serverless collections proto.
# Mirrors tools/generate_grpc_client.sh (same pinned tool versions).

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TEMP_ENV=$(mktemp -d)
VENV_DIR="$TEMP_ENV/grpc_generator_venv"

trap "rm -rf \"$TEMP_ENV\"" EXIT

PYTHON_BIN=""
if [[ "$(python --version 2>&1 | awk '{print $2}')" == "3.10.10" ]]; then
    PYTHON_BIN="python"
elif [[ "$(python3 --version 2>&1 | awk '{print $2}')" == "3.10.10" ]]; then
    PYTHON_BIN="python3"
elif [[ "$(python3.10 --version 2>&1 | awk '{print $2}')" == "3.10.10" ]]; then
    PYTHON_BIN="python3.10"
fi

if [[ -z "$PYTHON_BIN" ]]; then
    echo "Error: No suitable Python 3.10.10 installation found among {python, python3, python3.10}" >&2
    exit 1
fi

"$PYTHON_BIN" -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

pip install --upgrade pip
pip install "grpcio==1.62.0"
pip install "grpcio-tools==1.62.0"
pip install "mypy-protobuf==3.3.0"

cd "$PROJECT_ROOT"
PROTO_DIR="qdrant_client/serverless/proto"
OUT_DIR="qdrant_client/serverless/grpc"

# Renamed from collections.proto: the protobuf descriptor pool registers files by name,
# and "collections.proto" is already taken by the regular qdrant client proto.
# Client copy: buf.validate options are stripped (server-side only; wire format unchanged).
HEADER="// Source: https://github.com/qdrant/qdrant-cloud-public-api/blob/main/proto/qdrant/serverless/collections.proto
// Renamed to serverless_collections.proto: the protobuf descriptor pool registers files by
// name, and \"collections.proto\" is already taken by the regular qdrant client proto.
// Client copy: buf.validate options are stripped (server-side only; wire format unchanged).
// Regenerate with tools/generate_serverless_grpc_client.sh"
TMP_PROTO="$(mktemp)"
curl -fsSL https://raw.githubusercontent.com/qdrant/qdrant-cloud-public-api/main/proto/qdrant/serverless/collections.proto \
  > "$TMP_PROTO"
python3 - "$TMP_PROTO" "$PROTO_DIR/serverless_collections.proto" "$HEADER" <<'PY'
import re, sys
src_path, out_path, header = sys.argv[1], sys.argv[2], sys.argv[3]
src = open(src_path).read()
src = re.sub(r'\nimport "buf/validate/validate\.proto";\n', '\n', src)
src = re.sub(
    r' \[\(buf\.validate\.field\)\.uint32 = \{\s*gt: 0\s*lte: 100\s*\}\]',
    '',
    src,
)
if 'buf.validate' in src:
    raise SystemExit('failed to strip buf.validate annotations from collections.proto')
open(out_path, 'w').write(header + '\n' + src)
PY
rm -f "$TMP_PROTO"

"$VENV_DIR/bin/python" -m grpc_tools.protoc \
  --proto_path="$PROTO_DIR" \
  "$PROTO_DIR"/serverless_collections.proto \
  --python_out="$OUT_DIR" \
  --grpc_python_out="$OUT_DIR" \
  --mypy_out="$OUT_DIR"

# https://github.com/protocolbuffers/protobuf/issues/1491
sed -i -re 's/^import (\w*)_pb2/from . import \1_pb2/g' "$OUT_DIR"/*.py

deactivate
