#!/bin/bash

set -e

RELATIVE_PROJECT_ROOT="$(dirname "$0")/.."
cd "$RELATIVE_PROJECT_ROOT"
ABSOLUTE_PROJECT_ROOT=$(pwd)

# Preserve hand-written async health_check methods before generation replaces
# the generated files. The cache is temporary and is removed on exit.
HEALTH_CHECK_CACHE_DIR=$(mktemp -d)
trap 'rm -rf "$HEALTH_CHECK_CACHE_DIR"' EXIT

python3 - "$ABSOLUTE_PROJECT_ROOT" "$HEALTH_CHECK_CACHE_DIR" <<'PY'
import ast
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
cache_dir = pathlib.Path(sys.argv[2])
cache_dir.mkdir(parents=True, exist_ok=True)

for relative_path in (
    "qdrant_client/async_qdrant_client.py",
    "qdrant_client/async_qdrant_remote.py",
):
    source_path = root / relative_path
    if not source_path.exists():
        continue

    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    lines = source.splitlines(keepends=True)
    for class_node in tree.body:
        if not isinstance(class_node, ast.ClassDef):
            continue
        method_node = next(
            (
                node
                for node in class_node.body
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name == "health_check"
            ),
            None,
        )
        if method_node is not None:
            cached_method = "".join(lines[method_node.lineno - 1 : method_node.end_lineno])
            (cache_dir / pathlib.Path(relative_path).name).write_text(
                cached_method,
                encoding="utf-8",
            )
            break
PY

python3 -m tools.async_client_generator.base_client_generator
python3 -m tools.async_client_generator.fastembed_generator
python3 -m tools.async_client_generator.client_generator
python3 -m tools.async_client_generator.remote_generator
python3 -m tools.async_client_generator.local_generator

cd $ABSOLUTE_PROJECT_ROOT/tools/async_client_generator

mv async_client_base.py $ABSOLUTE_PROJECT_ROOT/qdrant_client/async_client_base.py
mv async_qdrant_client.py $ABSOLUTE_PROJECT_ROOT/qdrant_client/async_qdrant_client.py
mv async_qdrant_fastembed.py $ABSOLUTE_PROJECT_ROOT/qdrant_client/async_qdrant_fastembed.py
mv async_qdrant_remote.py $ABSOLUTE_PROJECT_ROOT/qdrant_client/async_qdrant_remote.py
mv async_qdrant_local.py $ABSOLUTE_PROJECT_ROOT/qdrant_client/async_qdrant_local.py

cd $ABSOLUTE_PROJECT_ROOT/qdrant_client

ls -1 async*.py | autoflake --recursive --imports qdrant_client --remove-unused-variables --in-place async*.py
ls -1 async*.py | xargs -I {} ruff format --line-length 99 {}

mv async_qdrant_local.py local/async_qdrant_local.py

# The AST transformer intentionally excludes health_check because the async
# facade and remote implementations differ from their sync counterparts.
# Re-inject the cached method after each regeneration. A missing cache is an
# invalid checkout, so fail loudly instead of silently using stale code.
python3 - "$ABSOLUTE_PROJECT_ROOT" "$HEALTH_CHECK_CACHE_DIR" <<'PY'
import ast
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
cache_dir = pathlib.Path(sys.argv[2])

targets = {
    "async_qdrant_client.py": "qdrant_client/async_qdrant_client.py",
    "async_qdrant_remote.py": "qdrant_client/async_qdrant_remote.py",
}

for filename, relative_path in targets.items():
    path = root / relative_path
    source = path.read_text(encoding="utf-8")

    if filename == "async_qdrant_client.py" and "import inspect\n" not in source:
        marker = "import warnings\n"
        if marker not in source:
            raise RuntimeError(f"regen: {relative_path}: import marker not found")
        source = source.replace(marker, "import inspect\n" + marker, 1)

    tree = ast.parse(source)
    classes = [node for node in tree.body if isinstance(node, ast.ClassDef)]
    if any(
        isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
        and item.name == "health_check"
        for class_node in classes
        for item in class_node.body
    ):
        path.write_text(source, encoding="utf-8")
        continue

    close_node = next(
        (
            item
            for class_node in classes
            for item in class_node.body
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
            and item.name == "close"
        ),
        None,
    )
    if close_node is None:
        raise RuntimeError(
            f"regen: {relative_path}: close() not found; health_check was not re-injected"
        )

    cached_path = cache_dir / filename
    if not cached_path.exists():
        raise RuntimeError(
            f"regen: {relative_path}: cached health_check is missing; "
            "restore the committed async implementation before regenerating"
        )
    method = cached_path.read_text(encoding="utf-8").rstrip("\n")

    lines = source.splitlines(keepends=True)
    insert_at = close_node.end_lineno
    source = "".join(lines[:insert_at]) + "\n" + method + "\n" + "".join(lines[insert_at:])
    path.write_text(source, encoding="utf-8")
    print(f"regen: injected health_check into {relative_path}")
PY

cd "$ABSOLUTE_PROJECT_ROOT/qdrant_client"
ruff format --line-length 99 async_qdrant_client.py async_qdrant_remote.py
