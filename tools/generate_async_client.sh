#!/bin/bash

set -e

RELATIVE_PROJECT_ROOT="$(dirname "$0")/.."
cd $RELATIVE_PROJECT_ROOT
ABSOLUTE_PROJECT_ROOT=$(pwd)

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

# Re-apply hand-mods that the AST transformer cannot reproduce.
# async_qdrant_client.py and async_qdrant_remote.py: async def qdrant_version
# (issue #1294). The transformer excludes the sync qdrant_version from the
# async mirror, and the re-injected async version needs to survive every
# regen. Fails loudly with non-zero exit if the close() method cannot be
# found (signature change), so a stale regen is caught immediately.
python3 - "$ABSOLUTE_PROJECT_ROOT" <<'PY'
import sys, ast, pathlib

root = pathlib.Path(sys.argv[1])
targets = [
    ("qdrant_client/async_qdrant_client.py", """\
    async def qdrant_version(self) -> str:
        \"\"\"Return the Qdrant server's version (for remote mode) or the
        qdrant-client library version (for local mode). On any failure
        (timeout, connection refused, non-2xx), returns ``\"<unknown>\"``
        rather than raising \u2014 the string is best-effort.

        The inner client may return a coroutine (remote mode) or a string
        (local mode); await the coroutine when present.\"\"\"
        if hasattr(self, \"_client\"):
            result = self._client.qdrant_version()
            if inspect.iscoroutine(result):
                result = await result
            return result
        return \"<unknown>\"
"""),
    ("qdrant_client/async_qdrant_remote.py", """\
    async def qdrant_version(self) -> str:
        \"\"\"Return the Qdrant server's version string (e.g. ``\"1.10.0\"``).
        Awaits the REST ``/`` endpoint via the existing async service
        API binding. On any failure (timeout, connection refused, non-2xx,
        API error), returns ``\"<unknown>\"`` rather than raising.\"\"\"
        try:
            root = await self.http.service_api.root()
            return root.version
        except Exception:
            return \"<unknown>\"
"""),
]

for rel, inject in targets:
    p = root / rel
    if not p.exists():
        continue
    text = p.read_text()
    if "async def qdrant_version" in text:
        continue
    # Find the close() method to insert after it.
    tree = ast.parse(text)
    close_node = None
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            for item in node.body:
                if (
                    isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and item.name == "close"
                ):
                    close_node = item
                    break
            if close_node is not None:
                break
    if close_node is None:
        print(f"regen: {rel}: close() not found; async def qdrant_version NOT re-injected (manual fix required)")
        sys.exit(1)
    lines = text.splitlines(keepends=True)
    insert_at = close_node.end_lineno
    new_text = "".join(lines[:insert_at]) + "\n" + inject + "".join(lines[insert_at:])
    p.write_text(new_text)
    print(f"regen: injected async def qdrant_version into {rel}")
PY
