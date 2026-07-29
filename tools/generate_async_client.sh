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
# async_qdrant_client.py and async_qdrant_remote.py: async def health_check
# (issue #1289). The transformer excludes the sync health_check from the
# async mirror, and the re-injected async version needs to survive every regen.
# Fails loudly with non-zero exit if the close() method cannot be found.
python3 - "$ABSOLUTE_PROJECT_ROOT" <<'PY'
import sys, ast, pathlib

root = pathlib.Path(sys.argv[1])
targets = [
    ("qdrant_client/async_qdrant_client.py", """\
    async def health_check(self) -> bool:
        \"\"\"Check whether the Qdrant server is reachable. Returns True on
        success, False on any failure. The call is best-effort: any
        exception is folded into a False return rather than raised.

        The inner client may return a coroutine (remote mode) or a bool
        (local mode); await the coroutine when present.\"\"\"
        if hasattr(self, \"_client\"):
            result = self._client.health_check()
            if inspect.iscoroutine(result):
                result = await result
            return result
        return False
"""),
    ("qdrant_client/async_qdrant_remote.py", """\
    async def health_check(self) -> bool:
        \"\"\"Check whether the Qdrant server is reachable via the REST
        ``/healthz`` endpoint. Returns True on a successful response, False
        on any exception (timeout, connection refused, non-2xx, API error).
        The call is best-effort: any exception is folded into a False return
        rather than raised.\"\"\"
        try:
            await self.http.service_api.healthz()
            return True
        except Exception:
            return False
"""),
]

for rel, inject in targets:
    p = root / rel
    if not p.exists():
        continue
    text = p.read_text()
    if "async def health_check" in text:
        continue
    # Parse the file and find the close() method, then insert after it.
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
        print(f"regen: {rel}: close() not found; async def health_check NOT re-injected (manual fix required)")
        sys.exit(1)
    # Insert after the close() method body. close_node.end_lineno is 1-based.
    lines = text.splitlines(keepends=True)
    insert_at = close_node.end_lineno  # 1-based, the line AFTER this index
    new_text = "".join(lines[:insert_at]) + "\n" + inject + "".join(lines[insert_at:])
    p.write_text(new_text)
    print(f"regen: injected async def health_check into {rel}")
PY
