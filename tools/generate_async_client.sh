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
# async_qdrant_client.py and async_qdrant_remote.py: async def
# get_alias_names (issue #1306). The transformer excludes the sync
# get_alias_names from the async mirror, and the re-injected async
# version needs to survive every regen. Fails loudly with non-zero
# exit if the close() method cannot be found (signature change), so a
# stale regen is caught immediately.
python3 - "$ABSOLUTE_PROJECT_ROOT" <<'PY'
import sys, ast, pathlib

root = pathlib.Path(sys.argv[1])
targets = [
    ("qdrant_client/async_qdrant_client.py", """\
    async def get_alias_names(self) -> list[str]:
        \"\"\"Return a list of all alias names.

        A shortcut for ``[a.alias_name for a in (await self.get_aliases()).aliases]``.
        The inner client may return a coroutine (remote mode) or a
        list (local mode, sync); await the coroutine when present.

        Returns:
            A list of alias names. Order matches the server's
            response (remote) or insertion order (local). Empty list
            if no aliases exist.
        \"\"\"
        result = self._client.get_alias_names()
        if inspect.iscoroutine(result):
            result = await result
        return result
""", True),
    ("qdrant_client/async_qdrant_remote.py", """\
    async def get_alias_names(self) -> list[str]:
        return [a.alias_name for a in (await self.get_aliases()).aliases]
""", False),
]

for rel, inject, needs_inspect in targets:
    p = root / rel
    if not p.exists():
        continue
    text = p.read_text()
    if "async def get_alias_names" in text:
        continue
    if needs_inspect and "\nimport inspect\n" not in text and not text.startswith("import inspect\n"):
        first_import_idx = 0
        for i, line in enumerate(text.splitlines(keepends=True)):
            stripped = line.strip()
            if stripped.startswith("import ") or stripped.startswith("from "):
                first_import_idx = i + 1
            elif stripped and not stripped.startswith("#") and first_import_idx > 0:
                break
        lines = text.splitlines(keepends=True)
        text = "".join(lines[:first_import_idx]) + "import inspect\n" + "".join(lines[first_import_idx:])
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
        print(f"regen: {rel}: close() not found; async def get_alias_names NOT re-injected (manual fix required)")
        sys.exit(1)
    lines = text.splitlines(keepends=True)
    insert_at = close_node.end_lineno
    new_text = "".join(lines[:insert_at]) + "\n" + inject + "".join(lines[insert_at:])
    p.write_text(new_text)
    print(f"regen: injected async def get_alias_names into {rel}")
PY
