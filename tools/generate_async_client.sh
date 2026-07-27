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

# Re-apply hand-mods that the AST transformer cannot reproduce.
# async_qdrant_client.py: __aenter__/__aexit__ for `async with` support (issue #1285).
# The transformer pipeline excludes the sync __enter__/__exit__ from the async mirror
# (httpx.AsyncClient convention: sync methods don't exist on the async class). The async
# context-manager methods are added here so they survive every regen.
if grep -q "async def close(self, grpc_grace: float | None = None, \*\*kwargs: Any) -> None:" async_qdrant_client.py; then
    if ! grep -q "async def __aenter__" async_qdrant_client.py; then
        python3 - <<'PY'
import pathlib
p = pathlib.Path("async_qdrant_client.py")
text = p.read_text()
needle = '    async def close(self, grpc_grace: float | None = None, **kwargs: Any) -> None:\n        """Closes the connection to Qdrant\n\n        Args:\n            grpc_grace: Grace period for gRPC connection close. Default: None\n        """\n        if hasattr(self, "_client"):\n            await self._client.close(grpc_grace=grpc_grace, **kwargs)\n'
inject = needle + "\n    async def __aenter__(self) -> \"AsyncQdrantClient\":\n        return self\n\n    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:\n        await self.close()\n"
if needle in text:
    p.write_text(text.replace(needle, inject, 1))
    print("regen: injected __aenter__/__aexit__ into async_qdrant_client.py")
else:
    print("regen: async_qdrant_client.py close() signature changed; __aenter__/__aexit__ not re-injected (manual fix required)")
PY
    fi
fi

mv async_qdrant_local.py local/async_qdrant_local.py
