#!/usr/bin/env python3
"""Release sanity checks for qdrant-client.

This script is meant to run in CI in an environment where the package is installed
as a wheel (without dev dependencies like pytest).
"""

from __future__ import annotations

import importlib
import pkgutil
import re
import sys
from pathlib import Path


def fail(message: str) -> None:
    print(f"::error::{message}")
    raise SystemExit(1)


def check_no_pytest_import_in_library_sources() -> None:
    root = Path("qdrant_client")
    offenders: list[str] = []

    for py_file in root.rglob("*.py"):
        rel = py_file.as_posix()

        # Test-only modules are allowed to import pytest.
        if "/tests/" in rel or rel.endswith("/conftest.py") or py_file.name.startswith("test_"):
            continue

        text = py_file.read_text(encoding="utf-8")
        if re.search(r"^\s*import\s+pytest\b", text, re.MULTILINE) or re.search(
            r"^\s*from\s+pytest\b", text, re.MULTILINE
        ):
            offenders.append(rel)

    if offenders:
        fail(
            "Found pytest import(s) in library code (outside tests): "
            + ", ".join(offenders)
        )


def check_runtime_imports() -> None:
    try:
        import qdrant_client  # noqa: F401
    except Exception as exc:  # pragma: no cover - defensive
        fail(f"Importing qdrant_client failed: {exc!r}")

    import qdrant_client as qc

    for module_info in pkgutil.walk_packages(qc.__path__, prefix="qdrant_client."):
        name = module_info.name

        # Never import tests in this sanity check.
        if ".tests" in name or name.endswith(".conftest"):
            continue

        try:
            importlib.import_module(name)
        except ModuleNotFoundError as exc:
            if exc.name == "pytest":
                fail(f"Module '{name}' requires pytest at runtime")
            # optional dependency modules are allowed to fail.
        except Exception:
            # Ignore other runtime side effects; this check targets pytest leakage.
            pass


if __name__ == "__main__":
    print(f"Python: {sys.version.split()[0]}")
    check_no_pytest_import_in_library_sources()
    check_runtime_imports()
    print("Release sanity checks passed")
