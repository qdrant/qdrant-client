"""Sanity check for built distributions.

Guards against test-only code (and its dependencies, e.g. ``pytest``) leaking
into the published ``qdrant_client`` package. This has previously caused broken
releases where importing the client in a clean environment failed because a
shipped module tried to ``import pytest``.

The check inspects the built wheel and sdist in a distribution directory
(default: ``dist``) and fails if any shipped Python module either:

* lives in a test location (a ``tests`` package or a ``test_*.py`` module), or
* imports a test-only dependency such as ``pytest``.

Run it after ``poetry build``::

    python tools/check_package_contents.py

Optionally pass one or more distribution directories or files as arguments.
"""

from __future__ import annotations

import re
import sys
import tarfile
import zipfile
from pathlib import Path
from typing import Iterable, Iterator

# Modules that must never be importable from the published package.
FORBIDDEN_IMPORTS: tuple[str, ...] = ("pytest", "_pytest")

# Match ``import pytest`` / ``from pytest import ...`` at the start of a line.
_IMPORT_RE = re.compile(
    r"^\s*(?:import|from)\s+(" + "|".join(map(re.escape, FORBIDDEN_IMPORTS)) + r")\b",
    re.MULTILINE,
)


def _is_test_module(path: str) -> bool:
    """Return True if a shipped file path looks like test-only code."""
    parts = path.split("/")
    if any(part == "tests" for part in parts):
        return True
    name = parts[-1]
    return name.startswith("test_") and name.endswith(".py")


def _forbidden_import(source: str) -> str | None:
    """Return the first forbidden import found in the source, if any."""
    match = _IMPORT_RE.search(source)
    return match.group(1) if match else None


def _iter_python_files(dist: Path) -> Iterator[tuple[str, str]]:
    """Yield ``(member_path, source)`` for every ``.py`` file in a distribution."""
    if dist.suffix == ".whl":
        with zipfile.ZipFile(dist) as archive:
            for name in archive.namelist():
                if name.endswith(".py"):
                    yield name, archive.read(name).decode("utf-8", "replace")
    elif dist.name.endswith(".tar.gz"):
        with tarfile.open(dist, "r:gz") as archive:
            for member in archive.getmembers():
                if member.isfile() and member.name.endswith(".py"):
                    handle = archive.extractfile(member)
                    if handle is not None:
                        yield member.name, handle.read().decode("utf-8", "replace")


def _resolve_distributions(targets: Iterable[str]) -> list[Path]:
    """Expand the given paths into a list of wheel/sdist files."""
    dists: list[Path] = []
    for target in targets:
        path = Path(target)
        if path.is_dir():
            dists.extend(sorted(path.glob("*.whl")))
            dists.extend(sorted(path.glob("*.tar.gz")))
        elif path.is_file():
            dists.append(path)
        else:
            raise SystemExit(f"error: no such file or directory: {target}")
    return dists


def check_distribution(dist: Path) -> list[str]:
    """Return a list of problems found in a single distribution."""
    problems: list[str] = []
    for name, source in _iter_python_files(dist):
        if _is_test_module(name):
            problems.append(f"{dist.name}: ships test module '{name}'")
        forbidden = _forbidden_import(source)
        if forbidden is not None:
            problems.append(f"{dist.name}: '{name}' imports '{forbidden}'")
    return problems


def main(argv: list[str]) -> int:
    targets = argv[1:] or ["dist"]
    dists = _resolve_distributions(targets)
    if not dists:
        raise SystemExit(
            "error: no distributions found; run 'poetry build' first "
            "or pass a path to a wheel/sdist"
        )

    problems: list[str] = []
    for dist in dists:
        problems.extend(check_distribution(dist))

    if problems:
        print("Package sanity check failed:")
        for problem in problems:
            print(f"  - {problem}")
        print(
            "\nTest-only code must not ship in the published package. "
            "Exclude it via the 'exclude' key in pyproject.toml."
        )
        return 1

    checked = ", ".join(dist.name for dist in dists)
    print(f"Package sanity check passed ({checked}).")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
