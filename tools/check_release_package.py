from __future__ import annotations

import importlib
import importlib.metadata
import pkgutil
import sys
from pathlib import Path


def get_distribution_requirements() -> list[str]:
    for distribution in importlib.metadata.distributions():
        name = distribution.metadata.get("Name", "")
        if name in {"qdrant-client", "qdrant_client"}:
            return distribution.requires or []

    raise RuntimeError("Installed qdrant-client distribution metadata was not found")


def check_runtime_dependencies() -> None:
    requirements = get_distribution_requirements()
    pytest_requirements = [requirement for requirement in requirements if "pytest" in requirement.lower()]
    if pytest_requirements:
        joined = ", ".join(pytest_requirements)
        raise RuntimeError(f"Unexpected pytest runtime dependency found: {joined}")


def import_non_test_modules() -> None:
    import qdrant_client

    failed_imports: list[tuple[str, str]] = []
    for module in sorted(
        {
            module_info.name
            for module_info in pkgutil.walk_packages(
                qdrant_client.__path__,
                prefix=f"{qdrant_client.__name__}.",
            )
            if ".tests" not in module_info.name
        }
    ):
        try:
            importlib.import_module(module)
        except Exception as exc:  # pragma: no cover - script exits on failure
            failed_imports.append((module, repr(exc)))

    if failed_imports:
        details = "\n".join(f"{module}: {error}" for module, error in failed_imports)
        raise RuntimeError(f"Failed to import installed package modules:\n{details}")


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python tools/check_release_package.py <wheel-path>")

    wheel_path = Path(sys.argv[1])
    if not wheel_path.exists():
        raise SystemExit(f"Wheel not found: {wheel_path}")

    check_runtime_dependencies()
    import_non_test_modules()
    print(f"Release package sanity checks passed for {wheel_path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
