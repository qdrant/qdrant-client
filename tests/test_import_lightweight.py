"""Regression tests: HTTP-only imports must stay lightweight when fastembed is installed."""

import subprocess
import sys


def _run_import_snippet(snippet: str) -> set[str]:
    code = f"""
import sys
{snippet}
heavy = [m for m in ("fastembed", "onnxruntime", "torch") if m in sys.modules]
print(",".join(heavy))
"""
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
    )
    loaded = proc.stdout.strip()
    return set(loaded.split(",")) if loaded else set()


def test_http_models_import_is_lightweight() -> None:
    loaded = _run_import_snippet(
        """
from qdrant_client.http import models as qmodels
_ = qmodels.Filter(must=[])
"""
    )
    assert loaded == set()


def test_models_import_is_lightweight() -> None:
    loaded = _run_import_snippet(
        """
from qdrant_client import models as qmodels
_ = qmodels.Filter(must=[])
"""
    )
    assert loaded == set()


def test_package_submodule_import_without_client() -> None:
    loaded = _run_import_snippet(
        """
import qdrant_client.http.models as _models
"""
    )
    assert loaded == set()
