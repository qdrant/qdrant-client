"""Tests for the gRPC default-timeout contract.

Covers:

* ``QdrantRemote.DEFAULT_GRPC_TIMEOUT`` and ``AsyncQdrantRemote.DEFAULT_GRPC_TIMEOUT``
  are finite seconds (currently 5s) — guards against the legacy "unlimited" wording.
* The ``timeout`` parameter block in the ``QdrantClient`` and ``AsyncQdrantClient``
  source docstrings no longer claims gRPC is unlimited by default and does not
  reintroduce that wording.

Regression coverage for https://github.com/qdrant/qdrant-client/issues/1023.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from qdrant_client.async_qdrant_remote import AsyncQdrantRemote
from qdrant_client.qdrant_remote import QdrantRemote


EXPECTED_DEFAULT_GRPC_TIMEOUT_SECONDS = 5
MISLEADING_PHRASE = "unlimited for gRPC"


class TestDefaultGrpcTimeoutConstant:
    def test_sync_default_is_five_seconds(self) -> None:
        assert QdrantRemote.DEFAULT_GRPC_TIMEOUT == EXPECTED_DEFAULT_GRPC_TIMEOUT_SECONDS

    def test_async_default_is_five_seconds(self) -> None:
        assert AsyncQdrantRemote.DEFAULT_GRPC_TIMEOUT == EXPECTED_DEFAULT_GRPC_TIMEOUT_SECONDS


@pytest.mark.parametrize(
    "source_path, class_name",
    [
        (
            Path(__file__).resolve().parents[1] / "qdrant_client" / "qdrant_client.py",
            "QdrantClient",
        ),
        (
            Path(__file__).resolve().parents[1]
            / "qdrant_client"
            / "async_qdrant_client.py",
            "AsyncQdrantClient",
        ),
    ],
    ids=["sync", "async"],
)
class TestTimeoutDocstring:
    def test_timeout_block_documents_finite_default(
        self, source_path: Path, class_name: str
    ) -> None:
        full = ast.parse(source_path.read_text())
        block = None
        for node in ast.walk(full):
            if isinstance(node, ast.ClassDef) and node.name == class_name:
                block = ast.get_docstring(node) or ""
                break
        assert block, f"{class_name} docstring is empty"
        assert MISLEADING_PHRASE not in block, (
            f"Outdated gRPC-default wording resurfaced in {class_name}"
        )
        assert "5 seconds" in block, (
            f"Expected explicit '5 seconds' default in {class_name} docstring"
        )
