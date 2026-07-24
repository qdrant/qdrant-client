"""Regression tests for ``qdrant_client.embed.utils.convert_paths``.

Issue: https://github.com/qdrant/qdrant-client/issues/1219

``convert_paths`` groups dot-separated paths into a ``FieldPath`` tree,
and ``FieldPath.as_str_list()`` flattens it back to strings. When two
inputs share a prefix the shorter one (``"a.b"`` before ``"a.b.c"``)
is silently dropped on master because the tree only marks *internal*
nodes — there is no way to record that ``"a.b"`` is itself a complete
path.

These tests pin down the round-trip contract for the supported inputs
and document the broken case as ``xfail`` until #1219 is fixed.
"""

from __future__ import annotations

import pytest

from qdrant_client.embed.utils import convert_paths


def _round_trip(paths: list[str]) -> list[str]:
    tree = convert_paths(paths)
    flat: list[str] = []
    for root in tree:
        flat.extend(root.as_str_list())
    return sorted(flat)


def test_single_root_path_round_trips() -> None:
    assert _round_trip(["a"]) == ["a"]


def test_two_disjoint_paths_round_trip() -> None:
    assert _round_trip(["a.b", "c.d"]) == ["a.b", "c.d"]


def test_two_paths_with_common_root() -> None:
    assert _round_trip(["a.b.c", "a.b.d"]) == ["a.b.c", "a.b.d"]


def test_inputs_are_not_mutated() -> None:
    inputs = ["a.b", "a.b.c", "c.d"]
    snapshot = list(inputs)
    _round_trip(inputs)
    assert inputs == snapshot


@pytest.mark.xfail(
    reason=(
        "Regression for #1219: 'a.b' was lost when 'a.b.c' was also "
        "given. Pass once the bug is fixed."
    ),
    strict=False,
)
def test_round_trip_when_path_is_prefix_of_another() -> None:
    assert _round_trip(["a.b", "a.b.c"]) == ["a.b", "a.b.c"]


@pytest.mark.xfail(
    reason="Regression for #1219 — prefix dropped when a longer sibling exists.",
    strict=False,
)
def test_round_trip_preserves_input_order_invariance() -> None:
    assert _round_trip(["a.b.c", "a.b"]) == ["a.b", "a.b.c"]


@pytest.mark.xfail(
    reason="Regression for #1219 — prefix dropped when a longer sibling exists.",
    strict=False,
)
def test_round_trip_three_levels_with_prefix() -> None:
    assert _round_trip(["a.b", "a.b.c", "a.b.d"]) == ["a.b", "a.b.c", "a.b.d"]
