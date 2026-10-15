import os
import subprocess
import sys
import textwrap

import pytest

# Both snippets print an order that must not change from one process to the next. The str ids and str group
# ids are the ones whose hash, and so whose position in a set, changes with PYTHONHASHSEED.
FORMULA_TIES = textwrap.dedent(
    """
    import uuid
    from qdrant_client import QdrantClient, models

    client = QdrantClient(":memory:")
    client.create_collection("c", vectors_config=models.VectorParams(size=2, distance=models.Distance.DOT))
    ids = [str(uuid.UUID(int=i + 1)) for i in range(6)]
    client.upsert("c", [models.PointStruct(id=i, vector=[1.0, 1.0]) for i in ids])
    prefetch = models.Prefetch(query=[1.0, 1.0], limit=6)
    first = [p.id for p in client.query_points("c", query=[1.0, 1.0], limit=6).points]
    scored = client.query_points("c", prefetch=prefetch, query=models.FormulaQuery(formula="$score"), limit=6)
    print([p.id for p in scored.points] == first)
    print([p.id for p in scored.points])
    """
)

ARRAY_GROUPS = textwrap.dedent(
    """
    from qdrant_client import QdrantClient, models

    client = QdrantClient(":memory:")
    client.create_collection("c", vectors_config=models.VectorParams(size=2, distance=models.Distance.DOT))
    tags = ["red", "green", "blue", "black", "white"]
    client.upsert("c", [models.PointStruct(id=1, vector=[1.0, 1.0], payload={"tags": tags})])
    groups = client.query_points_groups("c", query=[1.0, 1.0], group_by="tags", limit=3, group_size=1).groups
    print([g.id for g in groups])
    """
)


def run_with_hash_seed(code: str, seed: int) -> str:
    env = {**os.environ, "PYTHONHASHSEED": str(seed)}
    done = subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True
    )
    return done.stdout


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_formula_over_one_prefetch_keeps_the_prefetch_order_for_ties(seed):
    assert run_with_hash_seed(FORMULA_TIES, seed).splitlines()[0] == "True"


def test_formula_tie_order_does_not_depend_on_the_hash_seed():
    outputs = {run_with_hash_seed(FORMULA_TIES, seed) for seed in (1, 2, 3)}
    assert len(outputs) == 1


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_groups_of_an_array_field_follow_the_payload_order(seed):
    assert run_with_hash_seed(ARRAY_GROUPS, seed).strip() == "['red', 'green', 'blue']"
