from qdrant_client import models
from qdrant_client.client_base import QdrantBase
from tests.congruence_tests.test_common import (
    COLLECTION_NAME,
    compare_client_results,
    init_client,
    init_local,
    init_remote,
)

# Fixtures below are built so that MMR has to break an *exact* tie, which is where local mode
# used to diverge from core:
#   * core seeds the selection with the most relevant candidate and then picks the best MMR score
#     with Rust's `max_by_key`, which returns the *last* maximum on ties, while `np.argmax`
#     returns the first one;
#   * core holds the pending candidates in an `IndexSet` and drops the selected one with
#     `swap_remove`, which moves the last candidate into the freed slot and therefore changes the
#     order the remaining candidates are visited in.
#
# Relevance scores are kept distinct on purpose: core orders equally relevant candidates by
# whatever order the search returned them in, which is not stable, so a tie in *relevance* can't
# be asserted on. All coordinates are exact binary fractions, so the MMR ties are exact in f32
# both locally and in core.


def _mmr_query(client: QdrantBase, query: models.VectorInput, using: str | None = None) -> list:
    return client.query_points(
        collection_name=COLLECTION_NAME,
        query=models.NearestQuery(nearest=query, mmr=models.Mmr()),
        using=using,
        limit=10,
    ).points


def test_mmr_dense_dot_tie():
    """Two candidates tie on MMR score, with default diversity, DOT distance.

    Query is [1, 0, 0, 0], relevance is therefore the first coordinate:
    ids 2 and 3 both end up with an MMR score of -0.5 once id 1 is selected.
    """
    vectors_config = models.VectorParams(size=4, distance=models.Distance.DOT)
    points = [
        models.PointStruct(id=1, vector=[2.0, 1.0, 0.0, 0.0]),  # relevance 2.0, selected first
        models.PointStruct(id=2, vector=[1.0, 0.0, 0.0, 0.0]),  # relevance 1.0, MMR -0.5
        models.PointStruct(id=3, vector=[0.5, 0.5, 0.0, 0.0]),  # relevance 0.5, MMR -0.5
        models.PointStruct(id=4, vector=[0.25, 1.0, 0.0, 0.0]),  # relevance 0.25, MMR -0.625
    ]

    local_client = init_local()
    init_client(local_client, points, vectors_config=vectors_config)

    remote_client = init_remote()
    init_client(remote_client, points, vectors_config=vectors_config)

    compare_client_results(local_client, remote_client, _mmr_query, query=[1.0, 0.0, 0.0, 0.0])


def test_mmr_dense_euclid_tie():
    """Same tie, but with EUCLID, to show the tie-breaking is not specific to DOT."""
    vectors_config = models.VectorParams(size=2, distance=models.Distance.EUCLID)
    points = [
        models.PointStruct(id=1, vector=[0.25, 0.0]),  # relevance -0.0625, selected first
        models.PointStruct(id=2, vector=[0.5, 0.25]),  # relevance -0.3125, MMR -0.09375
        models.PointStruct(id=3, vector=[0.5, 0.5]),  # relevance -0.5, MMR -0.09375
        models.PointStruct(id=4, vector=[0.75, 0.5]),  # relevance -0.8125, MMR -0.15625
    ]

    local_client = init_local()
    init_client(local_client, points, vectors_config=vectors_config)

    remote_client = init_remote()
    init_client(remote_client, points, vectors_config=vectors_config)

    compare_client_results(local_client, remote_client, _mmr_query, query=[0.0, 0.0])


def test_mmr_multivector_dot_tie():
    """Same tie on a MAX_SIM multivector field, where the divergence was first spotted."""
    vectors_config = {
        "multi": models.VectorParams(
            size=4,
            distance=models.Distance.DOT,
            multivector_config=models.MultiVectorConfig(
                comparator=models.MultiVectorComparator.MAX_SIM
            ),
        )
    }
    points = [
        # the extra vector is never the closest one, it only exercises the MAX_SIM reduction
        models.PointStruct(id=1, vector={"multi": [[2.0, 1.0, 0.0, 0.0], [0.0, 0.0, -1.0, 0.0]]}),
        models.PointStruct(id=2, vector={"multi": [[1.0, 0.0, 0.0, 0.0]]}),
        models.PointStruct(id=3, vector={"multi": [[0.5, 0.5, 0.0, 0.0]]}),
        models.PointStruct(id=4, vector={"multi": [[0.25, 1.0, 0.0, 0.0]]}),
    ]

    local_client = init_local()
    init_client(local_client, points, vectors_config=vectors_config)

    remote_client = init_remote()
    init_client(remote_client, points, vectors_config=vectors_config)

    compare_client_results(
        local_client, remote_client, _mmr_query, query=[[1.0, 0.0, 0.0, 0.0]], using="multi"
    )
