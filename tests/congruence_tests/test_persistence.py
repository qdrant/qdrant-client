import tempfile

from tests.congruence_tests.test_common import (
    compare_collections,
    generate_fixtures,
    generate_multivector_fixtures,
    init_client,
    init_local,
    init_remote,
    multi_vector_config,
)


def test_reopened_collection_matches_remote():
    """A collection reopened from disk must still hold what the server holds.

    The other persistence tests here reopen a collection but only compare scored
    search results, which agree within `rel_tol=1e-4` even when the stored
    vectors do not. Comparing the vectors directly catches a reload that fails
    to rebuild them the way the write path stored them.
    """
    points = generate_fixtures()
    with tempfile.TemporaryDirectory() as tmpdir:
        local_client = init_local(tmpdir)
        init_client(local_client, points)
        local_client.close()

        remote_client = init_remote()
        init_client(remote_client, points)

        reopened_client = init_local(tmpdir)
        compare_collections(reopened_client, remote_client, len(points))
        reopened_client.close()


def test_reopened_multivector_collection_matches_remote():
    """Same as above, for multivectors."""
    points = generate_multivector_fixtures()
    with tempfile.TemporaryDirectory() as tmpdir:
        local_client = init_local(tmpdir)
        init_client(local_client, points, vectors_config=multi_vector_config)
        local_client.close()

        remote_client = init_remote()
        init_client(remote_client, points, vectors_config=multi_vector_config)

        reopened_client = init_local(tmpdir)
        compare_collections(reopened_client, remote_client, len(points))
        reopened_client.close()
