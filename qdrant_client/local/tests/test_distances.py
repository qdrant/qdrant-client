import numpy as np

from qdrant_client.http import models
from qdrant_client.local.distances import calculate_distance
from qdrant_client.local.multi_distances import calculate_multi_distance
from qdrant_client.local.sparse_distances import calculate_distance_sparse


def test_distances() -> None:
    query = np.array([1.0, 2.0, 3.0])
    vectors = np.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
    assert np.allclose(calculate_distance(query, vectors, models.Distance.DOT), [14.0, 14.0])
    assert np.allclose(calculate_distance(query, vectors, models.Distance.EUCLID), [0.0, 0.0])
    assert np.allclose(calculate_distance(query, vectors, models.Distance.MANHATTAN), [0.0, 0.0])
    assert np.allclose(calculate_distance(query, vectors, models.Distance.COSINE), [1.0, 1.0])

    query = np.array([1.0, 0.0, 1.0])
    vectors = np.array([[1.0, 2.0, 3.0], [0.0, 1.0, 0.0]])

    assert np.allclose(
        calculate_distance(query, vectors, models.Distance.DOT), [4.0, 0.0], atol=0.0001
    )
    assert np.allclose(
        calculate_distance(query, vectors, models.Distance.EUCLID),
        [2.82842712, 1.7320508],
        atol=0.0001,
    )

    assert np.allclose(
        calculate_distance(query, vectors, models.Distance.MANHATTAN),
        [4.0, 3.0],
        atol=0.0001,
    )
    assert np.allclose(
        calculate_distance(query, vectors, models.Distance.COSINE),
        [0.75592895, 0.0],
        atol=0.0001,
    )

    sparse_query = models.SparseVector(indices=[1, 2], values=[1, 2])
    sparse_vectors = [models.SparseVector(indices=[10, 20], values=[1, 2])]

    assert calculate_distance_sparse(sparse_query, sparse_vectors) == [np.float32("-inf")]

    sparse_vectors = [
        models.SparseVector(indices=[1, 2], values=[3, 4]),
        models.SparseVector(indices=[1, 2, 3], values=[1, 2, 3]),
    ]
    assert np.allclose(
        calculate_distance_sparse(sparse_query, sparse_vectors), [11.0, 5], atol=0.0001
    )

    multivector_query = np.array([[1, 2, 3], [3, 4, 5]])
    docs = [np.array([[1, 2, 3], [0, 1, 2]])]
    assert calculate_multi_distance(multivector_query, docs, models.Distance.DOT)[0] == 40.0


def test_cosine_does_not_mutate_query() -> None:
    # cosine_similarity used to normalize its `query` argument in place,
    # corrupting caller-owned arrays. `vectors` is intentionally still
    # normalized in place: on cosine collections it is always already
    # unit-normalized through the normal client API (normalized on upsert),
    # so mutating it in place is a no-op there, and it avoids an unnecessary
    # copy of what can be a large candidate set.
    query = np.array([3.0, 4.0], dtype=np.float32)
    vectors = np.array([[6.0, 8.0], [1.0, 0.0]], dtype=np.float32)
    query_snapshot = query.copy()

    result = calculate_distance(query, vectors, models.Distance.COSINE)

    assert np.array_equal(query, query_snapshot), "query must not be mutated"
    # [6, 8] is colinear with the query -> 1.0; [1, 0] -> 3/5 = 0.6
    assert np.allclose(result, [1.0, 0.6], atol=0.0001)

    # 2D (multivector-style) query path must also leave the query untouched
    query_2d = np.array([[3.0, 4.0], [0.0, 5.0]], dtype=np.float32)
    query_2d_snapshot = query_2d.copy()
    vectors_for_2d = np.array([[6.0, 8.0], [1.0, 0.0]], dtype=np.float32)
    result_2d = calculate_distance(query_2d, vectors_for_2d, models.Distance.COSINE)
    assert np.array_equal(query_2d, query_2d_snapshot), "2D query must not be mutated"
    # per-row cosine of [[3, 4], [0, 5]] against [[6, 8], [1, 0]]
    assert np.allclose(result_2d, [[1.0, 0.6], [0.8, 0.0]], atol=0.0001)


def test_cosine_accepts_integer_dtype_query() -> None:
    # In-place normalization of `query` used to raise UFuncTypeError on an
    # integer query array, unlike the dot/euclidean/manhattan distances.
    # `vectors` is still expected to be float, matching how the client always
    # stores cosine vectors (see comment in cosine_similarity).
    query = np.array([3, 4], dtype=np.int64)
    vectors = np.array([[6.0, 8.0], [1.0, 0.0]], dtype=np.float32)
    result = calculate_distance(query, vectors, models.Distance.COSINE)
    assert np.allclose(result, [1.0, 0.6], atol=0.0001)
