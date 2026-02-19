import asyncio
import concurrent.futures
import importlib.metadata
import os
import platform
import time
import uuid
from pprint import pprint
from tempfile import mkdtemp
from time import sleep

import numpy as np
import pytest
from httpx import Timeout
from grpc import Compression, RpcError

import qdrant_client.http.exceptions
from qdrant_client import QdrantClient, models
from qdrant_client._pydantic_compat import to_dict
from qdrant_client.conversions.common_types import PointVectors, StrictModeConfig
from qdrant_client.common.client_exceptions import ResourceExhaustedResponse
from qdrant_client.conversions.conversion import grpc_to_payload, json_to_value
from qdrant_client.local.qdrant_local import QdrantLocal
from qdrant_client.models import (
    Batch,
    CompressionRatio,
    CreateAlias,
    CreateAliasOperation,
    Distance,
    FieldCondition,
    Filter,
    HasIdCondition,
    HnswConfigDiff,
    MatchAny,
    MatchText,
    MatchValue,
    OptimizersConfigDiff,
    PayloadSchemaType,
    PointIdsList,
    PointStruct,
    ProductQuantization,
    ProductQuantizationConfig,
    QuantizationSearchParams,
    Range,
    ScalarQuantization,
    ScalarQuantizationConfig,
    ScalarType,
    SearchParams,
    TextIndexParams,
    TokenizerType,
    VectorParams,
    VectorParamsDiff,
)
from qdrant_client.qdrant_remote import QdrantRemote
from qdrant_client.uploader.grpc_uploader import payload_to_grpc
from tests.congruence_tests.test_common import (
    generate_fixtures,
    init_client,
    init_remote,
    initialize_fixture_collection,
)
from tests.fixtures.payload import (
    one_random_payload_please,
    random_payload,
    random_real_word,
)
from tests.fixtures.points import generate_points
from tests.utils import read_version

DIM = 100
NUM_VECTORS = 1_000
COLLECTION_NAME = "client_test"
COLLECTION_NAME_ALIAS = "client_test_alias"
WRITE_LIMIT = 3
READ_LIMIT = 2


TIMEOUT = 60


def create_random_vectors():
    vectors_path = os.path.join(mkdtemp(), "vectors.npy")
    fp = np.memmap(vectors_path, dtype="float32", mode="w+", shape=(NUM_VECTORS, DIM))

    data = np.random.rand(NUM_VECTORS, DIM).astype(np.float32)
    fp[:] = data[:]
    fp.flush()
    return vectors_path


def test_client_init():
    import tempfile
    import ssl

    client = QdrantClient(":memory:")
    assert isinstance(client._client, QdrantLocal)
    assert client._client.location == ":memory:"

    with tempfile.TemporaryDirectory() as tmpdir:
        client = QdrantClient(path=tmpdir + "/test.db")
        assert isinstance(client._client, QdrantLocal)
        assert client._client.location == tmpdir + "/test.db"

    client = QdrantClient(check_compatibility=False)
    assert isinstance(client._client, QdrantRemote)
    assert client._client.rest_uri == "http://localhost:6333"

    client = QdrantClient(":memory:")
    assert isinstance(client._client, QdrantLocal)

    client = QdrantClient(check_compatibility=False)
    assert isinstance(client._client, QdrantRemote)

    client = QdrantClient(prefer_grpc=True, check_compatibility=False)
    assert isinstance(client._client, QdrantRemote)

    client = QdrantClient(https=True, check_compatibility=False)
    assert isinstance(client._client, QdrantRemote)
    assert client._client.rest_uri == "https://localhost:6333"

    client = QdrantClient(https=True, port=7333, check_compatibility=False)
    assert isinstance(client._client, QdrantRemote)
    assert client._client.rest_uri == "https://localhost:7333"

    client = QdrantClient(host="hidden_port_addr.com", prefix="custom", check_compatibility=False)
    assert isinstance(client._client, QdrantRemote)
    assert client._client.rest_uri == "http://hidden_port_addr.com:6333/custom"

    client = QdrantClient(host="hidden_port_addr.com", port=None, check_compatibility=False)
    assert isinstance(client._client, QdrantRemote)
    assert client._client.rest_uri == "http://hidden_port_addr.com"

    client = QdrantClient(
        host="hidden_port_addr.com",
        port=None,
        prefix="custom",
        check_compatibility=False,
    )
    assert isinstance(client._client, QdrantRemote)
    assert client._client.rest_uri == "http://hidden_port_addr.com/custom"

    client = QdrantClient("http://hidden_port_addr.com", port=None, check_compatibility=False)
    assert isinstance(client._client, QdrantRemote)
    assert client._client.rest_uri == "http://hidden_port_addr.com"

    # url takes precedence over port, which has default value for a backward compatibility
    client = QdrantClient(url="http://localhost:6333", port=7333, check_compatibility=False)
    assert isinstance(client._client, QdrantRemote)
    assert client._client.rest_uri == "http://localhost:6333"

    client = QdrantClient(url="http://localhost:6333", prefix="custom", check_compatibility=False)
    assert isinstance(client._client, QdrantRemote)
    assert client._client.rest_uri == "http://localhost:6333/custom"

    for prefix in ("api/v1", "/api/v1"):
        client = QdrantClient(
            url="http://localhost:6333", prefix=prefix, check_compatibility=False
        )
        assert (
            isinstance(client._client, QdrantRemote)
            and client._client.rest_uri == "http://localhost:6333/api/v1"
        )

        client = QdrantClient(host="localhost", prefix=prefix, check_compatibility=False)
        assert (
            isinstance(client._client, QdrantRemote)
            and client._client.rest_uri == "http://localhost:6333/api/v1"
        )

    for prefix in ("api/v1/", "/api/v1/"):
        client = QdrantClient(
            url="http://localhost:6333", prefix=prefix, check_compatibility=False
        )
        assert (
            isinstance(client._client, QdrantRemote)
            and client._client.rest_uri == "http://localhost:6333/api/v1/"
        )

        client = QdrantClient(host="localhost", prefix=prefix, check_compatibility=False)
        assert (
            isinstance(client._client, QdrantRemote)
            and client._client.rest_uri == "http://localhost:6333/api/v1/"
        )

    client = QdrantClient(url="http://localhost:6333/custom", check_compatibility=False)
    assert isinstance(client._client, QdrantRemote)
    assert client._client.rest_uri == "http://localhost:6333/custom"
    assert client._client._prefix == "/custom"

    client = QdrantClient("my-domain.com", check_compatibility=False)
    assert isinstance(client._client, QdrantRemote)
    assert client._client.rest_uri == "http://my-domain.com:6333"

    client = QdrantClient("my-domain.com:80", check_compatibility=False)
    assert isinstance(client._client, QdrantRemote)
    assert client._client.rest_uri == "http://my-domain.com:80"

    with pytest.raises(ValueError):
        QdrantClient(url="http://localhost:6333", host="localhost", check_compatibility=False)

    with pytest.raises(ValueError):
        QdrantClient(
            url="http://localhost:6333/origin", prefix="custom", check_compatibility=False
        )

    client = QdrantClient("127.0.0.1:6333", check_compatibility=False)
    assert isinstance(client._client, QdrantRemote)
    assert client._client.rest_uri == "http://127.0.0.1:6333"

    client = QdrantClient("localhost:6333", check_compatibility=False)
    assert isinstance(client._client, QdrantRemote)
    assert client._client.rest_uri == "http://localhost:6333"

    client = QdrantClient(":memory:", not_exist_param="test")
    assert isinstance(client._client, QdrantLocal)

    grid_params = [
        {"location": ":memory:", "url": "http://localhost:6333"},
        {"location": ":memory:", "host": "localhost"},
        {"location": ":memory:", "path": "/tmp/test.db"},
        {"url": "http://localhost:6333", "host": "localhost"},
        {"url": "http://localhost:6333", "path": "/tmp/test.db"},
        {"host": "localhost", "path": "/tmp/test.db"},
    ]
    for params in grid_params:
        with pytest.raises(
            ValueError,
            match="Only one of <location>, <url>, <host> or <path> should be specified.",
        ):
            QdrantClient(**params)

    client = QdrantClient(
        url="http://localhost:6333",
        prefix="custom",
        metadata={"some-rest-meta": "some-value"},
        check_compatibility=False,
    )
    assert client.init_options["url"] == "http://localhost:6333"
    assert client.init_options["prefix"] == "custom"
    assert client.init_options["metadata"] == {"some-rest-meta": "some-value"}

    ssl_context = ssl.create_default_context()
    client = QdrantClient(
        ":memory:",
        verify=ssl_context,  # `verify` does not make sense for local client,
        # it's just a mock to check creation of `init_options` with unpickleable objects like ssl context
    )
    assert client.init_options["verify"] is ssl_context


@pytest.mark.parametrize("prefer_grpc", [False, True])
@pytest.mark.parametrize("parallel", [1, 2])
def test_point_upload(prefer_grpc, parallel):
    points = (
        PointStruct(
            id=idx,
            vector=np.random.rand(DIM).tolist(),
            payload=one_random_payload_please(idx),
        )
        for idx in range(NUM_VECTORS)
    )

    client = QdrantClient(prefer_grpc=prefer_grpc, timeout=TIMEOUT)

    if client.collection_exists(COLLECTION_NAME):
        client.delete_collection(collection_name=COLLECTION_NAME, timeout=TIMEOUT)
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=DIM, distance=Distance.DOT),
        timeout=TIMEOUT,
    )

    client.upload_points(collection_name=COLLECTION_NAME, points=points, parallel=parallel)

    # By default, Qdrant indexes data updates asynchronously, so client don't need to wait before sending next batch
    # Let's give it a second to actually add all points to a collection.
    # If you need to change this behaviour - simply enable synchronous processing by enabling `wait=true`
    sleep(1)

    collection_info = client.get_collection(collection_name=COLLECTION_NAME)

    assert collection_info.points_count == NUM_VECTORS

    result_count = client.count(
        COLLECTION_NAME,
        count_filter=Filter(
            must=[
                FieldCondition(
                    key="rand_number",  # Condition based on values of `rand_number` field.
                    range=Range(gte=0.5),  # Select only those results where `rand_number` >= 0.5
                )
            ]
        ),
    )

    assert result_count.count < 900
    assert result_count.count > 100

    client.delete_collection(collection_name=COLLECTION_NAME, timeout=TIMEOUT)
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=DIM, distance=Distance.DOT),
        timeout=TIMEOUT,
    )

    points = (
        PointStruct(id=idx, vector=np.random.rand(DIM).tolist()) for idx in range(NUM_VECTORS)
    )

    client.upload_points(
        collection_name=COLLECTION_NAME, points=points, parallel=parallel, wait=True
    )

    collection_info = client.get_collection(collection_name=COLLECTION_NAME)

    assert collection_info.points_count == NUM_VECTORS


@pytest.mark.parametrize("prefer_grpc", [False, True])
@pytest.mark.parametrize("parallel", [1, 2])
def test_upload_collection(prefer_grpc, parallel):
    size = 3
    batch_size = 2
    client = QdrantClient(prefer_grpc=prefer_grpc, timeout=TIMEOUT)

    if client.collection_exists(COLLECTION_NAME):
        client.delete_collection(collection_name=COLLECTION_NAME, timeout=TIMEOUT)
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=size, distance=Distance.DOT),
        timeout=TIMEOUT,
    )
    vectors = [
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0],
        [10.0, 11.0, 12.0],
        [13.0, 14.0, 15.0],
    ]
    payload = [{"a": 2}, {"b": 3}, {"c": 4}, {"d": 5}, {"e": 6}]
    ids = [1, 2, 3, 4, 5]

    client.upload_collection(
        collection_name=COLLECTION_NAME,
        vectors=vectors,
        parallel=parallel,
        wait=True,
        batch_size=batch_size,
    )

    assert client.get_collection(collection_name=COLLECTION_NAME).points_count == 5

    client.delete_collection(collection_name=COLLECTION_NAME, timeout=TIMEOUT)
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=size, distance=Distance.DOT),
        timeout=TIMEOUT,
    )

    client.upload_collection(
        collection_name=COLLECTION_NAME,
        vectors=vectors,
        payload=payload,
        ids=ids,
        parallel=parallel,
        wait=True,
        batch_size=batch_size,
    )

    assert client.get_collection(collection_name=COLLECTION_NAME).points_count == 5


@pytest.mark.parametrize("prefer_grpc", [False, True])
def test_multiple_vectors(prefer_grpc):
    num_vectors = 100
    points = [
        PointStruct(
            id=idx,
            vector={
                "image": np.random.rand(DIM).tolist(),
                "text": np.random.rand(DIM * 2).tolist(),
            },
            payload=one_random_payload_please(idx),
        )
        for idx in range(num_vectors)
    ]

    client = QdrantClient(prefer_grpc=prefer_grpc, timeout=TIMEOUT)
    if client.collection_exists(COLLECTION_NAME):
        client.delete_collection(collection_name=COLLECTION_NAME, timeout=TIMEOUT)
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config={
            "image": VectorParams(size=DIM, distance=Distance.DOT),
            "text": VectorParams(size=DIM * 2, distance=Distance.COSINE),
        },
        timeout=TIMEOUT,
    )

    client.upload_points(collection_name=COLLECTION_NAME, points=points, parallel=1)

    query_vector = list(np.random.rand(DIM))

    hits = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        using="image",
        with_vectors=True,
        limit=5,  # Return 5 closest points
    ).points

    assert len(hits) == 5
    assert "image" in hits[0].vector
    assert "text" in hits[0].vector

    hits = client.query_points(
        collection_name=COLLECTION_NAME,
        using="text",
        query=query_vector * 2,
        with_vectors=True,
        limit=5,  # Return 5 closest points
    ).points

    assert len(hits) == 5
    assert "image" in hits[0].vector
    assert "text" in hits[0].vector


@pytest.mark.parametrize("prefer_grpc", [False, True])
@pytest.mark.parametrize("numpy_upload", [False, True])
@pytest.mark.parametrize("local_mode", [False, True])
def test_qdrant_client_integration(prefer_grpc, numpy_upload, local_mode):
    vectors_path = create_random_vectors()

    if numpy_upload:
        vectors = np.memmap(vectors_path, dtype="float32", mode="r", shape=(NUM_VECTORS, DIM))
        vectors_2 = vectors[2].tolist()
    else:
        vectors = [np.random.rand(DIM).tolist() for _ in range(NUM_VECTORS)]
        vectors_2 = vectors[2]

    payload = random_payload(NUM_VECTORS)

    if local_mode:
        client = QdrantClient(location=":memory:", prefer_grpc=prefer_grpc)
    else:
        client = QdrantClient(prefer_grpc=prefer_grpc, timeout=TIMEOUT)
    if client.collection_exists(COLLECTION_NAME):
        client.delete_collection(collection_name=COLLECTION_NAME, timeout=TIMEOUT)
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=DIM, distance=Distance.DOT),
        timeout=TIMEOUT,
    )

    assert client.collection_exists(collection_name=COLLECTION_NAME)
    assert not client.collection_exists(collection_name="non_existing_collection")

    # Call Qdrant API to retrieve list of existing collections
    collections = client.get_collections().collections

    # Print all existing collections
    for collection in collections:
        print(to_dict(collection))

    # Retrieve detailed information about newly created collection
    test_collection = client.get_collection(COLLECTION_NAME)
    pprint(to_dict(test_collection))

    # Upload data to a new collection
    client.upload_collection(
        collection_name=COLLECTION_NAME,
        vectors=vectors,
        payload=payload,
        ids=range(len(vectors)),
        parallel=2,
    )

    # By default, Qdrant indexes data updates asynchronously, so client don't need to wait before sending next batch
    # Let's give it a second to actually add all points to a collection.
    # If you need to change this behaviour - simply enable synchronous processing by enabling `wait=true`
    sleep(1)

    result_count = client.count(
        COLLECTION_NAME,
        count_filter=Filter(
            must=[
                FieldCondition(
                    key="rand_number",  # Condition based on values of `rand_number` field.
                    range=Range(gte=0.5),  # Select only those results where `rand_number` >= 0.5
                )
            ]
        ),
    )

    assert result_count.count < 900
    assert result_count.count > 100

    client.update_collection_aliases(
        change_aliases_operations=[
            CreateAliasOperation(
                create_alias=CreateAlias(
                    collection_name=COLLECTION_NAME, alias_name=COLLECTION_NAME_ALIAS
                )
            )
        ]
    )

    collection_aliases = client.get_collection_aliases(COLLECTION_NAME)

    assert collection_aliases.aliases[0].collection_name == COLLECTION_NAME
    assert collection_aliases.aliases[0].alias_name == COLLECTION_NAME_ALIAS

    all_aliases = client.get_aliases()

    assert all_aliases.aliases[0].collection_name == COLLECTION_NAME
    assert all_aliases.aliases[0].alias_name == COLLECTION_NAME_ALIAS

    # Create payload index for field `rand_number`
    # If indexed field appear in filtering condition - search operation could be performed faster
    index_create_result = client.create_payload_index(
        COLLECTION_NAME, field_name="rand_number", field_schema=PayloadSchemaType.FLOAT
    )
    pprint(to_dict(index_create_result))
    # Again, with string field
    index_create_result = client.create_payload_index(
        COLLECTION_NAME, field_name="rand_number", field_schema="float"
    )
    pprint(to_dict(index_create_result))

    # Let's now check details about our new collection
    test_collection = client.get_collection(COLLECTION_NAME_ALIAS)
    pprint(to_dict(test_collection))

    # Now we can actually search in the collection
    # Let's create some random vector
    query_vector = np.random.rand(DIM)
    query_vector_1: list[float] = list(np.random.rand(DIM))
    query_vector_2: list[float] = list(np.random.rand(DIM))
    query_vector_3: list[float] = list(np.random.rand(DIM))

    #  and use it as a query
    hits = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        query_filter=None,  # Don't use any filters for now, search across all indexed points
        with_payload=True,  # Also return a stored payload for found points
        limit=5,  # Return 5 closest points
    ).points

    assert len(hits) == 5

    # Print found results
    print("Search result:")
    for hit in hits:
        print(hit)

    client.create_payload_index(COLLECTION_NAME, "id_str", field_schema=PayloadSchemaType.KEYWORD)
    #  and use it as a query
    hits = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        query_filter=Filter(must=[FieldCondition(key="id_str", match=MatchValue(value="11"))]),
        with_payload=True,
        limit=5,
    ).points

    assert "11" in hits[0].payload["id_str"]

    hits_should = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        query_filter=Filter(
            should=[
                FieldCondition(key="id_str", match=MatchValue(value="10")),
                FieldCondition(key="id_str", match=MatchValue(value="11")),
            ]
        ),
        with_payload=True,
        limit=5,
    ).points

    hits_match_any = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        query_filter=Filter(
            must=[
                FieldCondition(
                    key="id_str",
                    match=MatchAny(any=["10", "11"]),
                )
            ]
        ),
        with_payload=True,
        limit=5,
    ).points
    assert hits_should == hits_match_any

    hits_min_should = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        query_filter=Filter(
            min_should=models.MinShould(
                conditions=[
                    FieldCondition(key="id_str", match=MatchValue(value="11")),
                    FieldCondition(key="rand_digit", match=MatchAny(any=list(range(10)))),
                    FieldCondition(key="id", match=MatchAny(any=list(range(100, 150)))),
                ],
                min_count=2,
            )
        ),
        with_payload=True,
        limit=5,
    ).points
    assert len(hits_min_should) > 0

    hits_min_should_empty = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        query_filter=Filter(
            min_should=models.MinShould(
                conditions=[
                    FieldCondition(key="id_str", match=MatchValue(value="11")),
                ],
                min_count=2,
            )
        ),
        with_payload=True,
        limit=5,
    ).points
    assert len(hits_min_should_empty) == 0

    # Let's now query same vector with filter condition
    hits = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        query_filter=Filter(
            must=[  # These conditions are required for search results
                FieldCondition(
                    key="rand_number",  # Condition based on values of `rand_number` field.
                    range=Range(gte=0.5),  # Select only those results where `rand_number` >= 0.5
                )
            ]
        ),
        with_payload=True,
        limit=5,  # Return 5 closest points
    ).points

    print("Filtered search result (`rand_number` >= 0.5):")
    for hit in hits:
        print(hit)

    got_points = client.retrieve(
        collection_name=COLLECTION_NAME,
        ids=[1, 2, 3],
        with_payload=True,
        with_vectors=True,
    )

    # ------------------ Test for full-text filtering ------------------

    # Create index for full-text search
    client.create_payload_index(
        COLLECTION_NAME,
        "words",
        field_schema=TextIndexParams(
            type="text",
            tokenizer=TokenizerType.WORD,
            min_token_len=2,
            max_token_len=15,
            lowercase=True,
        ),
    )

    for i in range(10):
        query_word = random_real_word()
        hits, _offset = client.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=Filter(
                must=[FieldCondition(key="words", match=MatchText(text=query_word))]
            ),
            with_payload=True,
            limit=10,
        )

        assert len(hits) > 0

        for hit in hits:
            assert query_word in hit.payload["words"]

    # ------------------  Test for batch queries ------------------
    filter_1 = Filter(must=[FieldCondition(key="rand_number", range=Range(gte=0.3))])
    filter_2 = Filter(must=[FieldCondition(key="rand_number", range=Range(gte=0.5))])
    filter_3 = Filter(must=[FieldCondition(key="rand_number", range=Range(gte=0.7))])

    query_points_requests = [
        models.QueryRequest(
            query=query_vector_1,
            filter=filter_1,
            limit=5,
            with_payload=True,
        ),
        models.QueryRequest(
            query=query_vector_2,
            filter=filter_2,
            limit=5,
            with_payload=True,
        ),
        models.QueryRequest(
            query=query_vector_3,
            filter=filter_3,
            limit=5,
            with_payload=True,
        ),
    ]
    single_query_result_1 = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector_1,
        query_filter=filter_1,
        limit=5,
    )
    single_query_result_2 = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector_2,
        query_filter=filter_2,
        limit=5,
    )
    single_query_result_3 = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector_3,
        query_filter=filter_3,
        limit=5,
    )

    batch_query_result = client.query_batch_points(
        collection_name=COLLECTION_NAME, requests=query_points_requests
    )

    assert len(batch_query_result) == 3
    assert batch_query_result[0] == single_query_result_1
    assert batch_query_result[1] == single_query_result_2
    assert batch_query_result[2] == single_query_result_3

    # ------------------  End of batch queries test ----------------

    assert len(got_points) == 3

    client.delete(
        collection_name=COLLECTION_NAME,
        wait=True,
        points_selector=PointIdsList(points=[2, 3]),
    )

    got_points = client.retrieve(
        collection_name=COLLECTION_NAME,
        ids=[1, 2, 3],
        with_payload=True,
        with_vectors=True,
    )

    assert len(got_points) == 1

    client.upsert(
        collection_name=COLLECTION_NAME,
        wait=True,
        points=[PointStruct(id=2, payload={"hello": "world"}, vector=vectors_2)],
    )

    got_points = client.retrieve(
        collection_name=COLLECTION_NAME,
        ids=[1, 2, 3],
        with_payload=True,
        with_vectors=True,
    )

    assert len(got_points) == 2

    client.set_payload(
        collection_name=COLLECTION_NAME,
        payload={"new_key": 123},
        points=[1, 2],
        wait=True,
    )

    got_points = client.retrieve(
        collection_name=COLLECTION_NAME,
        ids=[1, 2],
        with_payload=True,
        with_vectors=True,
    )

    for point in got_points:
        assert point.payload.get("new_key") == 123

    client.delete_payload(
        collection_name=COLLECTION_NAME,
        keys=["new_key"],
        points=[1],
    )

    got_points = client.retrieve(
        collection_name=COLLECTION_NAME, ids=[1], with_payload=True, with_vectors=True
    )

    for point in got_points:
        assert "new_key" not in point.payload

    client.clear_payload(
        collection_name=COLLECTION_NAME,
        points_selector=PointIdsList(points=[1, 2]),
    )

    got_points = client.retrieve(
        collection_name=COLLECTION_NAME,
        ids=[1, 2],
        with_payload=True,
        with_vectors=True,
    )

    for point in got_points:
        assert not point.payload

    positive = [1, 2, query_vector.tolist()]
    negative = []

    recommended_points = client.query_points(
        collection_name=COLLECTION_NAME,
        query=models.RecommendQuery(
            recommend=models.RecommendInput(
                positive=positive,
                negative=negative,
            ),
        ),
        query_filter=Filter(
            must=[  # These conditions are required for recommend results
                FieldCondition(
                    key="rand_number",  # Condition based on values of `rand_number` field.
                    range=Range(lte=0.5),  # Select only those results where `rand_number` >= 0.5
                )
            ]
        ),
        limit=5,
        with_payload=True,
        with_vectors=False,
    ).points

    assert len(recommended_points) == 5

    scrolled_points, next_page = client.scroll(
        collection_name=COLLECTION_NAME,
        scroll_filter=Filter(
            must=[  # These conditions are required for scroll results
                FieldCondition(
                    key="rand_number",  # Condition based on values of `rand_number` field.
                    range=Range(lte=0.5),  # Return only those results where `rand_number` <= 0.5
                )
            ]
        ),
        limit=5,
        offset=None,
        with_payload=True,
        with_vectors=False,
    )

    assert isinstance(next_page, (int, str))

    assert len(scrolled_points) == 5

    _, next_page = client.scroll(
        collection_name=COLLECTION_NAME,
        scroll_filter=Filter(
            must=[  # These conditions are required for scroll results
                FieldCondition(
                    key="rand_number",  # Condition based on values of `rand_number` field.
                    range=Range(lte=0.5),  # Return only those results where `rand_number` <= 0.5
                )
            ]
        ),
        limit=1000,
        offset=None,
        with_payload=True,
        with_vectors=False,
    )

    assert next_page is None

    client.batch_update_points(
        collection_name=COLLECTION_NAME,
        ordering=models.WriteOrdering.STRONG,
        update_operations=[
            models.UpsertOperation(
                upsert=models.PointsList(
                    points=[
                        models.PointStruct(
                            id=1,
                            payload={"new_key": 123},
                            vector=vectors_2,
                        ),
                        models.PointStruct(
                            id=2,
                            payload={"new_key": 321},
                            vector=vectors_2,
                        ),
                    ]
                )
            ),
            models.DeleteOperation(delete=models.PointIdsList(points=[2])),
            models.SetPayloadOperation(
                set_payload=models.SetPayload(payload={"new_key2": 321}, points=[1])
            ),
            models.OverwritePayloadOperation(
                overwrite_payload=models.SetPayload(
                    payload={
                        "new_key3": 321,
                        "new_key4": 321,
                    },
                    points=[1],
                )
            ),
            models.DeletePayloadOperation(
                delete_payload=models.DeletePayload(keys=["new_key3"], points=[1])
            ),
            models.ClearPayloadOperation(clear_payload=models.PointIdsList(points=[1])),
            models.UpdateVectorsOperation(
                update_vectors=models.UpdateVectors(
                    points=[
                        models.PointVectors(
                            id=1,
                            vector=vectors_2,
                        )
                    ]
                )
            ),
            models.DeleteVectorsOperation(
                delete_vectors=models.DeleteVectors(points=[1], vector=[""])
            ),
        ],
    )


@pytest.mark.parametrize("prefer_grpc", [False, True])
def test_qdrant_client_integration_update_collection(prefer_grpc):
    client = QdrantClient(prefer_grpc=prefer_grpc, timeout=TIMEOUT)
    if client.collection_exists(COLLECTION_NAME):
        client.delete_collection(collection_name=COLLECTION_NAME, timeout=TIMEOUT)
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config={
            "text": VectorParams(size=DIM, distance=Distance.DOT),
        },
        timeout=TIMEOUT,
    )

    client.update_collection(
        collection_name=COLLECTION_NAME,
        vectors_config={
            "text": VectorParamsDiff(
                hnsw_config=HnswConfigDiff(
                    m=32,
                    ef_construct=123,
                ),
                quantization_config=ProductQuantization(
                    product=ProductQuantizationConfig(
                        compression=CompressionRatio.X32,
                        always_ram=True,
                    ),
                ),
                on_disk=True,
            ),
        },
        hnsw_config=HnswConfigDiff(
            ef_construct=123,
        ),
        quantization_config=ScalarQuantization(
            scalar=ScalarQuantizationConfig(
                type=ScalarType.INT8,
                quantile=0.8,
                always_ram=False,
            ),
        ),
        optimizers_config=OptimizersConfigDiff(max_segment_size=10000),
    )

    collection_info = client.get_collection(COLLECTION_NAME)

    assert collection_info.config.params.vectors["text"].hnsw_config.m == 32
    assert collection_info.config.params.vectors["text"].hnsw_config.ef_construct == 123
    assert (
        collection_info.config.params.vectors["text"].quantization_config.product.compression
        == CompressionRatio.X32
    )
    assert collection_info.config.params.vectors["text"].quantization_config.product.always_ram
    assert collection_info.config.params.vectors["text"].on_disk
    assert collection_info.config.hnsw_config.ef_construct == 123
    assert collection_info.config.quantization_config.scalar.type == ScalarType.INT8
    assert 0.7999 < collection_info.config.quantization_config.scalar.quantile < 0.8001
    assert not collection_info.config.quantization_config.scalar.always_ram

    assert collection_info.config.optimizer_config.max_segment_size == 10000


@pytest.mark.parametrize("prefer_grpc", [False, True])
def test_points_crud(prefer_grpc):
    client = QdrantClient(prefer_grpc=prefer_grpc, timeout=TIMEOUT)
    if client.collection_exists(COLLECTION_NAME):
        client.delete_collection(collection_name=COLLECTION_NAME, timeout=TIMEOUT)
    collection_params = dict(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=DIM, distance=Distance.DOT),
        timeout=TIMEOUT,
    )

    major, minor, patch, dev = read_version()
    if not dev and None not in (major, minor, patch) and (major, minor, patch) < (1, 16, 0):
        client.create_collection(**collection_params)
    else:
        collection_metadata = {"ownership": "Bart Simpson's property"}
        collection_params["metadata"] = collection_metadata  # type: ignore
        client.create_collection(**collection_params)
        collection_info = client.get_collection(COLLECTION_NAME)
        assert collection_info.config.metadata == collection_metadata

        new_metadata = {"due_date": "12.12.2222"}
        client.update_collection(COLLECTION_NAME, metadata=new_metadata)
        updated_collection_info = client.get_collection(COLLECTION_NAME)
        assert updated_collection_info.config.metadata == {**collection_metadata, **new_metadata}

    # Create a single point
    client.upsert(
        collection_name=COLLECTION_NAME,
        points=[
            PointStruct(id=123, payload={"test": "value"}, vector=np.random.rand(DIM).tolist())
        ],
        wait=True,
    )

    client.upsert(
        collection_name=COLLECTION_NAME,
        points=Batch(
            ids=[3, 4],
            vectors=[np.random.rand(DIM).tolist(), np.random.rand(DIM).tolist()],
            payloads=[
                {"test": "value", "test2": "value2"},
                {"test": "value", "test2": {"haha": "???"}},
            ],
        ),
    )

    # Read a single point

    points = client.retrieve(COLLECTION_NAME, ids=[123])

    print("read a single point", points)

    # Update a single point

    client.set_payload(
        collection_name=COLLECTION_NAME,
        payload={"test2": ["value2", "value3"]},
        points=[123],
    )

    # Delete a single point
    client.delete(collection_name=COLLECTION_NAME, points_selector=PointIdsList(points=[123]))


@pytest.mark.parametrize("prefer_grpc", [False, True])
def test_quantization_config(prefer_grpc):
    client = QdrantClient(prefer_grpc=prefer_grpc, timeout=TIMEOUT)
    if client.collection_exists(COLLECTION_NAME):
        client.delete_collection(collection_name=COLLECTION_NAME, timeout=TIMEOUT)
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=DIM, distance=Distance.DOT),
        quantization_config=ScalarQuantization(
            scalar=ScalarQuantizationConfig(
                type=ScalarType.INT8,
                quantile=1.0,
                always_ram=True,
            ),
        ),
        timeout=TIMEOUT,
    )

    client.upsert(
        collection_name=COLLECTION_NAME,
        points=[
            PointStruct(id=2001, vector=np.random.rand(DIM).tolist()),
            PointStruct(id=2002, vector=np.random.rand(DIM).tolist()),
            PointStruct(id=2003, vector=np.random.rand(DIM).tolist()),
            PointStruct(id=2004, vector=np.random.rand(DIM).tolist()),
        ],
        wait=True,
    )

    collection_info = client.get_collection(COLLECTION_NAME)

    quantization_config = collection_info.config.quantization_config

    assert isinstance(quantization_config, ScalarQuantization)
    assert quantization_config.scalar.type == ScalarType.INT8
    assert quantization_config.scalar.quantile == 1.0
    assert quantization_config.scalar.always_ram is True

    _res = client.query_points(
        collection_name=COLLECTION_NAME,
        query=np.random.rand(DIM),
        search_params=SearchParams(
            quantization=QuantizationSearchParams(
                rescore=True,
            )
        ),
    )


@pytest.mark.parametrize("prefer_grpc", [False, True])
def test_custom_sharding(prefer_grpc):
    client = QdrantClient(prefer_grpc=prefer_grpc, timeout=TIMEOUT)

    def init_collection():
        if client.collection_exists(COLLECTION_NAME):
            client.delete_collection(collection_name=COLLECTION_NAME)
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=DIM, distance=Distance.DOT),
            sharding_method=models.ShardingMethod.CUSTOM,
        )

        client.create_shard_key(collection_name=COLLECTION_NAME, shard_key=cats_shard_key)
        client.create_shard_key(collection_name=COLLECTION_NAME, shard_key=dogs_shard_key)
        major, minor, patch, dev = read_version()
        if major is None or dev or (major, minor, patch) >= (1, 16, 0):
            fish_shard_key = "fish"
            client.create_shard_key(
                collection_name=COLLECTION_NAME,
                shard_key=fish_shard_key,
                initial_state=models.ReplicaState.ACTIVE,
            )
            print("created shard key with replica state")

    cat_ids = [1, 2, 3]
    cat_vectors = [np.random.rand(DIM).tolist() for _ in range(len(cat_ids))]
    cat_payload = [{"name": "Barsik"}, {"name": "Murzik"}, {"name": "Chubais"}]
    cats_shard_key = "cats"

    dog_ids = [4, 5, 6]
    dog_vectors = [np.random.rand(DIM).tolist() for _ in range(len(dog_ids))]
    dog_payload = [{"name": "Sharik"}, {"name": "Tuzik"}, {"name": "Bobik"}]
    dogs_shard_key = "dogs"

    cat_points = [
        PointStruct(id=id_, vector=vector, payload=payload)
        for id_, vector, payload in zip(cat_ids, cat_vectors, cat_payload)
    ]
    dog_points = [
        PointStruct(id=id_, vector=vector, payload=payload)
        for id_, vector, payload in zip(dog_ids, dog_vectors, dog_payload)
    ]

    # region upsert
    init_collection()

    client.upsert(
        collection_name=COLLECTION_NAME,
        points=cat_points,
        shard_key_selector=cats_shard_key,
    )

    client.upsert(
        collection_name=COLLECTION_NAME,
        points=dog_points,
        shard_key_selector=dogs_shard_key,
    )

    query_vector = np.random.rand(DIM)
    res = client.query_points(
        collection_name=COLLECTION_NAME, query=query_vector, shard_key_selector=cats_shard_key
    ).points
    assert len(res) == 3
    for record in res:
        assert record.shard_key == cats_shard_key

    query_vector = np.random.rand(DIM)
    res = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        shard_key_selector=[cats_shard_key, dogs_shard_key],
    ).points
    assert len(res) == 6

    query_vector = np.random.rand(DIM)
    res = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
    ).points
    assert len(res) == 6
    # endregion

    # region upload_collection
    init_collection()

    client.upload_collection(
        collection_name=COLLECTION_NAME,
        vectors=cat_vectors,
        ids=cat_ids,
        payload=cat_payload,
        shard_key_selector=cats_shard_key,
    )

    query_vector = np.random.rand(DIM)
    res = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        shard_key_selector=cats_shard_key,
    ).points
    assert len(res) == 3

    for record in res:
        assert record.shard_key == cats_shard_key

    # endregion

    # region upload_points
    init_collection()

    cat_points = [
        PointStruct(id=id_, vector=vector, payload=payload)
        for id_, vector, payload in zip(cat_ids, cat_vectors, cat_payload)
    ]

    client.upload_points(
        collection_name=COLLECTION_NAME,
        points=cat_points,
        shard_key_selector=cats_shard_key,
    )

    query_vector = np.random.rand(DIM)

    res = client.query_points(
        collection_name=COLLECTION_NAME, query=query_vector, shard_key_selector=cats_shard_key
    ).points
    assert len(res) == 3

    query_vector = np.random.rand(DIM)
    res = client.query_points(
        collection_name=COLLECTION_NAME, query=query_vector, shard_key_selector=dogs_shard_key
    ).points
    assert len(res) == 0

    # endregion

    client.delete_shard_key(collection_name=COLLECTION_NAME, shard_key=dogs_shard_key)


@pytest.mark.parametrize("prefer_grpc", [False, True])
def test_sparse_vectors(prefer_grpc):
    client = QdrantClient(prefer_grpc=prefer_grpc, timeout=TIMEOUT)
    if client.collection_exists(COLLECTION_NAME):
        client.delete_collection(collection_name=COLLECTION_NAME)
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config={},
        sparse_vectors_config={
            "text": models.SparseVectorParams(
                index=models.SparseIndexParams(
                    on_disk=False,
                    full_scan_threshold=100,
                )
            )
        },
    )

    client.upsert(
        collection_name=COLLECTION_NAME,
        points=[
            models.PointStruct(
                id=1,
                vector={
                    "text": models.SparseVector(
                        indices=[1, 2, 3],
                        values=[1.0, 2.0, 3.0],
                    )
                },
            ),
            models.PointStruct(
                id=2,
                vector={
                    "text": models.SparseVector(
                        indices=[3, 4, 5],
                        values=[1.0, 2.0, 3.0],
                    )
                },
            ),
            models.PointStruct(
                id=3,
                vector={
                    "text": models.SparseVector(
                        indices=[5, 6, 7],
                        values=[1.0, 2.0, 3.0],
                    )
                },
            ),
        ],
    )

    result = client.query_points(
        collection_name=COLLECTION_NAME,
        using="text",
        query=models.SparseVector(indices=[1, 7], values=[2.0, 1.0]),
        with_vectors=["text"],
    ).points

    assert len(result) == 2
    assert result[0].id == 3
    assert result[1].id == 1

    assert result[0].score == 3.0
    assert result[1].score == 2.0

    assert result[0].vector["text"].indices == [5, 6, 7]
    assert result[0].vector["text"].values == [1.0, 2.0, 3.0]
    assert result[1].vector["text"].indices == [1, 2, 3]
    assert result[1].vector["text"].values == [1.0, 2.0, 3.0]


@pytest.mark.parametrize("prefer_grpc", [False, True])
def test_sparse_vectors_batch(prefer_grpc):
    client = QdrantClient(prefer_grpc=prefer_grpc, timeout=TIMEOUT)
    if client.collection_exists(COLLECTION_NAME):
        client.delete_collection(collection_name=COLLECTION_NAME)
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config={},
        sparse_vectors_config={
            "text": models.SparseVectorParams(
                index=models.SparseIndexParams(
                    on_disk=False,
                    full_scan_threshold=100,
                )
            )
        },
    )

    client.upsert(
        collection_name=COLLECTION_NAME,
        points=[
            models.PointStruct(
                id=1,
                vector={
                    "text": models.SparseVector(
                        indices=[1, 2, 3],
                        values=[1.0, 2.0, 3.0],
                    )
                },
            ),
            models.PointStruct(
                id=2,
                vector={
                    "text": models.SparseVector(
                        indices=[3, 4, 5],
                        values=[1.0, 2.0, 3.0],
                    )
                },
            ),
            models.PointStruct(
                id=3,
                vector={
                    "text": models.SparseVector(
                        indices=[5, 6, 7],
                        values=[1.0, 2.0, 3.0],
                    )
                },
            ),
        ],
    )

    request = models.QueryRequest(
        query=models.SparseVector(
            indices=[1, 7],
            values=[2.0, 1.0],
        ),
        using="text",
        limit=3,
        with_vector=["text"],
    )

    results = client.query_batch_points(
        collection_name=COLLECTION_NAME,
        requests=[request],
    )

    result = results[0].points

    assert len(result) == 2
    assert result[0].id == 3
    assert result[1].id == 1

    assert result[0].score == 3.0
    assert result[1].score == 2.0

    assert result[0].vector["text"].indices == [5, 6, 7]
    assert result[0].vector["text"].values == [1.0, 2.0, 3.0]
    assert result[1].vector["text"].indices == [1, 2, 3]
    assert result[1].vector["text"].values == [1.0, 2.0, 3.0]


@pytest.mark.parametrize("prefer_grpc", [False, True])
def test_vector_update(prefer_grpc):
    client = QdrantClient(prefer_grpc=prefer_grpc, timeout=TIMEOUT)
    if client.collection_exists(COLLECTION_NAME):
        client.delete_collection(collection_name=COLLECTION_NAME, timeout=TIMEOUT)
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=DIM, distance=Distance.DOT),
        timeout=TIMEOUT,
    )

    uuid1 = str(uuid.uuid4())
    uuid2 = str(uuid.uuid4())
    uuid3 = str(uuid.uuid4())
    uuid4 = str(uuid.uuid4())

    client.upsert(
        collection_name=COLLECTION_NAME,
        points=[
            PointStruct(id=uuid1, payload={"a": 1}, vector=np.random.rand(DIM).tolist()),
            PointStruct(id=uuid2, payload={"a": 2}, vector=np.random.rand(DIM).tolist()),
            PointStruct(id=uuid3, payload={"b": 1}, vector=np.random.rand(DIM).tolist()),
            PointStruct(id=uuid4, payload={"b": 2}, vector=np.random.rand(DIM).tolist()),
        ],
        wait=True,
    )

    client.update_vectors(
        collection_name=COLLECTION_NAME,
        points=[
            PointVectors(
                id=uuid2,
                vector=[1.0] * DIM,
            )
        ],
    )

    result = client.retrieve(
        collection_name=COLLECTION_NAME,
        ids=[uuid2],
        with_vectors=True,
    )[0]

    assert result.vector == [1] * DIM

    client.delete_vectors(
        collection_name=COLLECTION_NAME,
        vectors=[""],
        points=Filter(must=[FieldCondition(key="b", range=Range(gte=1))]),
    )

    result = client.retrieve(
        collection_name=COLLECTION_NAME,
        ids=[uuid4],
        with_vectors=True,
    )[0]

    assert result.vector == {}


@pytest.mark.parametrize("prefer_grpc", [False, True])
def test_conditional_payload_update(prefer_grpc):
    client = QdrantClient(prefer_grpc=prefer_grpc, timeout=TIMEOUT)
    if client.collection_exists(COLLECTION_NAME):
        client.delete_collection(collection_name=COLLECTION_NAME, timeout=TIMEOUT)
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=DIM, distance=Distance.DOT),
        timeout=TIMEOUT,
    )

    uuid1 = str(uuid.uuid4())
    uuid2 = str(uuid.uuid4())
    uuid3 = str(uuid.uuid4())
    uuid4 = str(uuid.uuid4())

    client.upsert(
        collection_name=COLLECTION_NAME,
        points=[
            PointStruct(id=uuid1, payload={"a": 1}, vector=np.random.rand(DIM).tolist()),
            PointStruct(id=uuid2, payload={"a": 2}, vector=np.random.rand(DIM).tolist()),
            PointStruct(id=uuid3, payload={"b": 1}, vector=np.random.rand(DIM).tolist()),
            PointStruct(id=uuid4, payload={"b": 2}, vector=np.random.rand(DIM).tolist()),
        ],
        wait=True,
    )

    res = client.retrieve(
        collection_name=COLLECTION_NAME,
        ids=[uuid1, uuid2, uuid4],
    )

    assert len(res) == 3
    retrieved_ids = [uuid.UUID(point.id) for point in res]

    assert uuid.UUID(uuid1) in retrieved_ids
    assert uuid.UUID(uuid2) in retrieved_ids
    assert uuid.UUID(uuid4) in retrieved_ids


@pytest.mark.parametrize("prefer_grpc", [False, True])
def test_conditional_payload_update_2(prefer_grpc):
    client = QdrantClient(prefer_grpc=prefer_grpc, timeout=TIMEOUT)
    if client.collection_exists(COLLECTION_NAME):
        client.delete_collection(collection_name=COLLECTION_NAME, timeout=TIMEOUT)
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=DIM, distance=Distance.DOT),
        timeout=TIMEOUT,
    )

    client.upsert(
        collection_name=COLLECTION_NAME,
        points=[
            PointStruct(id=1001, payload={"a": 1}, vector=np.random.rand(DIM).tolist()),
            PointStruct(id=1002, payload={"a": 2}, vector=np.random.rand(DIM).tolist()),
            PointStruct(id=1003, payload={"b": 1}, vector=np.random.rand(DIM).tolist()),
            PointStruct(id=1004, payload={"b": 2}, vector=np.random.rand(DIM).tolist()),
        ],
        wait=True,
    )

    client.set_payload(
        collection_name=COLLECTION_NAME,
        payload={"c": 1},
        points=Filter(must=[FieldCondition(key="a", range=Range(gte=1))]),
        wait=True,
    )

    points = client.retrieve(
        collection_name=COLLECTION_NAME,
        ids=[1001, 1002, 1003, 1004],
        with_payload=True,
        with_vectors=False,
    )

    points = sorted(points, key=lambda p: p.id)

    assert points[0].payload.get("c") == 1
    assert points[1].payload.get("c") == 1
    assert points[2].payload.get("c") is None
    assert points[3].payload.get("c") is None

    client.overwrite_payload(
        collection_name=COLLECTION_NAME,
        payload={"c": 2},
        points=Filter(must=[FieldCondition(key="b", range=Range(lt=10))]),
    )

    points = client.retrieve(
        collection_name=COLLECTION_NAME,
        ids=[1001, 1002, 1003, 1004],
        with_payload=True,
        with_vectors=False,
    )
    points = sorted(points, key=lambda p: p.id)

    assert points[0].payload.get("c") == 1
    assert points[1].payload.get("c") == 1
    assert points[2].payload == {"c": 2}
    assert points[3].payload == {"c": 2}


def test_has_id_condition():
    query = to_dict(
        Filter(
            must=[
                HasIdCondition(has_id=[42, 43]),
                FieldCondition(key="field_name", match=MatchValue(value="field_value_42")),
            ]
        )
    )

    assert query["must"][0]["has_id"] == [42, 43]


def test_insert_float():
    point = PointStruct(id=123, payload={"value": 0.123}, vector=np.random.rand(DIM).tolist())

    assert isinstance(point.payload["value"], float)


@pytest.mark.parametrize("prefer_grpc", [False, True])
def test_empty_vector(prefer_grpc):
    client = QdrantClient(prefer_grpc=prefer_grpc, timeout=TIMEOUT)
    if client.collection_exists(COLLECTION_NAME):
        client.delete_collection(collection_name=COLLECTION_NAME, timeout=TIMEOUT)
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config={},
        timeout=TIMEOUT,
    )

    client.upsert(
        collection_name=COLLECTION_NAME,
        points=[
            PointStruct(id=123, payload={"test": "value"}, vector={}),
        ],
    )


def test_value_serialization():
    v = json_to_value(123)
    print(v)


def test_serialization():
    from qdrant_client.grpc import PointId as PointIdGrpc
    from qdrant_client.grpc import PointStruct as PointStructGrpc
    from qdrant_client.grpc import Vector, Vectors

    point = PointStructGrpc(
        id=PointIdGrpc(num=1),
        vectors=Vectors(vector=Vector(data=[1.0, 2.0, 3.0, 4.0])),
        payload=payload_to_grpc(
            {
                "a": 123,
                "b": "text",
                "c": [1, 2, 3],
                "d": {
                    "val1": "val2",
                    "val2": [1, 2, 3],
                    "val3": [],
                    "val4": {},
                },
                "e": True,
                "f": None,
            }
        ),
    )
    print("\n")
    print(point.payload)
    data = point.SerializeToString()
    res = PointStructGrpc()
    res.ParseFromString(data)
    print(res.payload)
    print(grpc_to_payload(res.payload))


def test_client_close():
    import tempfile

    from qdrant_client.http import exceptions as qdrant_exceptions

    # region http
    client_http = QdrantClient(timeout=TIMEOUT)
    if client_http.collection_exists("test"):
        client_http.delete_collection("test")
    client_http.create_collection(
        "test", vectors_config=VectorParams(size=100, distance=Distance.COSINE)
    )
    client_http.close()
    with pytest.raises(qdrant_exceptions.ResponseHandlingException):
        if client_http.collection_exists("test"):
            client_http.delete_collection("test")
        client_http.create_collection(
            "test", vectors_config=VectorParams(size=100, distance=Distance.COSINE)
        )
    # endregion

    # region grpc
    client_grpc = QdrantClient(prefer_grpc=True, timeout=TIMEOUT)
    if client_grpc.collection_exists("test"):
        client_grpc.delete_collection("test")
    client_grpc.create_collection(
        "test", vectors_config=VectorParams(size=100, distance=Distance.COSINE)
    )
    client_grpc.close()
    with pytest.raises(ValueError):
        client_grpc.get_collection("test")
    with pytest.raises(
        RuntimeError
    ):  # prevent reinitializing grpc connection, since http connection is closed
        client_grpc._client._init_grpc_channel()

    client_grpc_do_nothing = QdrantClient(
        prefer_grpc=True, timeout=TIMEOUT
    )  # do not establish a connection
    client_grpc_do_nothing.close()
    with pytest.raises(
        RuntimeError
    ):  # prevent initializing grpc connection, since http connection is closed
        _ = client_grpc_do_nothing.get_collection("test")
    # endregion grpc

    # region local
    local_client_in_mem = QdrantClient(":memory:")
    if local_client_in_mem.collection_exists("test"):
        local_client_in_mem.delete_collection("test")
    local_client_in_mem.create_collection(
        "test", vectors_config=VectorParams(size=100, distance=Distance.COSINE)
    )
    local_client_in_mem.close()
    assert local_client_in_mem._client.closed is True

    with pytest.raises(RuntimeError):
        local_client_in_mem.upsert(
            "test", [PointStruct(id=1, vector=np.random.rand(100).tolist())]
        )

    with pytest.raises(RuntimeError):
        if not local_client_in_mem.collection_exists("test"):
            local_client_in_mem.create_collection(
                "test", vectors_config=VectorParams(size=100, distance=Distance.COSINE)
            )

    with pytest.raises(RuntimeError):
        if local_client_in_mem.collection_exists("test"):
            local_client_in_mem.delete_collection("test")

    with tempfile.TemporaryDirectory() as tmpdir:
        path = tmpdir + "/test.db"

        local_client_persist_1 = QdrantClient(path=path)
        if local_client_persist_1.collection_exists("test"):
            local_client_persist_1.delete_collection("test")
        local_client_persist_1.create_collection(
            "test", vectors_config=VectorParams(size=100, distance=Distance.COSINE)
        )
        local_client_persist_1.close()

        local_client_persist_2 = QdrantClient(path=path)
        if local_client_persist_2.collection_exists("test"):
            local_client_persist_2.delete_collection("test")
        local_client_persist_2.create_collection(
            "test", vectors_config=VectorParams(size=100, distance=Distance.COSINE)
        )
        local_client_persist_2.close()
    # endregion local


def test_timeout_propagation():
    client = QdrantClient()
    vectors_config = models.VectorParams(size=2, distance=models.Distance.COSINE)
    with pytest.raises(
        qdrant_client.http.exceptions.ResponseHandlingException, match=r"timed out"
    ):
        # timeout is Optional[int]
        # if we set it to 0 - recreate_collection raises operation is in progress instead of timed out
        client.http.client._client._timeout = Timeout(0.01)
        if client.collection_exists(COLLECTION_NAME):
            client.delete_collection(collection_name=COLLECTION_NAME)
        client.create_collection(collection_name=COLLECTION_NAME, vectors_config=vectors_config)
    sleep(0.5)
    if client.collection_exists(COLLECTION_NAME):
        client.delete_collection(collection_name=COLLECTION_NAME, timeout=10)
    client.create_collection(
        collection_name=COLLECTION_NAME, vectors_config=vectors_config, timeout=10
    )


def test_grpc_options():
    client_version = importlib.metadata.version("qdrant-client")
    user_agent = f"python-client/{client_version}"
    python_version = f"python/{platform.python_version()}"

    client = QdrantClient(prefer_grpc=True)
    assert client._client._grpc_options == {
        "grpc.primary_user_agent": f"{user_agent} {python_version}"
    }

    client = QdrantClient(prefer_grpc=True, grpc_options={"grpc.max_send_message_length": 3})
    assert client._client._grpc_options == {
        "grpc.max_send_message_length": 3,
        "grpc.primary_user_agent": f"{user_agent} {python_version}",
    }

    with pytest.raises(RpcError):
        if not client.collection_exists("grpc_collection"):
            client.create_collection(
                "grpc_collection",
                vectors_config=models.VectorParams(size=100, distance=models.Distance.COSINE),
            )


def test_grpc_compression():
    client = QdrantClient(prefer_grpc=True, grpc_compression=Compression.Gzip)
    client.get_collections()

    client = QdrantClient(prefer_grpc=True, grpc_compression=Compression.NoCompression)
    client.get_collections()

    with pytest.raises(ValueError):
        # creates a grpc client with not supported Compression type
        QdrantClient(prefer_grpc=True, grpc_compression=Compression.Deflate)

    with pytest.raises(TypeError):
        QdrantClient(prefer_grpc=True, grpc_compression="gzip")


def test_auth_token_provider():
    """Check that the token provided is called for both http and grpc clients."""
    token = ""
    call_num = 0

    def auth_token_provider():
        nonlocal token
        nonlocal call_num

        token = f"token_{call_num}"
        call_num += 1
        return token

    # Additional sync request is sent during client init to check compatibility
    client = QdrantClient(auth_token_provider=auth_token_provider)
    client.get_collections()
    assert token == "token_1"
    client.get_collections()
    assert token == "token_2"

    token = ""
    call_num = 0

    client = QdrantClient(check_compatibility=False, auth_token_provider=auth_token_provider)
    client.get_collections()
    assert token == "token_0"
    client.get_collections()
    assert token == "token_1"

    token = ""
    call_num = 0

    # Additional sync http request is sent during client init to check compatibility
    client = QdrantClient(prefer_grpc=True, auth_token_provider=auth_token_provider)
    client.get_collections()
    assert token == "token_1"
    client.get_collections()
    assert token == "token_2"

    client.get_collections()
    assert token == "token_3"

    token = ""
    call_num = 0

    client = QdrantClient(
        prefer_grpc=True, check_compatibility=False, auth_token_provider=auth_token_provider
    )
    client.get_collections()
    assert token == "token_0"
    client.get_collections()
    assert token == "token_1"

    client.get_collections()
    assert token == "token_2"


def test_async_auth_token_provider():
    """Check that initialization fails if async auth_token_provider is provided to sync client."""
    token = ""

    async def auth_token_provider():
        nonlocal token
        await asyncio.sleep(0.1)
        token = "test_token"
        return token

    client = QdrantClient(auth_token_provider=auth_token_provider)

    with pytest.raises(
        qdrant_client.http.exceptions.ResponseHandlingException,
        match="Synchronous token provider is not set.",
    ):
        client.get_collections()

    assert token == ""

    client = QdrantClient(auth_token_provider=auth_token_provider, prefer_grpc=True)
    with pytest.raises(
        ValueError, match="Synchronous channel requires synchronous auth token provider."
    ):
        client.get_collections()

    assert token == ""


@pytest.mark.parametrize("prefer_grpc", [True, False])
def test_read_consistency(prefer_grpc):
    fixture_points = generate_fixtures(vectors_sizes=DIM, num=NUM_VECTORS)
    client = init_remote(prefer_grpc=prefer_grpc)
    init_client(
        client,
        fixture_points,
        collection_name=COLLECTION_NAME,
        vectors_config=models.VectorParams(size=DIM, distance=models.Distance.DOT),
    )

    query_vector = fixture_points[0].vector

    client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=5,  # Return 5 closest points
        consistency=models.ReadConsistencyType.MAJORITY,
    )

    client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=5,  # Return 5 closest points
        consistency=models.ReadConsistencyType.MAJORITY,
    )

    client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=5,  # Return 5 closest points
        consistency=2,
    )

    query_requests = [models.QueryRequest(query=query_vector, limit=5)]
    client.query_batch_points(
        collection_name=COLLECTION_NAME,
        requests=query_requests,
    )

    client.query_batch_points(
        collection_name=COLLECTION_NAME,
        requests=query_requests,
        consistency=models.ReadConsistencyType.MAJORITY,
    )

    client.query_batch_points(
        collection_name=COLLECTION_NAME, requests=query_requests, consistency=2
    )

    client.query_points_groups(
        collection_name=COLLECTION_NAME,
        group_by="word",
        query=query_vector,
        limit=5,  # Return 5 closest points
        consistency=models.ReadConsistencyType.MAJORITY,
    )

    client.query_points_groups(
        collection_name=COLLECTION_NAME,
        group_by="word",
        query=query_vector,
        limit=5,  # Return 5 closest points
        consistency=models.ReadConsistencyType.MAJORITY,
    )

    client.query_points_groups(
        collection_name=COLLECTION_NAME,
        group_by="word",
        query=query_vector,
        limit=5,  # Return 5 closest points
        consistency=models.ReadConsistencyType.MAJORITY,
    )


@pytest.mark.parametrize("prefer_grpc", (False, True))
def test_create_payload_index(prefer_grpc):
    client = init_remote(prefer_grpc=prefer_grpc)
    initialize_fixture_collection(client, COLLECTION_NAME, vectors_config={})

    client.create_payload_index(
        COLLECTION_NAME, "keyword", models.PayloadSchemaType.KEYWORD, wait=True
    )
    client.create_payload_index(
        COLLECTION_NAME, "integer", models.PayloadSchemaType.INTEGER, wait=True
    )
    client.create_payload_index(
        COLLECTION_NAME, "float", models.PayloadSchemaType.FLOAT, wait=True
    )
    client.create_payload_index(COLLECTION_NAME, "geo", models.PayloadSchemaType.GEO, wait=True)
    client.create_payload_index(COLLECTION_NAME, "text", models.PayloadSchemaType.TEXT, wait=True)
    client.create_payload_index(COLLECTION_NAME, "bool", models.PayloadSchemaType.BOOL, wait=True)
    client.create_payload_index(
        COLLECTION_NAME, "datetime", models.PayloadSchemaType.DATETIME, wait=True
    )

    client.create_payload_index(
        COLLECTION_NAME,
        "text_parametrized",
        models.TextIndexParams(
            type=models.TextIndexType.TEXT,
            tokenizer=models.TokenizerType.PREFIX,
            min_token_len=3,
            max_token_len=7,
            lowercase=True,
        ),
        wait=True,
    )

    client.create_payload_index(COLLECTION_NAME, "uuid", models.PayloadSchemaType.UUID, wait=True)

    client.create_payload_index(
        COLLECTION_NAME,
        "keyword_parametrized",
        models.KeywordIndexParams(
            type=models.KeywordIndexType.KEYWORD, is_tenant=False, on_disk=True
        ),
        wait=True,
    )
    payload_schema = client.get_collection(COLLECTION_NAME).payload_schema
    assert payload_schema["keyword_parametrized"].params.is_tenant is False
    assert payload_schema["keyword_parametrized"].params.on_disk is True

    client.create_payload_index(
        COLLECTION_NAME,
        "integer_parametrized",
        models.IntegerIndexParams(
            type=models.IntegerIndexType.INTEGER,
            lookup=True,
            range=False,
            is_principal=False,
            on_disk=True,
        ),
        wait=True,
    )
    if prefer_grpc:
        rest_client = QdrantClient()
        _ = rest_client.get_collection(COLLECTION_NAME).payload_schema
    payload_schema = client.get_collection(COLLECTION_NAME).payload_schema
    assert payload_schema["integer_parametrized"].params.lookup is True
    assert payload_schema["integer_parametrized"].params.range is False
    assert payload_schema["integer_parametrized"].params.is_principal is False
    assert payload_schema["integer_parametrized"].params.on_disk is True

    client.create_payload_index(
        COLLECTION_NAME,
        "float_parametrized",
        models.FloatIndexParams(
            type=models.FloatIndexType.FLOAT, is_principal=False, on_disk=True
        ),
        wait=True,
    )

    client.create_payload_index(
        COLLECTION_NAME,
        "datetime_parametrized",
        models.DatetimeIndexParams(
            type=models.DatetimeIndexType.DATETIME, is_principal=False, on_disk=True
        ),
        wait=True,
    )
    client.create_payload_index(
        COLLECTION_NAME,
        "uuid_parametrized",
        models.UuidIndexParams(type=models.UuidIndexType.UUID, is_tenant=False, on_disk=True),
        wait=True,
    )

    client.create_payload_index(
        COLLECTION_NAME,
        "geo_parametrized",
        models.GeoIndexParams(type=models.GeoIndexType.GEO),
        wait=True,
    )
    client.create_payload_index(
        COLLECTION_NAME,
        "bool_parametrized",
        models.BoolIndexParams(type=models.BoolIndexType.BOOL),
        wait=True,
    )


@pytest.mark.parametrize("prefer_grpc", (False, True))
def test_strict_mode(prefer_grpc):
    major, minor, patch, dev = read_version()
    if not (major is None or dev):
        if (major, minor, patch) < (1, 13, 0):
            pytest.skip("Strict mode is supported as of qdrant 1.13.0")

    client = init_remote(prefer_grpc=prefer_grpc)
    initialize_fixture_collection(client, COLLECTION_NAME, vectors_config={})
    strict_mode_config = StrictModeConfig(
        enabled=True,
        max_query_limit=150,
    )
    client.update_collection(COLLECTION_NAME, strict_mode_config=strict_mode_config)
    collection_info = client.get_collection(COLLECTION_NAME)
    strict_mode_config = collection_info.config.strict_mode_config
    assert strict_mode_config.enabled is True
    assert strict_mode_config.max_query_limit == 150

    if major is None or dev or (major, minor, patch) >= (1, 14, 0):
        strict_mode_config = StrictModeConfig(
            max_points_count=100,
        )
        client.update_collection(COLLECTION_NAME, strict_mode_config=strict_mode_config)
        collection_info = client.get_collection(COLLECTION_NAME)
        strict_mode_config = collection_info.config.strict_mode_config
        assert strict_mode_config.max_points_count == 100


@pytest.mark.parametrize("prefer_grpc", [False, True])
def test_upsert_hits_large_request_limit(prefer_grpc):
    major, minor, patch, dev = read_version()

    client = QdrantClient(prefer_grpc=prefer_grpc, timeout=TIMEOUT)
    if client.collection_exists(COLLECTION_NAME):
        client.delete_collection(collection_name=COLLECTION_NAME, timeout=TIMEOUT)
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=models.VectorParams(size=DIM, distance=models.Distance.DOT),
        timeout=TIMEOUT,
        strict_mode_config=models.StrictModeConfig(
            enabled=True, read_rate_limit=READ_LIMIT, write_rate_limit=WRITE_LIMIT
        ),
    )

    points = generate_points(num_points=100, vector_sizes=DIM)

    if dev or None in (major, minor, patch) or (major, minor, patch) >= (1, 13, 0):
        if prefer_grpc:
            exception_class = RpcError
        else:
            exception_class = qdrant_client.http.exceptions.UnexpectedResponse

        with pytest.raises(
            exception_class,
            match="Write rate limit exceeded",
        ):
            client.upsert(COLLECTION_NAME, points)
    else:
        client.upsert(COLLECTION_NAME, points)


@pytest.mark.parametrize("prefer_grpc", [False, True])
def test_upsert_hits_write_rate_limit(prefer_grpc):
    major, minor, patch, dev = read_version()

    client = QdrantClient(prefer_grpc=prefer_grpc, timeout=TIMEOUT)
    if client.collection_exists(COLLECTION_NAME):
        client.delete_collection(collection_name=COLLECTION_NAME, timeout=TIMEOUT)
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=models.VectorParams(size=DIM, distance=models.Distance.DOT),
        timeout=TIMEOUT,
    )

    client.update_collection(
        collection_name=COLLECTION_NAME,
        strict_mode_config=models.StrictModeConfig(
            enabled=True, read_rate_limit=READ_LIMIT, write_rate_limit=WRITE_LIMIT
        ),
    )  # there is a bug in core in v1.12.6 which ignores the value set in write_rate_limit and assigns read_rate_limit
    # value to both rate limits

    points = generate_points(num_points=WRITE_LIMIT, vector_sizes=DIM)

    if dev or None in (major, minor, patch) or (major, minor, patch) >= (1, 13, 0):
        exception_class = ResourceExhaustedResponse
    else:
        exception_class = (
            RpcError if prefer_grpc else qdrant_client.http.exceptions.UnexpectedResponse
        )

    with pytest.raises(exception_class):
        try:
            for _ in range(WRITE_LIMIT + 1):
                client.upsert(collection_name=COLLECTION_NAME, points=points)
        except Exception as e:
            raise e


@pytest.mark.parametrize("prefer_grpc", [False, True])
def test_query_hits_read_rate_limit(prefer_grpc):
    major, minor, patch, dev = read_version()

    client = QdrantClient(prefer_grpc=prefer_grpc, timeout=TIMEOUT)
    if client.collection_exists(COLLECTION_NAME):
        client.delete_collection(collection_name=COLLECTION_NAME, timeout=TIMEOUT)
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=models.VectorParams(size=DIM, distance=models.Distance.DOT),
        timeout=TIMEOUT,
        strict_mode_config=models.StrictModeConfig(
            enabled=True, read_rate_limit=READ_LIMIT, write_rate_limit=WRITE_LIMIT
        ),
    )

    dense_vector_query_batch_text = [
        models.QueryRequest(
            query=np.random.random(DIM).tolist(),
            prefetch=models.Prefetch(query=np.random.random(DIM).tolist(), limit=5),
            limit=5,
            with_payload=True,
        )
        for _ in range(READ_LIMIT)
    ]

    if dev or None in (major, minor, patch) or (major, minor, patch) >= (1, 13, 0):
        exception_class = ResourceExhaustedResponse
    else:
        exception_class = (
            RpcError if prefer_grpc else qdrant_client.http.exceptions.UnexpectedResponse
        )

    with pytest.raises(exception_class):
        for _ in range(READ_LIMIT + 1):
            client.query_batch_points(
                collection_name=COLLECTION_NAME, requests=dense_vector_query_batch_text
            )


@pytest.mark.parametrize("prefer_grpc", [False, True])
def test_upload_collection_succeeds_with_limits(prefer_grpc, mocker):
    major, minor, patch, dev = read_version()

    client = QdrantClient(prefer_grpc=prefer_grpc, timeout=TIMEOUT)
    if client.collection_exists(COLLECTION_NAME):
        client.delete_collection(collection_name=COLLECTION_NAME, timeout=TIMEOUT)
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=models.VectorParams(size=DIM, distance=models.Distance.DOT),
        timeout=TIMEOUT,
        strict_mode_config=models.StrictModeConfig(
            enabled=True, read_rate_limit=READ_LIMIT, write_rate_limit=WRITE_LIMIT
        ),
    )
    # pre-condition: hit the limit first then do upload_collection
    points = generate_points(num_points=WRITE_LIMIT, vector_sizes=DIM)

    try:
        for _ in range(10):
            client.upsert(COLLECTION_NAME, points)
    except Exception as ex:
        pass
    # end of pre-condition

    if dev or None in (major, minor, patch) or (major, minor, patch) >= (1, 13, 0):
        if prefer_grpc:
            mock = mocker.patch(
                "qdrant_client.grpc.points_pb2.UpsertPoints",
                side_effect=ResourceExhaustedResponse("test too many resources", retry_after_s=1),
            )
        else:
            mock = mocker.patch(
                "qdrant_client.http.api.points_api.SyncPointsApi.upsert_points",
                side_effect=ResourceExhaustedResponse("test too many resources", retry_after_s=1),
            )

        def update_collection():
            time.sleep(2)
            client.update_collection(
                collection_name=COLLECTION_NAME,
                strict_mode_config=models.StrictModeConfig(enabled=False),
            )
            mock.side_effect = None

        def run_upload_points():
            client.upload_points(COLLECTION_NAME, points=points, wait=True, max_retries=1)

            results = client.scroll(
                collection_name=COLLECTION_NAME,
                with_vectors=False,
                with_payload=False,
            )
            result = results[0]
            assert len(result) == WRITE_LIMIT

        with concurrent.futures.ThreadPoolExecutor() as executor:
            future1 = executor.submit(update_collection)
            future2 = executor.submit(run_upload_points)
            concurrent.futures.wait([future1, future2])
    else:
        if prefer_grpc:
            exception_class = RpcError
        else:
            exception_class = qdrant_client.http.exceptions.UnexpectedResponse

        with pytest.raises(exception_class):
            client.upload_points(COLLECTION_NAME, points=points, wait=True, max_retries=1)


@pytest.mark.parametrize("prefer_grpc", [False, True])
def test_cluster_collection_update(prefer_grpc):
    major, minor, patch, dev = read_version()
    if not (major is None or dev):
        if (major, minor, patch) < (1, 16, 0):
            pytest.skip("Cluster collection update is supported as of qdrant 1.16.0")

    client = QdrantClient(prefer_grpc=prefer_grpc)
    if client.collection_exists(COLLECTION_NAME):
        client.delete_collection(COLLECTION_NAME, timeout=TIMEOUT)
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=models.VectorParams(size=DIM, distance=models.Distance.DOT),
        timeout=TIMEOUT,
        sharding_method=models.ShardingMethod.CUSTOM,
    )

    client.cluster_collection_update(
        COLLECTION_NAME,
        cluster_operation=models.CreateShardingKeyOperation(
            create_sharding_key=models.CreateShardingKey(
                shard_key="fish",
                shards_number=1,
            )
        ),
    )

    client.cluster_collection_update(
        COLLECTION_NAME,
        cluster_operation=models.CreateShardingKeyOperation(
            create_sharding_key=models.CreateShardingKey(
                shard_key="lion",
                shards_number=1,
                initial_state=models.ReplicaState.PARTIAL,
            )
        ),
    )

    client.upsert(
        COLLECTION_NAME,
        points=[models.PointStruct(id=1, vector={}), models.PointStruct(id=2, vector={})],
        shard_key_selector="fish",
    )

    fallback_shard_key = models.ShardKeyWithFallback(target="lion", fallback="fish")

    client.upsert(
        COLLECTION_NAME,
        points=[models.PointStruct(id=3, vector={})],
        shard_key_selector=fallback_shard_key,
    )
    assert len(client.scroll(COLLECTION_NAME, shard_key_selector=fallback_shard_key)[0]) > 0

    client.cluster_collection_update(
        collection_name=COLLECTION_NAME,
        cluster_operation=models.ReplicatePointsOperation(
            replicate_points=models.ReplicatePoints(
                from_shard_key="fish",
                to_shard_key="lion",
                filter=models.Filter(must=models.HasIdCondition(has_id=[1])),
            )
        ),
    )

    client.cluster_collection_update(
        COLLECTION_NAME,
        cluster_operation=models.DropShardingKeyOperation(
            drop_sharding_key=models.DropShardingKey(shard_key="fish")
        ),
    )


@pytest.mark.parametrize("prefer_grpc", [False, True])
def test_cluster_methods(prefer_grpc):
    major, minor, patch, dev = read_version()
    if not (major is None or dev):
        if (major, minor, patch) < (1, 16, 0):
            pytest.skip("Cluster collection update is supported as of qdrant 1.16.0")

    client = QdrantClient(prefer_grpc=prefer_grpc)
    if client.collection_exists(COLLECTION_NAME):
        client.delete_collection(COLLECTION_NAME, timeout=TIMEOUT)
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=models.VectorParams(size=DIM, distance=models.Distance.DOT),
        timeout=TIMEOUT,
    )
    client.upsert(
        COLLECTION_NAME, points=[models.PointStruct(id=2, vector=np.random.rand(DIM).tolist())]
    )
    cluster_info = client.collection_cluster_info(collection_name=COLLECTION_NAME)
    assert cluster_info.shard_count == 1
    assert len(cluster_info.local_shards) == 1
    assert cluster_info.remote_shards == []
    assert cluster_info.shard_transfers == []

    client.recover_current_peer()

    cluster_status = client.cluster_status()
    assert cluster_status.status == "enabled"


@pytest.mark.parametrize("prefer_grpc", [False, True])
def test_create_payload_index_enable_hnsw(prefer_grpc: bool):
    major, minor, patch, dev = read_version()
    if not (major is None or dev):
        if (major, minor, patch) < (1, 17, 0):
            pytest.skip("enable_hnsw is supported as of Qdrant 1.17.0")

    client = QdrantClient(prefer_grpc=prefer_grpc)
    if client.collection_exists(COLLECTION_NAME):
        client.delete_collection(COLLECTION_NAME, timeout=TIMEOUT)
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=models.VectorParams(size=DIM, distance=models.Distance.DOT),
        timeout=TIMEOUT,
    )

    client.create_payload_index(
        COLLECTION_NAME,
        field_name="uuid_field",
        field_schema=models.UuidIndexParams(
            type=models.UuidIndexType.UUID,
            enable_hnsw=False,
        ),
    )

    client.create_payload_index(
        COLLECTION_NAME,
        field_name="bool_field",
        field_schema=models.BoolIndexParams(
            type=models.BoolIndexType.BOOL,
            enable_hnsw=False,
        ),
    )

    client.create_payload_index(
        COLLECTION_NAME,
        field_name="text_field",
        field_schema=models.TextIndexParams(
            type=models.TextIndexType.TEXT,
            tokenizer=models.TokenizerType.WHITESPACE,
            enable_hnsw=False,
        ),
    )

    client.create_payload_index(
        COLLECTION_NAME,
        field_name="keyword_field",
        field_schema=models.KeywordIndexParams(
            type=models.KeywordIndexType.KEYWORD,
            enable_hnsw=False,
        ),
    )

    client.create_payload_index(
        COLLECTION_NAME,
        field_name="integer_field",
        field_schema=models.IntegerIndexParams(
            type=models.IntegerIndexType.INTEGER,
            enable_hnsw=False,
        ),
    )

    client.create_payload_index(
        COLLECTION_NAME,
        field_name="geo_field",
        field_schema=models.GeoIndexParams(
            type=models.GeoIndexType.GEO,
            enable_hnsw=False,
        ),
    )

    client.create_payload_index(
        COLLECTION_NAME,
        field_name="float_field",
        field_schema=models.FloatIndexParams(
            type=models.FloatIndexType.FLOAT,
            enable_hnsw=False,
        ),
    )

    client.create_payload_index(
        COLLECTION_NAME,
        field_name="datetime_field",
        field_schema=models.DatetimeIndexParams(
            type=models.DatetimeIndexType.DATETIME,
            enable_hnsw=False,
        ),
    )

    info = client.get_collection(COLLECTION_NAME)
    for field_name in [
        "uuid_field",
        "bool_field",
        "text_field",
        "keyword_field",
        "integer_field",
        "geo_field",
        "float_field",
        "datetime_field",
    ]:
        assert info.payload_schema[field_name].params.enable_hnsw is False, field_name
