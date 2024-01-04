import os
import uuid
from pprint import pprint
from tempfile import mkdtemp
from time import sleep
from typing import List

import numpy as np
import pytest
from grpc import RpcError

from qdrant_client import QdrantClient, models
from qdrant_client._pydantic_compat import to_dict
from qdrant_client.conversions.common_types import PointVectors, Record
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
    RecommendRequest,
    ScalarQuantization,
    ScalarQuantizationConfig,
    ScalarType,
    SearchParams,
    SearchRequest,
    TextIndexParams,
    TokenizerType,
    VectorParams,
    VectorParamsDiff,
)
from qdrant_client.qdrant_remote import QdrantRemote
from qdrant_client.uploader.grpc_uploader import payload_to_grpc
from tests.fixtures.payload import (
    one_random_payload_please,
    random_payload,
    random_real_word,
)

DIM = 100
NUM_VECTORS = 1_000
COLLECTION_NAME = "client_test"
COLLECTION_NAME_ALIAS = "client_test_alias"


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

    client = QdrantClient(":memory:")
    assert isinstance(client._client, QdrantLocal)
    assert client._client.location == ":memory:"

    with tempfile.TemporaryDirectory() as tmpdir:
        client = QdrantClient(path=tmpdir + "/test.db")
        assert isinstance(client._client, QdrantLocal)
        assert client._client.location == tmpdir + "/test.db"

    client = QdrantClient()
    assert isinstance(client._client, QdrantRemote)
    assert client._client.rest_uri == "http://localhost:6333"

    client = QdrantClient(https=True)
    assert isinstance(client._client, QdrantRemote)
    assert client._client.rest_uri == "https://localhost:6333"

    client = QdrantClient(https=True, port=7333)
    assert isinstance(client._client, QdrantRemote)
    assert client._client.rest_uri == "https://localhost:7333"

    client = QdrantClient(host="hidden_port_addr.com", prefix="custom")
    assert isinstance(client._client, QdrantRemote)
    assert client._client.rest_uri == "http://hidden_port_addr.com:6333/custom"

    client = QdrantClient(host="hidden_port_addr.com", port=None)
    assert isinstance(client._client, QdrantRemote)
    assert client._client.rest_uri == "http://hidden_port_addr.com"

    client = QdrantClient(
        host="hidden_port_addr.com",
        port=None,
        prefix="custom",
    )
    assert isinstance(client._client, QdrantRemote)
    assert client._client.rest_uri == "http://hidden_port_addr.com/custom"

    client = QdrantClient("http://hidden_port_addr.com", port=None)
    assert isinstance(client._client, QdrantRemote)
    assert client._client.rest_uri == "http://hidden_port_addr.com"

    # url takes precedence over port, which has default value for a backward compatibility
    client = QdrantClient(url="http://localhost:6333", port=7333)
    assert isinstance(client._client, QdrantRemote)
    assert client._client.rest_uri == "http://localhost:6333"

    client = QdrantClient(url="http://localhost:6333", prefix="custom")
    assert isinstance(client._client, QdrantRemote)
    assert client._client.rest_uri == "http://localhost:6333/custom"

    client = QdrantClient("my-domain.com")
    assert isinstance(client._client, QdrantRemote)
    assert client._client.rest_uri == "http://my-domain.com:6333"

    client = QdrantClient("my-domain.com:80")
    assert isinstance(client._client, QdrantRemote)
    assert client._client.rest_uri == "http://my-domain.com:80"

    with pytest.raises(ValueError):
        QdrantClient(url="http://localhost:6333", host="localhost")

    with pytest.raises(ValueError):
        QdrantClient(url="http://localhost:6333/origin", prefix="custom")

    client = QdrantClient("127.0.0.1:6333")
    assert isinstance(client._client, QdrantRemote)
    assert client._client.rest_uri == "http://127.0.0.1:6333"

    client = QdrantClient("localhost:6333")
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


@pytest.mark.parametrize("prefer_grpc", [False, True])
def test_record_upload(prefer_grpc):
    records = (
        Record(id=idx, vector=np.random.rand(DIM).tolist(), payload=one_random_payload_please(idx))
        for idx in range(NUM_VECTORS)
    )

    client = QdrantClient(prefer_grpc=prefer_grpc, timeout=TIMEOUT)

    client.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=DIM, distance=Distance.DOT),
        timeout=TIMEOUT,
    )

    client.upload_records(collection_name=COLLECTION_NAME, records=records, parallel=2)

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


@pytest.mark.parametrize("prefer_grpc", [False, True])
def test_multiple_vectors(prefer_grpc):
    num_vectors = 100
    records = [
        Record(
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

    client.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config={
            "image": VectorParams(size=DIM, distance=Distance.DOT),
            "text": VectorParams(size=DIM * 2, distance=Distance.COSINE),
        },
        timeout=TIMEOUT,
    )

    client.upload_records(collection_name=COLLECTION_NAME, records=records, parallel=1)

    query_vector = list(np.random.rand(DIM))

    hits = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=("image", query_vector),
        with_vectors=True,
        limit=5,  # Return 5 closest points
    )

    assert len(hits) == 5
    assert "image" in hits[0].vector
    assert "text" in hits[0].vector

    hits = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=("text", query_vector * 2),
        with_vectors=True,
        limit=5,  # Return 5 closest points
    )

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

    client.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=DIM, distance=Distance.DOT),
        timeout=TIMEOUT,
    )

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
        ids=None,  # Let client auto-assign sequential ids
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

    version = os.getenv("QDRANT_VERSION")

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

    # Let's now check details about our new collection
    test_collection = client.get_collection(COLLECTION_NAME_ALIAS)
    pprint(to_dict(test_collection))

    # Now we can actually search in the collection
    # Let's create some random vector
    query_vector = np.random.rand(DIM)
    query_vector_1: List[float] = list(np.random.rand(DIM))
    query_vector_2: List[float] = list(np.random.rand(DIM))
    query_vector_3: List[float] = list(np.random.rand(DIM))

    #  and use it as a query
    hits = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vector,
        query_filter=None,  # Don't use any filters for now, search across all indexed points
        with_payload=True,  # Also return a stored payload for found points
        limit=5,  # Return 5 closest points
    )

    assert len(hits) == 5

    # Print found results
    print("Search result:")
    for hit in hits:
        print(hit)

    client.create_payload_index(COLLECTION_NAME, "id_str", field_schema=PayloadSchemaType.KEYWORD)
    #  and use it as a query
    hits = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vector,
        query_filter=Filter(must=[FieldCondition(key="id_str", match=MatchValue(value="11"))]),
        with_payload=True,
        limit=5,
    )

    assert "11" in hits[0].payload["id_str"]

    hits_should = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vector,
        query_filter=Filter(
            should=[
                FieldCondition(key="id_str", match=MatchValue(value="10")),
                FieldCondition(key="id_str", match=MatchValue(value="11")),
            ]
        ),
        with_payload=True,
        limit=5,
    )

    hits_match_any = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vector,
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
    )

    assert hits_should == hits_match_any

    # Let's now query same vector with filter condition
    hits = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vector,
        query_filter=Filter(
            must=[  # These conditions are required for search results
                FieldCondition(
                    key="rand_number",  # Condition based on values of `rand_number` field.
                    range=Range(gte=0.5),  # Select only those results where `rand_number` >= 0.5
                )
            ]
        ),
        append_payload=True,  # Also return a stored payload for found points
        limit=5,  # Return 5 closest points
    )

    print("Filtered search result (`rand_number` >= 0.5):")
    for hit in hits:
        print(hit)

    got_points = client.retrieve(
        collection_name=COLLECTION_NAME, ids=[1, 2, 3], with_payload=True, with_vectors=True
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

    search_queries = [
        SearchRequest(
            vector=query_vector_1,
            filter=filter_1,
            limit=5,
            with_payload=True,
        ),
        SearchRequest(
            vector=query_vector_2,
            filter=filter_2,
            limit=5,
            with_payload=True,
        ),
        SearchRequest(
            vector=query_vector_3,
            filter=filter_3,
            limit=5,
            with_payload=True,
        ),
    ]
    single_search_result_1 = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vector_1,
        query_filter=filter_1,
        limit=5,
    )
    single_search_result_2 = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vector_2,
        query_filter=filter_2,
        limit=5,
    )
    single_search_result_3 = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vector_3,
        query_filter=filter_3,
        limit=5,
    )

    batch_search_result = client.search_batch(
        collection_name=COLLECTION_NAME, requests=search_queries
    )

    assert len(batch_search_result) == 3
    assert batch_search_result[0] == single_search_result_1
    assert batch_search_result[1] == single_search_result_2
    assert batch_search_result[2] == single_search_result_3

    recommend_queries = [
        RecommendRequest(
            positive=[1],
            negative=[],
            filter=filter_1,
            limit=5,
            with_payload=True,
        ),
        RecommendRequest(
            positive=[2],
            negative=[],
            filter=filter_2,
            limit=5,
            with_payload=True,
        ),
        RecommendRequest(
            positive=[3],
            negative=[],
            filter=filter_3,
            limit=5,
            with_payload=True,
        ),
    ]
    reco_result_1 = client.recommend(
        collection_name=COLLECTION_NAME, positive=[1], query_filter=filter_1, limit=5
    )
    reco_result_2 = client.recommend(
        collection_name=COLLECTION_NAME, positive=[2], query_filter=filter_2, limit=5
    )
    reco_result_3 = client.recommend(
        collection_name=COLLECTION_NAME, positive=[3], query_filter=filter_3, limit=5
    )

    batch_reco_result = client.recommend_batch(
        collection_name=COLLECTION_NAME, requests=recommend_queries
    )

    assert len(batch_reco_result) == 3
    assert batch_reco_result[0] == reco_result_1
    assert batch_reco_result[1] == reco_result_2
    assert batch_reco_result[2] == reco_result_3

    # ------------------  End of batch queries test ----------------

    assert len(got_points) == 3

    client.delete(
        collection_name=COLLECTION_NAME, wait=True, points_selector=PointIdsList(points=[2, 3])
    )

    got_points = client.retrieve(
        collection_name=COLLECTION_NAME, ids=[1, 2, 3], with_payload=True, with_vectors=True
    )

    assert len(got_points) == 1

    client.upsert(
        collection_name=COLLECTION_NAME,
        wait=True,
        points=[PointStruct(id=2, payload={"hello": "world"}, vector=vectors_2)],
    )

    got_points = client.retrieve(
        collection_name=COLLECTION_NAME, ids=[1, 2, 3], with_payload=True, with_vectors=True
    )

    assert len(got_points) == 2

    client.set_payload(
        collection_name=COLLECTION_NAME, payload={"new_key": 123}, points=[1, 2], wait=True
    )

    got_points = client.retrieve(
        collection_name=COLLECTION_NAME, ids=[1, 2], with_payload=True, with_vectors=True
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
        collection_name=COLLECTION_NAME, ids=[1, 2], with_payload=True, with_vectors=True
    )

    for point in got_points:
        assert not point.payload

    positive = [1, 2, query_vector.tolist()]
    negative = []

    if version is not None and version < "v1.6.0":
        positive = [1, 2]
        negative = []

    recommended_points = client.recommend(
        collection_name=COLLECTION_NAME,
        positive=positive,
        negative=negative,
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
    )

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

    if version is None or (version >= "v1.5.0" or version == "dev"):
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

    client.recreate_collection(
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

    # Many collection update parameters are available since v1.4.0
    version = os.getenv("QDRANT_VERSION")
    if version is None or (version >= "v1.4.0" or version == "dev"):
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

    client.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=DIM, distance=Distance.DOT),
        timeout=TIMEOUT,
    )

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
        collection_name=COLLECTION_NAME, payload={"test2": ["value2", "value3"]}, points=[123]
    )

    # Delete a single point
    client.delete(collection_name=COLLECTION_NAME, points_selector=PointIdsList(points=[123]))


@pytest.mark.parametrize("prefer_grpc", [False, True])
def test_quantization_config(prefer_grpc):
    client = QdrantClient(prefer_grpc=prefer_grpc, timeout=TIMEOUT)

    client.recreate_collection(
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

    _res = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=np.random.rand(DIM),
        search_params=SearchParams(
            quantization=QuantizationSearchParams(
                rescore=True,
            )
        ),
    )


@pytest.mark.parametrize("prefer_grpc", [False, True])
def test_custom_sharding(prefer_grpc):
    version = os.getenv("QDRANT_VERSION")
    if version is not None and version < "v1.7.0":
        pytest.skip("Custom sharding is supported since v1.7.0")

    client = QdrantClient(prefer_grpc=prefer_grpc, timeout=TIMEOUT)

    client.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=DIM, distance=Distance.DOT),
        sharding_method=models.ShardingMethod.CUSTOM,
    )

    client.create_shard_key(collection_name=COLLECTION_NAME, shard_key="cats")

    client.create_shard_key(collection_name=COLLECTION_NAME, shard_key="dogs")

    client.upsert(
        collection_name=COLLECTION_NAME,
        points=[
            PointStruct(id=1, vector=np.random.rand(DIM).tolist(), payload={"name": "Barsik"}),
            PointStruct(id=2, vector=np.random.rand(DIM).tolist(), payload={"name": "Murzik"}),
            PointStruct(id=3, vector=np.random.rand(DIM).tolist(), payload={"name": "Chubais"}),
        ],
        shard_key_selector="cats",
    )

    client.upsert(
        collection_name=COLLECTION_NAME,
        points=[
            PointStruct(id=4, vector=np.random.rand(DIM).tolist(), payload={"name": "Sharik"}),
            PointStruct(id=5, vector=np.random.rand(DIM).tolist(), payload={"name": "Tuzik"}),
            PointStruct(id=6, vector=np.random.rand(DIM).tolist(), payload={"name": "Bobik"}),
        ],
        shard_key_selector="dogs",
    )

    res = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=np.random.rand(DIM),
        shard_key_selector="cats",
    )

    assert len(res) == 3
    for record in res:
        assert record.shard_key == "cats"

    res = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=np.random.rand(DIM),
        shard_key_selector=["cats", "dogs"],
    )

    assert len(res) == 6

    res = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=np.random.rand(DIM),
    )

    assert len(res) == 6

    client.delete_shard_key(collection_name=COLLECTION_NAME, shard_key="dogs")


@pytest.mark.parametrize("prefer_grpc", [False, True])
def test_sparse_vectors(prefer_grpc):
    version = os.getenv("QDRANT_VERSION")
    if version is not None and version < "v1.7.0":
        pytest.skip("Sparse vectors are supported since v1.7.0")

    client = QdrantClient(prefer_grpc=prefer_grpc, timeout=TIMEOUT)

    client.recreate_collection(
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

    result = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=models.NamedSparseVector(
            name="text",
            vector=models.SparseVector(
                indices=[1, 7],
                values=[2.0, 1.0],
            ),
        ),
        with_vectors=["text"],
    )

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

    client.recreate_collection(
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

    client.recreate_collection(
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
def test_conditional_payload_update(prefer_grpc):
    client = QdrantClient(prefer_grpc=prefer_grpc, timeout=TIMEOUT)

    client.recreate_collection(
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


def test_locks():
    client = QdrantClient(timeout=TIMEOUT)

    client.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=DIM, distance=Distance.DOT),
        timeout=TIMEOUT,
    )

    client.lock_storage(reason="testing reason")

    try:
        # Create a single point
        client.upsert(
            collection_name=COLLECTION_NAME,
            points=[
                PointStruct(id=123, payload={"test": "value"}, vector=np.random.rand(DIM).tolist())
            ],
            wait=True,
        )
        assert False, "Should not be able to insert a point when storage is locked"
    except Exception as e:
        assert "testing reason" in str(e)
        pass

    lock_options = client.get_locks()
    assert lock_options.write is True
    assert lock_options.error_message == "testing reason"

    client.unlock_storage()

    # should be fine now
    client.upsert(
        collection_name=COLLECTION_NAME,
        points=[
            PointStruct(id=123, payload={"test": "value"}, vector=np.random.rand(DIM).tolist())
        ],
        wait=True,
    )


@pytest.mark.parametrize("prefer_grpc", [False, True])
def test_empty_vector(prefer_grpc):
    client = QdrantClient(prefer_grpc=prefer_grpc, timeout=TIMEOUT)

    client.recreate_collection(
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


def test_legacy_imports():
    try:
        from qdrant_openapi_client.api.points_api import SyncPointsApi
        from qdrant_openapi_client.exceptions import UnexpectedResponse
        from qdrant_openapi_client.models.models import FieldCondition, Filter
    except ImportError:
        assert False  # can't import, fail


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
    client_http.recreate_collection(
        "test", vectors_config=VectorParams(size=100, distance=Distance.COSINE)
    )
    client_http.close()
    with pytest.raises(qdrant_exceptions.ResponseHandlingException):
        client_http.recreate_collection(
            "test", vectors_config=VectorParams(size=100, distance=Distance.COSINE)
        )
    # endregion

    # region grpc
    client_grpc = QdrantClient(prefer_grpc=True, timeout=TIMEOUT)
    client_grpc.recreate_collection(
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

    client_aio_grpc = QdrantClient(prefer_grpc=True, timeout=TIMEOUT)
    _ = client_aio_grpc.async_grpc_collections
    client_aio_grpc.close()

    client_aio_grpc = QdrantClient(prefer_grpc=True, timeout=TIMEOUT)
    _ = client_aio_grpc.async_grpc_collections
    client_aio_grpc.close(grace=2.0)
    with pytest.raises(RuntimeError):
        client_aio_grpc._client._init_async_grpc_channel()  # prevent reinitializing grpc connection, since
        # http connection is closed

    client_aio_grpc_do_nothing = QdrantClient(prefer_grpc=True, timeout=TIMEOUT)
    client_aio_grpc_do_nothing.close()
    with pytest.raises(
        RuntimeError
    ):  # prevent initializing grpc connection, since http connection is closed
        _ = client_aio_grpc_do_nothing.async_grpc_collections
    # endregion grpc

    # region local
    local_client_in_mem = QdrantClient(":memory:")
    local_client_in_mem.recreate_collection(
        "test", vectors_config=VectorParams(size=100, distance=Distance.COSINE)
    )
    local_client_in_mem.close()

    with tempfile.TemporaryDirectory() as tmpdir:
        path = tmpdir + "/test.db"

        local_client_persist_1 = QdrantClient(path=path)
        local_client_persist_1.recreate_collection(
            "test", vectors_config=VectorParams(size=100, distance=Distance.COSINE)
        )
        local_client_persist_1.close()

        local_client_persist_2 = QdrantClient(path=path)
        local_client_persist_2.recreate_collection(
            "test", vectors_config=VectorParams(size=100, distance=Distance.COSINE)
        )
        local_client_persist_2.close()
    # endregion local


def test_grpc_options():
    client = QdrantClient(prefer_grpc=True)
    assert client._client._grpc_options is None

    client = QdrantClient(prefer_grpc=True, grpc_options={"grpc.max_send_message_length": 3})
    assert client._client._grpc_options == {"grpc.max_send_message_length": 3}

    with pytest.raises(RpcError):
        client.create_collection(
            "grpc_collection",
            vectors_config=models.VectorParams(size=100, distance=models.Distance.COSINE),
        )


if __name__ == "__main__":
    test_qdrant_client_integration()
    test_points_crud()
    test_has_id_condition()
    test_insert_float()
    test_legacy_imports()
