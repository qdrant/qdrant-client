import uuid

import numpy as np
import pytest

from qdrant_client import models, QdrantClient

from tests.congruence_tests.test_common import (
    init_local,
    init_remote,
    generate_fixtures,
    compare_client_results,
    compare_collections,
)
from tests.fixtures.payload import one_random_payload_please

COLLECTION_NAME = "test_uuid_input_collection"


@pytest.mark.parametrize("prefer_grpc", (True, False))
def test_uuid_input(prefer_grpc):
    remote_client = init_remote(prefer_grpc=prefer_grpc)
    local_client = init_local()

    text_dim = 100
    code_dim = 10
    fixture_points = generate_fixtures(
        random_ids=True, vectors_sizes={"text": text_dim, "code": code_dim}
    )
    vectors_config = {
        "text": models.VectorParams(size=text_dim, distance=models.Distance.COSINE),
        "code": models.VectorParams(size=code_dim, distance=models.Distance.COSINE),
    }

    for point in fixture_points:
        point.id = uuid.UUID(point.id)
    predefined_id = uuid.uuid4()

    known_point = models.PointStruct(
        id=predefined_id,
        vector={
            "text": np.random.random(text_dim).tolist(),
        },
        payload=one_random_payload_please(101),
    )
    fixture_points.append(known_point)
    for cl in (remote_client, local_client):
        if cl.collection_exists(COLLECTION_NAME):
            cl.delete_collection(COLLECTION_NAME)

        cl.create_collection(
            COLLECTION_NAME,
            vectors_config=vectors_config,
        )
        cl.create_payload_index(COLLECTION_NAME, "field", models.PayloadSchemaType.KEYWORD)
        cl.upsert(COLLECTION_NAME, fixture_points)

    def query_points_uuid(client: QdrantClient):
        return client.query_points(COLLECTION_NAME, query=predefined_id, using="text", limit=1)

    compare_client_results(local_client, remote_client, query_points_uuid)

    random_query = np.random.random(text_dim).tolist()
    id_filter = models.Filter(must=models.HasIdCondition(has_id=[predefined_id]))

    def query_points_filter_uuid(client: QdrantClient):
        return client.query_points(
            COLLECTION_NAME,
            query=random_query,
            using="text",
            query_filter=id_filter,
        )

    compare_client_results(local_client, remote_client, query_points_filter_uuid)

    def query_batch_points_uuid(client: QdrantClient):
        query_batch = [models.QueryRequest(query=predefined_id, using="text")]
        return client.query_batch_points(COLLECTION_NAME, query_batch)

    compare_client_results(local_client, remote_client, query_batch_points_uuid)

    def query_points_groups_uuid(client: QdrantClient):
        return client.query_points_groups(
            COLLECTION_NAME, group_by="field", limit=1, using="text", query=predefined_id
        )

    compare_client_results(local_client, remote_client, query_points_groups_uuid)

    def query_points_groups_uuid_filter(client: QdrantClient):
        return client.query_points_groups(
            COLLECTION_NAME,
            group_by="field",
            limit=1,
            using="text",
            query=np.random.random(text_dim).tolist(),
            query_filter=models.Filter(must=models.HasIdCondition(has_id=[predefined_id])),
        )

    compare_client_results(local_client, remote_client, query_points_groups_uuid_filter)

    def search_matrix_pairs_uuid_filter(client: QdrantClient):
        return client.search_matrix_pairs(COLLECTION_NAME, query_filter=id_filter, using="text")

    compare_client_results(local_client, remote_client, search_matrix_pairs_uuid_filter)

    def search_matrix_offsets_uuid_filter(client: QdrantClient):
        return client.search_matrix_offsets(COLLECTION_NAME, query_filter=id_filter, using="text")

    compare_client_results(local_client, remote_client, search_matrix_offsets_uuid_filter)

    cl.scroll(COLLECTION_NAME, scroll_filter=id_filter)

    def scroll_uuid_filter(client: QdrantClient):
        return client.scroll(COLLECTION_NAME, scroll_filter=id_filter)

    compare_client_results(local_client, remote_client, scroll_uuid_filter)

    def facet_uuid_filter(client: QdrantClient):
        return client.facet(COLLECTION_NAME, key="field", facet_filter=id_filter)

    compare_client_results(local_client, remote_client, facet_uuid_filter)

    def retrieve_uuid_filter(client: QdrantClient):
        return client.retrieve(COLLECTION_NAME, ids=[predefined_id])

    compare_client_results(local_client, remote_client, retrieve_uuid_filter)

    random_vector = np.random.random(text_dim).tolist()
    random_named_vector = {"text": random_vector}

    for cl in (local_client, remote_client):
        cl.update_vectors(
            COLLECTION_NAME,
            points=[models.PointVectors(id=predefined_id, vector=random_named_vector)],
            update_filter=id_filter,
        )

        cl.delete_vectors(COLLECTION_NAME, vectors=["code"], points=id_filter)
        cl.delete_vectors(COLLECTION_NAME, vectors=["code"], points=[predefined_id])
        cl.delete_vectors(
            COLLECTION_NAME, vectors=["code"], points=models.PointIdsList(points=[predefined_id])
        )
        cl.delete_vectors(
            COLLECTION_NAME, vectors=["code"], points=models.FilterSelector(filter=id_filter)
        )

        cl.delete(COLLECTION_NAME, points_selector=id_filter)
        cl.delete(COLLECTION_NAME, points_selector=[predefined_id])
        cl.delete(COLLECTION_NAME, points_selector=models.PointIdsList(points=[predefined_id]))
        cl.delete(COLLECTION_NAME, points_selector=models.FilterSelector(filter=id_filter))

        cl.upsert(
            COLLECTION_NAME,
            points=[models.PointStruct(id=predefined_id, vector=random_named_vector)],
        )
        cl.set_payload(COLLECTION_NAME, payload={"qwe": "rty"}, points=id_filter)
        cl.set_payload(COLLECTION_NAME, payload={"qwe": "rty"}, points=[predefined_id])
        cl.set_payload(
            COLLECTION_NAME,
            payload={"qwe": "rty"},
            points=models.PointIdsList(points=[predefined_id]),
        )
        cl.set_payload(
            COLLECTION_NAME, payload={"qwe": "rty"}, points=models.FilterSelector(filter=id_filter)
        )

        cl.overwrite_payload(COLLECTION_NAME, payload={"qwe": "rty"}, points=id_filter)
        cl.overwrite_payload(COLLECTION_NAME, payload={"qwe": "rty"}, points=[predefined_id])
        cl.overwrite_payload(
            COLLECTION_NAME,
            payload={"qwe": "rty"},
            points=models.PointIdsList(points=[predefined_id]),
        )
        cl.overwrite_payload(
            COLLECTION_NAME, payload={"qwe": "rty"}, points=models.FilterSelector(filter=id_filter)
        )

        cl.delete_payload(COLLECTION_NAME, keys=["qwe"], points=id_filter)
        cl.delete_payload(COLLECTION_NAME, keys=["qwe"], points=[predefined_id])
        cl.delete_payload(
            COLLECTION_NAME, keys=["qwe"], points=models.PointIdsList(points=[predefined_id])
        )
        cl.delete_payload(
            COLLECTION_NAME, keys=["qwe"], points=models.FilterSelector(filter=id_filter)
        )

        cl.clear_payload(COLLECTION_NAME, points_selector=id_filter)
        cl.clear_payload(COLLECTION_NAME, points_selector=[predefined_id])
        cl.clear_payload(
            COLLECTION_NAME, points_selector=models.PointIdsList(points=[predefined_id])
        )
        cl.clear_payload(COLLECTION_NAME, points_selector=models.FilterSelector(filter=id_filter))

        cl.upload_collection(
            COLLECTION_NAME,
            ids=[predefined_id],
            vectors={"text": np.array([random_vector])},
        )

        cl.batch_update_points(
            COLLECTION_NAME,
            update_operations=[
                models.UpsertOperation(
                    upsert=models.PointsBatch(
                        batch=models.Batch(
                            ids=[predefined_id],
                            vectors={"text": [random_vector]},
                        )
                    )
                ),
                models.UpsertOperation(
                    upsert=models.PointsList(
                        points=[
                            models.PointStruct(
                                id=predefined_id,
                                vector=random_named_vector,
                            )
                        ]
                    )
                ),
                models.SetPayloadOperation(
                    set_payload=models.SetPayload(payload={"qwe": "rty"}, filter=id_filter)
                ),
                models.SetPayloadOperation(
                    set_payload=models.SetPayload(payload={"qwe": "rty"}, points=[predefined_id])
                ),
                models.OverwritePayloadOperation(
                    overwrite_payload=models.SetPayload(payload={"qwe": "rty"}, filter=id_filter)
                ),
                models.OverwritePayloadOperation(
                    overwrite_payload=models.SetPayload(
                        payload={"qwe": "rty"}, points=[predefined_id]
                    )
                ),
                models.DeletePayloadOperation(
                    delete_payload=models.DeletePayload(keys=["qwe"], filter=id_filter)
                ),
                models.DeletePayloadOperation(
                    delete_payload=models.DeletePayload(keys=["qwe"], points=[predefined_id])
                ),
                models.ClearPayloadOperation(
                    clear_payload=models.PointIdsList(points=[predefined_id])
                ),
                models.ClearPayloadOperation(
                    clear_payload=models.FilterSelector(filter=id_filter)
                ),
                models.UpdateVectorsOperation(
                    update_vectors=models.UpdateVectors(
                        points=[
                            models.PointVectors(
                                id=predefined_id,
                                vector=random_named_vector,
                            )
                        ]
                    ),
                ),
                models.UpdateVectorsOperation(
                    update_vectors=models.UpdateVectors(
                        points=[
                            models.PointVectors(
                                id=predefined_id,
                                vector=random_named_vector,
                            )
                        ],
                        update_filter=id_filter,
                    ),
                ),
                models.DeleteVectorsOperation(
                    delete_vectors=models.DeleteVectors(filter=id_filter, vector=["code"])
                ),
                models.DeleteVectorsOperation(
                    delete_vectors=models.DeleteVectors(points=[predefined_id], vector=["code"])
                ),
                models.DeleteOperation(delete=models.PointIdsList(points=[predefined_id])),
                models.DeleteOperation(delete=models.FilterSelector(filter=id_filter)),
            ],
        )

    compare_collections(
        local_client, remote_client, num_vectors=1000, collection_name=COLLECTION_NAME
    )
