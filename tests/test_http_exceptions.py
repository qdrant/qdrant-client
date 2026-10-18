import copy
import multiprocessing
import pickle
from concurrent.futures import ProcessPoolExecutor
from typing import Callable

import httpx
import pytest

from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse


@pytest.fixture(params=["response", "kwargs", "positional"])
def error(request: pytest.FixtureRequest) -> UnexpectedResponse:
    response = httpx.Response(
        404,
        content=b'{"status":{"error":"Collection not found"}}',
        headers=[("x-request-id", "request-1"), ("x-detail", "first"), ("x-detail", "second")],
    )
    if request.param == "response":
        result = UnexpectedResponse.for_response(response)
        assert isinstance(result, UnexpectedResponse)
        return result
    if request.param == "kwargs":
        return UnexpectedResponse(
            status_code=response.status_code,
            reason_phrase=response.reason_phrase,
            content=response.content,
            headers=response.headers,
        )
    return UnexpectedResponse(
        response.status_code, response.reason_phrase, response.content, response.headers
    )


def pickle_round_trip(error: UnexpectedResponse) -> UnexpectedResponse:
    restored = pickle.loads(pickle.dumps(error))
    assert isinstance(restored, UnexpectedResponse)
    return restored


@pytest.mark.parametrize(
    "round_trip", [copy.copy, copy.deepcopy, pickle_round_trip], ids=["copy", "deepcopy", "pickle"]
)
def test_unexpected_response_round_trip(
    error: UnexpectedResponse, round_trip: Callable[[UnexpectedResponse], UnexpectedResponse]
) -> None:
    restored = round_trip(error)

    assert type(restored) is UnexpectedResponse
    assert restored is not error
    assert restored.status_code == error.status_code
    assert restored.reason_phrase == error.reason_phrase
    assert restored.content == error.content
    assert restored.headers.raw == error.headers.raw
    assert restored.structured() == error.structured()
    assert str(restored) == str(error)


def missing_collection(request: httpx.Request) -> httpx.Response:
    return httpx.Response(404, content=b"Collection not found", request=request)


def request_missing_collection() -> None:
    client = QdrantClient(
        url="http://qdrant.invalid:6333",
        check_compatibility=False,
        transport=httpx.MockTransport(missing_collection),
    )
    try:
        client.get_collection("missing")
    finally:
        client.close()


def test_client_error_survives_process_pool() -> None:
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=1, mp_context=context) as pool:
        future = pool.submit(request_missing_collection)
        with pytest.raises(UnexpectedResponse) as exception_info:
            future.result(timeout=20)

        assert exception_info.value.status_code == 404
        assert exception_info.value.content == b"Collection not found"
        assert pool.submit(abs, -1).result(timeout=20) == 1
