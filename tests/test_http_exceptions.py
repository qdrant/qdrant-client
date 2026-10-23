import pickle

import httpx

from qdrant_client.common.client_exceptions import ResourceExhaustedResponse
from qdrant_client.http.exceptions import UnexpectedResponse


def test_exceptions_survive_pickling() -> None:
    # Errors raised in a worker process are pickled to reach the parent process.
    # The client creates both of these with keyword arguments.
    unexpected = UnexpectedResponse.for_response(httpx.Response(404, content=b"Not found"))
    restored = pickle.loads(pickle.dumps(unexpected))
    assert restored.status_code == 404
    assert str(restored) == str(unexpected)

    exhausted = ResourceExhaustedResponse(message="Too many requests", retry_after_s=5)
    restored = pickle.loads(pickle.dumps(exhausted))
    assert restored.retry_after_s == 5
    assert str(restored) == str(exhausted)
