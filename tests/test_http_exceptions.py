from httpx import Request, Response

from qdrant_client.http.exceptions import UnexpectedResponse


def test_unexpected_response_includes_request_url():
    request = Request("POST", "https://example.com/api/collections?wait=true")
    response = Response(404, request=request, content=b"not found")

    exception = UnexpectedResponse.for_response(response)

    assert exception.url == "https://example.com/api/collections?wait=true"
    assert "Request URL: https://example.com/api/collections?wait=true" in str(
        exception
    )


def test_unexpected_response_without_request_omits_url():
    response = Response(404, content=b"not found")

    exception = UnexpectedResponse.for_response(response)

    assert exception.url is None
    assert "Request URL:" not in str(exception)
