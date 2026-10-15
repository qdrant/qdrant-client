from unittest.mock import patch
from typing import Any

import pytest

from qdrant_client.http.api_client import ApiClient


@pytest.mark.parametrize(
    ("type_", "response"),
    [
        (dict, {"status": "ok"}),
        (None, None),
    ],
)
def test_api_client_request_sync_returns_synchronous_response(
    type_: Any, response: Any
) -> None:
    client = ApiClient("http://localhost:6333")

    try:
        with patch.object(client, "request", return_value=response) as request:
            result = client.request_sync(type_=type_, method="GET", url="/collections")

        assert result == response
        request.assert_called_once_with(type_=type_, method="GET", url="/collections")
    finally:
        client.close()
