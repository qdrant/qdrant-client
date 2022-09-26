import logging

import qdrant_client.http.api as api
import qdrant_client.http.exceptions as exceptions
import qdrant_client.http.api_client as api_client

logging.warning("Use of deprecated import: use `qdrant_client.http` instead of `qdrant_openapi_client`")
