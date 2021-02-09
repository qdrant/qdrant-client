import inspect

from pydantic import BaseModel
from qdrant_openapi_client.api_client import ApiClient, AsyncApis, SyncApis  # noqa F401
from qdrant_openapi_client.models import models

for model in inspect.getmembers(models, inspect.isclass):
    if model[1].__module__ == "qdrant_openapi_client.models.models":
        model_class = model[1]
        if issubclass(model_class, BaseModel):
            model_class.update_forward_refs()
