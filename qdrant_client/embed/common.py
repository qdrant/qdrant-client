from typing import Type

from qdrant_client.http import models

INFERENCE_OBJECT_NAMES: set[str] = {"Document", "Image", "InferenceObject"}
INFERENCE_OBJECT_TYPES: tuple[
    Type[models.Document], Type[models.Image], Type[models.InferenceObject]
] = (models.Document, models.Image, models.InferenceObject)
