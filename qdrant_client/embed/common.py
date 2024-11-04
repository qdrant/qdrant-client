from typing import Set, Type, Tuple

from qdrant_client.http import models

INFERENCE_OBJECT_NAMES: Set[str] = {"Document", "Image", "InferenceObject"}
INFERENCE_OBJECT_TYPES: Tuple[
    Type[models.Document], Type[models.Image], Type[models.InferenceObject]
] = (models.Document, models.Image, models.InferenceObject)
