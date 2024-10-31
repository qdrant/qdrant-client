from typing import Set, Type, Tuple

from qdrant_client.http import models

INFERENCE_OBJECT_NAMES: Set[str] = {"Document", "Image"}
INFERENCE_OBJECT_TYPES: Tuple[Type[models.Document], Type[models.Image]] = (
    models.Document,
    models.Image,
)
