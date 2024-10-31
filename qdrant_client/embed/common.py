from typing import Set, Tuple

from qdrant_client.http import models

INFERENCE_OBJECT_NAMES: Set[str] = {"Document", "Image"}
INFERENCE_OBJECT_TYPES: Tuple = (models.Document, models.Image)
