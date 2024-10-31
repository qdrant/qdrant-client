from qdrant_client import models

INFERENCE_OBJECT_NAMES = {"Document", "Image"}
INFERENCE_OBJECT_TYPES = (models.Document, models.Image)
