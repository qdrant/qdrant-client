from typing import Any

from pydantic import BaseModel, Field

from qdrant_client.conversions.common_types import SparseVector
from qdrant_client.http import models

_FASTEMBED_CLASSES: dict[str, Any] = {}
_FASTEMBED_CHECKED = False


def _load_fastembed_classes() -> None:
    global _FASTEMBED_CHECKED
    if _FASTEMBED_CHECKED:
        return
    _FASTEMBED_CHECKED = True
    try:
        from fastembed import (
            ImageEmbedding,
            LateInteractionMultimodalEmbedding,
            LateInteractionTextEmbedding,
            SparseTextEmbedding,
            TextEmbedding,
        )
        from fastembed.common import ImageInput, OnnxProvider

        _FASTEMBED_CLASSES.update(
            {
                "TextEmbedding": TextEmbedding,
                "SparseTextEmbedding": SparseTextEmbedding,
                "ImageEmbedding": ImageEmbedding,
                "LateInteractionTextEmbedding": LateInteractionTextEmbedding,
                "LateInteractionMultimodalEmbedding": LateInteractionMultimodalEmbedding,
                "OnnxProvider": OnnxProvider,
                "ImageInput": ImageInput,
            }
        )
    except ImportError:
        pass


def _fastembed_class(name: str) -> Any:
    _load_fastembed_classes()
    return _FASTEMBED_CLASSES.get(name)


class QueryResponse(BaseModel, extra="forbid"):  # type: ignore
    id: str | int
    embedding: list[float] | None
    sparse_embedding: SparseVector | None = Field(default=None)
    metadata: dict[str, Any]
    document: str
    score: float


class FastEmbedMisc:
    IS_INSTALLED: bool = False
    _TEXT_MODELS: set[str] = set()
    _IMAGE_MODELS: set[str] = set()
    _LATE_INTERACTION_TEXT_MODELS: set[str] = set()
    _LATE_INTERACTION_MULTIMODAL_MODELS: set[str] = set()
    _SPARSE_MODELS: set[str] = set()

    @classmethod
    def is_installed(cls) -> bool:
        if cls.IS_INSTALLED:
            return cls.IS_INSTALLED

        try:
            import fastembed  # noqa: F401, PLC0415
        except ImportError:
            cls.IS_INSTALLED = False
        else:
            cls.IS_INSTALLED = True

        return cls.IS_INSTALLED

    @classmethod
    def import_fastembed(cls) -> None:
        if cls.is_installed():
            return

        raise ImportError(
            "fastembed is not installed."
            " Please install it to compute embedding for document implicitly with `pip install fastembed`."
        )

    @classmethod
    def list_text_models(cls) -> dict[str, tuple[int, models.Distance]]:
        text_embedding = _fastembed_class("TextEmbedding")
        if text_embedding is None:
            return {}
        return {
            model["model"]: (model["dim"], models.Distance.COSINE)
            for model in text_embedding.list_supported_models()
        }

    @classmethod
    def list_image_models(cls) -> dict[str, tuple[int, models.Distance]]:
        image_embedding = _fastembed_class("ImageEmbedding")
        if image_embedding is None:
            return {}
        return {
            model["model"]: (model["dim"], models.Distance.COSINE)
            for model in image_embedding.list_supported_models()
        }

    @classmethod
    def list_late_interaction_text_models(cls) -> dict[str, tuple[int, models.Distance]]:
        late_interaction = _fastembed_class("LateInteractionTextEmbedding")
        if late_interaction is None:
            return {}
        return {
            model["model"]: (model["dim"], models.Distance.COSINE)
            for model in late_interaction.list_supported_models()
        }

    @classmethod
    def list_late_interaction_multimodal_models(cls) -> dict[str, tuple[int, models.Distance]]:
        late_interaction = _fastembed_class("LateInteractionMultimodalEmbedding")
        if late_interaction is None:
            return {}
        return {
            model["model"]: (model["dim"], models.Distance.COSINE)
            for model in late_interaction.list_supported_models()
        }

    @classmethod
    def list_sparse_models(cls) -> dict[str, dict[str, Any]]:
        sparse_embedding = _fastembed_class("SparseTextEmbedding")
        if sparse_embedding is None:
            return {}
        descriptions: dict[str, dict[str, Any]] = {}
        for description in sparse_embedding.list_supported_models():
            descriptions[description.pop("model")] = description
        return descriptions

    @classmethod
    def is_supported_text_model(cls, model_name: str) -> bool:
        if model_name.lower() in cls._TEXT_MODELS:
            return True
        cls._TEXT_MODELS = {model.lower() for model in cls.list_text_models()}
        if model_name.lower() in cls._TEXT_MODELS:
            return True
        return False

    @classmethod
    def is_supported_image_model(cls, model_name: str) -> bool:
        if model_name.lower() in cls._IMAGE_MODELS:
            return True
        cls._IMAGE_MODELS = {model.lower() for model in cls.list_image_models()}
        if model_name.lower() in cls._IMAGE_MODELS:
            return True
        return False

    @classmethod
    def is_supported_late_interaction_text_model(cls, model_name: str) -> bool:
        if model_name.lower() in cls._LATE_INTERACTION_TEXT_MODELS:
            return True
        cls._LATE_INTERACTION_TEXT_MODELS = {
            model.lower() for model in cls.list_late_interaction_text_models()
        }
        if model_name.lower() in cls._LATE_INTERACTION_TEXT_MODELS:
            return True
        return False

    @classmethod
    def is_supported_late_interaction_multimodal_model(cls, model_name: str) -> bool:
        if model_name.lower() in cls._LATE_INTERACTION_MULTIMODAL_MODELS:
            return True
        cls._LATE_INTERACTION_MULTIMODAL_MODELS = {
            model.lower() for model in cls.list_late_interaction_multimodal_models()
        }
        if model_name.lower() in cls._LATE_INTERACTION_MULTIMODAL_MODELS:
            return True
        return False

    @classmethod
    def is_supported_sparse_model(cls, model_name: str) -> bool:
        if model_name.lower() in cls._SPARSE_MODELS:
            return True
        cls._SPARSE_MODELS = {model.lower() for model in cls.list_sparse_models()}
        if model_name.lower() in cls._SPARSE_MODELS:
            return True
        return False


# region deprecated
# prefer using methods builtin into QdrantClient, e.g. list_supported_text_models, list_supported_idf_models, etc.


def _supported_embedding_models() -> dict[str, tuple[int, models.Distance]]:
    return FastEmbedMisc.list_text_models()


def _supported_sparse_embedding_models() -> dict[str, dict[str, Any]]:
    sparse_embedding = _fastembed_class("SparseTextEmbedding")
    if sparse_embedding is None:
        return {}
    return {model["model"]: model for model in sparse_embedding.list_supported_models()}


def _idf_embedding_models() -> set[str]:
    sparse_embedding = _fastembed_class("SparseTextEmbedding")
    if sparse_embedding is None:
        return set()
    return {
        model_config["model"]
        for model_config in sparse_embedding.list_supported_models()
        if model_config.get("requires_idf", None)
    }


def _late_interaction_embedding_models() -> dict[str, tuple[int, models.Distance]]:
    return FastEmbedMisc.list_late_interaction_text_models()


def _image_embedding_models() -> dict[str, tuple[int, models.Distance]]:
    return FastEmbedMisc.list_image_models()


def _late_interaction_multimodal_embedding_models() -> dict[str, tuple[int, models.Distance]]:
    return FastEmbedMisc.list_late_interaction_multimodal_models()


_LAZY_MODULE_ATTRS: dict[str, Any] = {
    "TextEmbedding": lambda: _fastembed_class("TextEmbedding"),
    "SparseTextEmbedding": lambda: _fastembed_class("SparseTextEmbedding"),
    "ImageEmbedding": lambda: _fastembed_class("ImageEmbedding"),
    "LateInteractionTextEmbedding": lambda: _fastembed_class("LateInteractionTextEmbedding"),
    "LateInteractionMultimodalEmbedding": lambda: _fastembed_class(
        "LateInteractionMultimodalEmbedding"
    ),
    "OnnxProvider": lambda: _fastembed_class("OnnxProvider"),
    "ImageInput": lambda: _fastembed_class("ImageInput"),
    "SUPPORTED_EMBEDDING_MODELS": _supported_embedding_models,
    "SUPPORTED_SPARSE_EMBEDDING_MODELS": _supported_sparse_embedding_models,
    "IDF_EMBEDDING_MODELS": _idf_embedding_models,
    "_LATE_INTERACTION_EMBEDDING_MODELS": _late_interaction_embedding_models,
    "_IMAGE_EMBEDDING_MODELS": _image_embedding_models,
    "_LATE_INTERACTION_MULTIMODAL_EMBEDDING_MODELS": _late_interaction_multimodal_embedding_models,
}


def __getattr__(name: str) -> object:
    factory = _LAZY_MODULE_ATTRS.get(name)
    if factory is not None:
        result = factory()
        globals()[name] = result
        return result
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)

# endregion
