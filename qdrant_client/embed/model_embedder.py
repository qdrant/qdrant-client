import os
from copy import deepcopy
from multiprocessing import get_all_start_methods
from typing import Optional, Union, Iterable, Any, Type, get_args

from pydantic import BaseModel

from qdrant_client.http import models
from qdrant_client.embed.common import INFERENCE_OBJECT_TYPES
from qdrant_client.embed.embed_inspector import InspectorEmbed
from qdrant_client.embed.embedder import Embedder
from qdrant_client.embed.models import NumericVector
from qdrant_client.embed.schema_parser import ModelSchemaParser
from qdrant_client.embed.utils import FieldPath
from qdrant_client.fastembed_common import (
    SUPPORTED_EMBEDDING_MODELS,
    SUPPORTED_SPARSE_EMBEDDING_MODELS,
    _LATE_INTERACTION_EMBEDDING_MODELS,
    _IMAGE_EMBEDDING_MODELS,
)
from qdrant_client.parallel_processor import ParallelWorkerPool, Worker
from qdrant_client.uploader.uploader import iter_batch


class ModelEmbedderWorker(Worker):
    def __init__(self, **kwargs: Any):
        self.model_embedder = ModelEmbedder()

    @classmethod
    def start(cls, **kwargs: Any) -> "ModelEmbedderWorker":
        return cls(**kwargs)

    def process(self, items: Iterable[tuple[int, Any]]) -> Iterable[tuple[int, Any]]:
        for idx, batch in items:
            yield idx, list(self.model_embedder.embed_models_batch(batch))


class ModelEmbedder:
    def __init__(self, parser: Optional[ModelSchemaParser] = None):
        self._batch_accumulator: dict[str, list[INFERENCE_OBJECT_TYPES]] = {}
        self._embed_storage: dict[str, list[NumericVector]] = {}
        self._embed_inspector = InspectorEmbed(parser=parser)
        self.embedder = Embedder()

    def embed_models(
        self,
        raw_models: Union[BaseModel, Iterable[BaseModel]],
        is_query: bool = False,
        batch_size: int = 32,
    ) -> Iterable[BaseModel]:
        """Embed raw data fields in models and return models with vectors

            If any of model fields required inference, a deepcopy of a model with computed embeddings is returned,
            otherwise returns original models.
        Args:
            raw_models: Iterable[BaseModel] - models which can contain fields with raw data
            is_query: bool - flag to determine which embed method to use. Defaults to False.
            batch_size: int - batch size for inference
        Returns:
            list[BaseModel]: models with embedded fields
        """
        if isinstance(raw_models, BaseModel):
            raw_models = [raw_models]
        for raw_models_batch in iter_batch(raw_models, batch_size):
            yield from self.embed_models_batch(raw_models_batch, is_query)

    def embed_models_strict(
        self,
        raw_models: Iterable[Union[dict[str, BaseModel], BaseModel]],
        batch_size: int = 32,
        parallel: Optional[int] = None,
    ) -> Iterable[Union[dict[str, BaseModel], BaseModel]]:
        """Embed raw data fields in models and return models with vectors

        Requires every input sequences element to contain raw data fields to inference.
        Does not accept ready vectors.

        Args:
            raw_models: Iterable[BaseModel] - models which contain fields with raw data to inference
            batch_size: int - batch size for inference
            parallel: int - number of parallel processes to use. Defaults to None.

        Returns:
            Iterable[Union[dict[str, BaseModel], BaseModel]]: models with embedded fields
        """
        is_small = False

        if isinstance(raw_models, list):
            if len(raw_models) < batch_size:
                is_small = True

        raw_models_batches = iter_batch(raw_models, batch_size)

        if parallel is None or parallel == 1 or is_small:
            for batch in raw_models_batches:
                yield from self.embed_models_batch(batch)
        else:
            if parallel == 0:
                parallel = os.cpu_count()

            start_method = "forkserver" if "forkserver" in get_all_start_methods() else "spawn"
            assert parallel is not None  # just a mypy complaint
            pool = ParallelWorkerPool(
                num_workers=parallel,
                worker=self._get_worker_class(),
                start_method=start_method,
            )

            for batch in pool.ordered_map(raw_models_batches):
                yield from batch

    def embed_models_batch(
        self,
        raw_models: list[Union[dict[str, BaseModel], BaseModel]],
        is_query: bool = False,
    ) -> Iterable[BaseModel]:
        """Embed a batch of models with raw data fields and return models with vectors

            If any of model fields required inference, a deepcopy of a model with computed embeddings is returned,
            otherwise returns original models.
        Args:
            raw_models: list[Union[dict[str, BaseModel], BaseModel]] - models which can contain fields with raw data
            is_query: bool - flag to determine which embed method to use. Defaults to False.
        Returns:
            Iterable[BaseModel]: models with embedded fields
        """
        for raw_model in raw_models:
            self._process_model(raw_model, is_query=is_query, accumulating=True)

        if not self._batch_accumulator:
            yield from raw_models
        else:
            yield from (
                self._process_model(raw_model, is_query=is_query, accumulating=False)
                for raw_model in raw_models
            )

    def _process_model(
        self,
        model: Union[dict[str, BaseModel], BaseModel],
        paths: Optional[list[FieldPath]] = None,
        is_query: bool = False,
        accumulating: bool = False,
    ) -> Union[dict[str, BaseModel], dict[str, NumericVector], BaseModel, NumericVector]:
        """Embed model's fields requiring inference

        Args:
            model: Qdrant http model containing fields to embed
            paths: Path to fields to embed. E.g. [FieldPath(current="recommend", tail=[FieldPath(current="negative", tail=None)])]
            is_query: Flag to determine which embed method to use. Defaults to False.
            accumulating: Flag to determine if we are accumulating models for batch embedding. Defaults to False.

        Returns:
            A deepcopy of the method with embedded fields
        """

        if isinstance(model, get_args(INFERENCE_OBJECT_TYPES)):
            if accumulating:
                self._accumulate(model)
            else:
                return self._drain_accumulator(model, is_query=is_query)

        if paths is None:
            model = deepcopy(model) if not accumulating else model

        if isinstance(model, dict):
            for key, value in model.items():
                if accumulating:
                    self._process_model(value, paths, accumulating=True)
                else:
                    model[key] = self._process_model(
                        value, paths, is_query=is_query, accumulating=False
                    )
            return model

        paths = paths if paths is not None else self._embed_inspector.inspect(model)

        for path in paths:
            list_model = [model] if not isinstance(model, list) else model
            for item in list_model:
                current_model = getattr(item, path.current, None)
                if current_model is None:
                    continue
                if path.tail:
                    self._process_model(
                        current_model, path.tail, is_query=is_query, accumulating=accumulating
                    )
                else:
                    was_list = isinstance(current_model, list)
                    current_model = current_model if was_list else [current_model]

                    if not accumulating:
                        embeddings = [
                            self._drain_accumulator(data, is_query=is_query)
                            for data in current_model
                        ]
                        if was_list:
                            setattr(item, path.current, embeddings)
                        else:
                            setattr(item, path.current, embeddings[0])
                    else:
                        for data in current_model:
                            self._accumulate(data)
        return model

    def _accumulate(self, data: models.VectorStruct) -> None:
        """Add data to batch accumulator

        Args:
            data: models.VectorStruct - any vector struct data, if inference object types instances in `data` - add them
                to the accumulator, otherwise - do nothing. `InferenceObject` instances are converted to proper types.

        Returns:
            None
        """
        if isinstance(data, dict):
            for value in data.values():
                self._accumulate(value)
            return None

        if isinstance(data, list):
            for value in data:
                if not isinstance(value, get_args(INFERENCE_OBJECT_TYPES)):  # if value is a vector
                    return None
                self._accumulate(value)

        if not isinstance(data, get_args(INFERENCE_OBJECT_TYPES)):
            return None

        data = self._resolve_inference_object(data)
        if data.model not in self._batch_accumulator:
            self._batch_accumulator[data.model] = []
        self._batch_accumulator[data.model].append(data)

    def _drain_accumulator(self, data: models.VectorStruct, is_query: bool) -> models.VectorStruct:
        """Drain accumulator and replaces inference objects with computed embeddings
            It is assumed objects are traversed in the same order as they were added to the accumulator

        Args:
            data: models.VectorStruct - any vector struct data, if inference object types instances in `data` - replace
                them with computed embeddings. If embeddings haven't yet been computed - compute them and then replace
                inference objects.

        Returns:
            models.VectorStruct: data with replaced inference objects
        """
        if isinstance(data, dict):
            for key, value in data.items():
                data[key] = self._drain_accumulator(value, is_query=is_query)
            return data

        if isinstance(data, list):
            for i, value in enumerate(data):
                if not isinstance(value, get_args(INFERENCE_OBJECT_TYPES)):  # if value is vector
                    return data

                data[i] = self._drain_accumulator(value, is_query=is_query)
            return data

        if not isinstance(data, get_args(INFERENCE_OBJECT_TYPES)):
            return data

        if not self._embed_storage or not self._embed_storage.get(data.model, None):
            self._embed_accumulator(is_query=is_query)

        return self._next_embed(data.model)

    def _embed_accumulator(self, is_query: bool = False) -> None:
        """Embed all accumulated objects for all models

        Args:
            is_query: bool - flag to determine which embed method to use. Defaults to False.

        Returns:
            None
        """
        for model_name, objects in self._batch_accumulator.items():
            if model_name not in (
                *SUPPORTED_EMBEDDING_MODELS.keys(),
                *SUPPORTED_SPARSE_EMBEDDING_MODELS.keys(),
                *_LATE_INTERACTION_EMBEDDING_MODELS.keys(),
                *_IMAGE_EMBEDDING_MODELS,
            ):
                raise ValueError(f"{model_name} is not among supported models")

            options = next(iter(objects)).options
            for obj in objects:
                if options != obj.options:
                    raise ValueError(
                        f"Options for {model_name} model should be the same for all objects in one request"
                    )

        for model_name, objects in self._batch_accumulator.items():
            options = next(iter(objects)).options or {}
            if model_name in [
                *SUPPORTED_EMBEDDING_MODELS,
                *SUPPORTED_SPARSE_EMBEDDING_MODELS,
                *_LATE_INTERACTION_EMBEDDING_MODELS,
            ]:
                texts = [obj.text for obj in objects]
                embeddings = [
                    embedding
                    for embedding in self.embedder.embed(
                        texts=texts, is_query=is_query, model_name=model_name, **options
                    )
                ]
            else:
                images = [obj.image for obj in objects]
                embeddings = [
                    embedding
                    for embedding in self.embedder.embed(
                        images=images, is_query=is_query, model_name=model_name, **options
                    )
                ]

            self._embed_storage[model_name] = embeddings
        self._batch_accumulator.clear()

    def _next_embed(self, model_name: str) -> NumericVector:
        """Get next computed embedding from embedded batch

        Args:
            model_name: str - retrieve embedding from the storage by this model name

        Returns:
            NumericVector: computed embedding
        """
        return self._embed_storage[model_name].pop(0)

    @staticmethod
    def _resolve_inference_object(data: models.VectorStruct) -> models.VectorStruct:
        """Resolve inference object into a model

        Args:
            data: models.VectorStruct - data to resolve, if it's an inference object, convert it to a proper type,
                otherwise - keep unchanged

        Returns:
            models.VectorStruct: resolved data
        """

        if not isinstance(data, models.InferenceObject):
            return data

        model_name = data.model
        value = data.object
        options = data.options
        if model_name in (
            *SUPPORTED_EMBEDDING_MODELS.keys(),
            *SUPPORTED_SPARSE_EMBEDDING_MODELS.keys(),
            *_LATE_INTERACTION_EMBEDDING_MODELS.keys(),
        ):
            return models.Document(model=model_name, text=value, options=options)
        if model_name in _IMAGE_EMBEDDING_MODELS:
            return models.Image(model=model_name, image=value, options=options)

        raise ValueError(f"{model_name} is not among supported models")

    @classmethod
    def _get_worker_class(cls) -> Type[ModelEmbedderWorker]:
        return ModelEmbedderWorker
