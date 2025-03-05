from typing import Union, Optional, Iterable

from pydantic import BaseModel

from qdrant_client._pydantic_compat import model_fields_set
from qdrant_client.embed.common import INFERENCE_OBJECT_TYPES
from qdrant_client.embed.schema_parser import ModelSchemaParser
from qdrant_client.embed.utils import FieldPath
from qdrant_client.http import models


class Inspector:
    """Inspector which tries to find at least one occurrence of an object requiring inference

    Inspector is stateful and accumulates parsed model schemes in its parser.

    Attributes:
        parser: ModelSchemaParser instance to inspect model json schemas
    """

    def __init__(self, parser: Optional[ModelSchemaParser] = None) -> None:
        self.parser = ModelSchemaParser() if parser is None else parser

    def inspect(self, points: Union[Iterable[BaseModel], BaseModel]) -> bool:
        """Looks for at least one occurrence of an object requiring inference in the received models

        Args:
            points: models to inspect

        Returns:
            True if at least one object requiring inference is found, False otherwise
        """
        if isinstance(points, BaseModel):
            self.parser.parse_model(points.__class__)
            return self._inspect_model(points)

        elif isinstance(points, Iterable):
            for point in points:
                if isinstance(point, BaseModel):
                    self.parser.parse_model(point.__class__)
                    if self._inspect_model(point):
                        return True
        return False

    def _inspect_model(self, model: BaseModel, paths: Optional[list[FieldPath]] = None) -> bool:
        if isinstance(model, INFERENCE_OBJECT_TYPES):
            return True

        paths = (
            self.parser.path_cache.get(model.__class__.__name__, []) if paths is None else paths
        )

        for path in paths:
            type_found = self._inspect_inner_models(
                model, path.current, path.tail if path.tail else []
            )
            if type_found:
                return True
        return False

    def _inspect_inner_models(
        self, original_model: BaseModel, current_path: str, tail: list[FieldPath]
    ) -> bool:
        def inspect_recursive(member: BaseModel) -> bool:
            recursive_paths = []
            for field_name in model_fields_set(member):
                if field_name in self.parser.name_recursive_ref_mapping:
                    mapped_model_name = self.parser.name_recursive_ref_mapping[field_name]
                    if mapped_model_name not in self.parser.path_cache:
                        # We found a model, which we haven't parsed yet, need to fill up the cache with its paths
                        self.parser.parse_model(getattr(models, mapped_model_name))
                    recursive_paths.extend(self.parser.path_cache[mapped_model_name])

            if recursive_paths:
                found = self._inspect_model(member, recursive_paths)
                if found:
                    return True

            return False

        model = getattr(original_model, current_path, None)
        if model is None:
            return False

        if isinstance(model, INFERENCE_OBJECT_TYPES):
            return True

        if isinstance(model, BaseModel):
            type_found = inspect_recursive(model)
            if type_found:
                return True

            for next_path in tail:
                type_found = self._inspect_inner_models(
                    model, next_path.current, next_path.tail if next_path.tail else []
                )
                if type_found:
                    return True
            return False

        elif isinstance(model, list):
            for current_model in model:
                if isinstance(current_model, INFERENCE_OBJECT_TYPES):
                    return True

                if not isinstance(current_model, BaseModel):
                    continue

                type_found = inspect_recursive(current_model)
                if type_found:
                    return True

            for next_path in tail:
                for current_model in model:
                    type_found = self._inspect_inner_models(
                        current_model, next_path.current, next_path.tail if next_path.tail else []
                    )
                    if type_found:
                        return True
            return False

        elif isinstance(model, dict):
            for key, values in model.items():
                values = [values] if not isinstance(values, list) else values
                for current_model in values:
                    if isinstance(current_model, INFERENCE_OBJECT_TYPES):
                        return True

                    if not isinstance(current_model, BaseModel):
                        continue

                    found_type = inspect_recursive(current_model)
                    if found_type:
                        return True

                for next_path in tail:
                    for current_model in values:
                        found_type = self._inspect_inner_models(
                            current_model,
                            next_path.current,
                            next_path.tail if next_path.tail else [],
                        )
                        if found_type:
                            return True
        return False
