from typing import Union, List, Optional, Iterable

from pydantic import BaseModel

from qdrant_client.embed.schema_parser import OpenApiSchemaParser
from qdrant_client.embed.utils import Path
from qdrant_client.http import models


class Inspector:
    def __init__(self, parser: Optional[OpenApiSchemaParser] = None):
        self.parser = parser if parser else OpenApiSchemaParser()

    def inspect(self, points: Union[Iterable[BaseModel], BaseModel, List]) -> bool:
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

    def _inspect_model(self, model: BaseModel, paths: Optional[List[Path]] = None) -> bool:
        if isinstance(model, models.Document):
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
        self, original_model: BaseModel, current_path: str, tail: List[Path]
    ) -> bool:
        def inspect_recursive(member: BaseModel) -> bool:
            recursive_paths = []
            for field_name in member.model_fields_set:
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

        if isinstance(model, models.Document):
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
                if isinstance(current_model, models.Document):
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
                    if isinstance(current_model, models.Document):
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
