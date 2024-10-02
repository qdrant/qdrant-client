from copy import copy
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
            self.parser.check_model(points.__class__)
            return self._inspect_model(points)

        elif isinstance(points, Iterable):
            for point in points:
                if isinstance(point, BaseModel):
                    self.parser.check_model(point.__class__)
                    if self._inspect_model(point):
                        return True
        return False

    def _inspect_model(self, mod: BaseModel, paths: Optional[List[Path]] = None) -> bool:
        if isinstance(mod, models.Document):
            return True

        if paths is None:
            paths = self.parser.path_cache.get(mod.__class__.__name__, None)

        if paths is None:
            return False

        for path in paths:
            type_found = self._inspect_inner_models(
                mod, path.current, path.tail if path.tail else []
            )
            if type_found:
                return True
        return False

    def _inspect_inner_models(
        self, original_model: BaseModel, current: str, tail: List[Path]
    ) -> bool:
        def inspect_recursive(member: BaseModel) -> bool:
            recursive_paths = []
            for field in member.model_fields_set:
                if field in self.parser.name_recursive_ref_mapping:
                    mapped_field = self.parser.name_recursive_ref_mapping[field]
                    if mapped_field not in self.parser.path_cache:
                        self.parser.check_model(getattr(models, mapped_field))
                    recursive_paths.extend(self.parser.path_cache[mapped_field])

            if recursive_paths:
                found = self._inspect_model(member, copy(recursive_paths))
                if found:
                    return True

            return False

        model = getattr(original_model, current, None)
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
                if not isinstance(current_model, BaseModel):
                    continue

                if isinstance(current_model, models.Document):
                    return True

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
                    if not isinstance(current_model, BaseModel):
                        continue

                    if isinstance(current_model, models.Document):
                        return True

                    found_type = inspect_recursive(current_model)
                    if found_type:
                        return True

                for next_path in tail:
                    for current_model in model:
                        found_type = self._inspect_inner_models(
                            current_model,
                            next_path.current,
                            next_path.tail if next_path.tail else [],
                        )
                        if found_type:
                            return True
        return False
