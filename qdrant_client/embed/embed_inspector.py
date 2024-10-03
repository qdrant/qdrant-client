from copy import copy
from typing import Union, List, Optional, Iterable

from pydantic import BaseModel

from qdrant_client._pydantic_compat import model_fields_set
from qdrant_client.embed.schema_parser import OpenApiSchemaParser

from qdrant_client.embed.utils import convert_paths, Path
from qdrant_client.http import models


class InspectorEmbed:
    def __init__(self, parser: Optional[OpenApiSchemaParser] = None):
        self.parser = parser if parser else OpenApiSchemaParser()

    def inspect(self, points: Union[Iterable[BaseModel], BaseModel, List]) -> List[Path]:
        paths = []
        points = [points] if not isinstance(points, list) else points
        for point in points:
            if isinstance(point, BaseModel):
                self.parser.check_model(point.__class__)
                paths.extend(self._inspect_model(point))

        paths = sorted(set(paths))

        return convert_paths(paths)

    def _inspect_model(
        self, mod: BaseModel, paths: Optional[List[Path]] = None, accum: Optional[str] = None
    ) -> List[str]:
        if paths is None:
            paths = self.parser.path_cache.get(mod.__class__.__name__, None)

        if paths is None:
            return []

        found_paths = []
        for path in paths:
            found_paths.extend(
                self._inspect_inner_models(
                    mod, path.current, path.tail if path.tail else [], accum
                )
            )
        return found_paths

    def _inspect_inner_models(
        self,
        original_model: BaseModel,
        current: str,
        tail: List[Path],
        accum: Optional[str] = None,
    ) -> List[str]:
        found_paths = []
        if accum is None:
            accum = current
        else:
            accum += f".{current}"

        def inspect_recursive(member: BaseModel, accum: str) -> List[str]:
            recursive_paths = []
            for field in model_fields_set(member):
                if field in self.parser.name_recursive_ref_mapping:
                    mapped_field = self.parser.name_recursive_ref_mapping[field]
                    if mapped_field not in self.parser.path_cache:
                        self.parser.check_model(getattr(models, mapped_field))
                    recursive_paths.extend(self.parser.path_cache[mapped_field])

            return self._inspect_model(member, copy(recursive_paths), accum)

        model = getattr(original_model, current, None)
        if model is None:
            return []

        if isinstance(model, models.Document):
            return [accum]

        if isinstance(model, BaseModel):
            found_paths.extend(inspect_recursive(model, accum))

            for next_path in tail:
                found_paths.extend(
                    self._inspect_inner_models(
                        model, next_path.current, next_path.tail if next_path.tail else [], accum
                    )
                )

            return found_paths

        elif isinstance(model, list):
            for current_model in model:
                if not isinstance(current_model, BaseModel):
                    continue

                if isinstance(current_model, models.Document):
                    found_paths.append(accum)

                found_paths.extend(inspect_recursive(current_model, accum))

            for next_path in tail:
                for current_model in model:
                    found_paths.extend(
                        self._inspect_inner_models(
                            current_model,
                            next_path.current,
                            next_path.tail if next_path.tail else [],
                            accum,
                        )
                    )
            return found_paths

        elif isinstance(model, dict):
            found_paths = []
            for key, values in model.items():
                values = [values] if not isinstance(values, list) else values
                for current_model in values:
                    if not isinstance(current_model, BaseModel):
                        continue

                    if isinstance(current_model, models.Document):
                        found_paths.append(accum)

                    found_paths.extend(inspect_recursive(current_model, accum))

                for next_path in tail:
                    for current_model in model:
                        found_paths.extend(
                            self._inspect_inner_models(
                                current_model,
                                next_path.current,
                                next_path.tail if next_path.tail else [],
                                accum,
                            )
                        )
        return found_paths
