import importlib.util
import sys
from pathlib import Path
from typing import Union

from pydantic import BaseModel

from qdrant_client import models
from qdrant_client.embed.schema_parser import ModelSchemaParser
from types import ModuleType


def dynamic_import(file_path: Union[str, Path], module_name: str) -> ModuleType:
    # Create a module spec from the file path
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None:
        raise ImportError(f"Cannot create a spec for module '{module_name}' at '{file_path}'")

    # Create a new module based on the spec
    module = importlib.util.module_from_spec(spec)

    # Execute the module to load its contents
    spec.loader.exec_module(module)  # type: ignore

    # Optionally, add the module to sys.modules to access it globally
    sys.modules[module_name] = module
    return module


if __name__ == "__main__":
    parser = ModelSchemaParser()
    for model_name in dir(models):
        if not model_name[0].isupper():
            continue

        model = getattr(models, model_name)
        if not isinstance(model, type):
            continue

        if not issubclass(model, BaseModel) or model == BaseModel:
            continue

        if "extra" not in model.model_config or model.model_config["extra"] != "forbid":  # type: ignore
            continue

        parser.parse_model(model)

    current_path = Path(__name__)
    file_path = str(
        current_path.parent.parent / "qdrant_client" / "embed" / "_inspection_cache.py"
    )
    parser._persist(file_path)
    module_name = "_inspections_cache"

    # Import the newly created file dynamically
    _inspections_cache = dynamic_import(file_path, module_name)
    result = _inspections_cache.INCLUDED_RECURSIVE_REFS

    assert parser.name_recursive_ref_mapping == _inspections_cache.NAME_RECURSIVE_REF_MAPPING
    assert parser._defs == _inspections_cache.DEFS
    assert parser._recursive_refs == set(_inspections_cache.RECURSIVE_REFS)
    assert parser._included_recursive_refs == set(_inspections_cache.INCLUDED_RECURSIVE_REFS)
    assert parser._excluded_recursive_refs == set(_inspections_cache.EXCLUDED_RECURSIVE_REFS)
