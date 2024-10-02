from copy import copy
from typing import List, Type, Dict

from pydantic import BaseModel

from qdrant_client.embed.utils import Path, convert_paths


class OpenApiSchemaParser:
    def __init__(self):
        self._defs = {}
        self._cache = {}
        self._recursive_refs = set()
        self._not_document_recursive_refs = {
            "Filter"
        }  # initialize with a known time-consuming model
        self.path_cache = {}
        self._doc_recursive_paths = set()
        self.name_recursive_ref_mapping = {}

    def convert_to_paths(self) -> Dict[str, List[Path]]:
        path_cache = {model: convert_paths(paths) for model, paths in self._cache.items()}
        self.path_cache = path_cache
        return path_cache

    def replace_refs(self, schema, parent=None, seen_refs=None):
        parent = parent if parent else None
        seen_refs = seen_refs if seen_refs else set()

        if isinstance(schema, dict):
            if "$ref" in schema:
                ref_path = schema["$ref"]
                def_key = ref_path.split("/")[-1]
                if def_key == parent or def_key in seen_refs:
                    self._recursive_refs.add(def_key)
                    return schema
                seen_refs.add(def_key)
                return self.replace_refs(schema=self._defs[def_key], seen_refs=copy(seen_refs))

            schemes = {}
            for k, v in schema.items():
                if isinstance(v, dict) and "properties" in v:
                    schemes[k] = self.replace_refs(schema=v, parent=k, seen_refs=copy(seen_refs))
                else:
                    schemes[k] = self.replace_refs(
                        schema=v, parent=parent, seen_refs=copy(seen_refs)
                    )
            return schemes
        elif isinstance(schema, list):
            return [
                self.replace_refs(schema=item, parent=parent, seen_refs=copy(seen_refs))
                for item in schema
            ]
        else:
            return schema

    def find_document_paths(
        self, schema, current_path="", after_properties=False, seen_refs=None
    ) -> List[str]:
        document_paths = []
        seen_recursive_refs = seen_refs if seen_refs is not None else set()

        parts = current_path.split(".")
        if len(parts) != len(set(parts)):
            return document_paths

        if not isinstance(schema, dict):
            return document_paths

        if "title" in schema and schema["title"] == "Document":
            document_paths.append(current_path)
            return document_paths

        for key, value in schema.items():
            if key == "$defs":
                continue

            if key == "$ref":
                model_name = value.split("/")[-1]

                value = self._defs[model_name]
                if model_name in self._not_document_recursive_refs:
                    continue

                if model_name in self._recursive_refs:
                    seen_recursive_refs.add(model_name)
                    self.name_recursive_ref_mapping[current_path.split(".")[-1]] = model_name

            if after_properties:
                if current_path:
                    new_path = f"{current_path}.{key}"
                else:
                    new_path = key
            else:
                new_path = current_path

            if isinstance(value, dict):
                document_paths.extend(
                    self.find_document_paths(
                        value, new_path, key == "properties", seen_refs=seen_recursive_refs
                    )
                )
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        document_paths.extend(
                            self.find_document_paths(
                                item,
                                new_path,
                                key == "properties",
                                seen_refs=seen_recursive_refs,
                            )
                        )

        return sorted(set(document_paths))

    def check_model(self, model: Type[BaseModel]) -> List[str]:
        model_name = model.__name__

        if model_name in self._cache:
            return self._cache[model_name]
        schema = model.model_json_schema()
        self._defs.update(schema.get("$defs", {}))

        defs = self.replace_refs(schema)
        self._cache[model_name] = self.find_document_paths(defs, after_properties=False)

        for ref in self._recursive_refs:
            if ref in self._not_document_recursive_refs or ref in self._doc_recursive_paths:
                continue

            if self.find_document_paths(self._defs[ref]):
                self._doc_recursive_paths.add(ref)
            else:
                self._not_document_recursive_refs.add(ref)

        self.convert_to_paths()
        return self._cache[model.__name__]
