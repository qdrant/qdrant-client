from copy import copy
from typing import List, Type, Dict, Union, Any, Set, Optional

from pydantic import BaseModel

from qdrant_client.embed.utils import Path, convert_paths


class ModelSchemaParser:
    """Model schema parser. Parses json schemas to retrieve paths to objects requiring inference.

    The parser is stateful, it accumulates the results of parsing in its internal structures.

    Attributes:
        _defs: definitions extracted from json schemas
        _recursive_refs: set of recursive refs found in the processed schemas, e.g.:
            {"Filter", "Prefetch"}
        _not_document_recursive_refs: predefined time-consuming recursive refs which don't have inference objects, e.g.:
            {"Filter"}
        _doc_recursive_paths: set of recursive refs which have inference objects, e.g.:
            {"Prefetch"}
        _cache: cache of string paths for models containing objects for inference, e.g.:
            {"Prefetch": ['prefetch.query', 'prefetch.query.context.negative', ...]}
        path_cache: cache of Path objects for models containing objects for inference, e.g.:
            {
                 "Prefetch": [
                     Path(
                         current="prefetch",
                         tail=[
                             Path(
                                 current="query",
                                 tail=[
                                     Path(
                                         current="recommend",
                                         tail=[
                                             Path(current="negative", tail=None),
                                             Path(current="positive", tail=None),
                                         ],
                                     ),
                                     ...,
                                 ],
                             ),
                         ],
                     )
                 ]
            }
        name_recursive_ref_mapping: mapping of model field names to ref names, e.g.:
            {"prefetch": "Prefetch"}
    """

    def __init__(self) -> None:
        self._defs: Dict[str, Union[Dict[str, Any], List[Dict[str, Any]]]] = {}
        self._cache: Dict[str, List[str]] = {}
        self._recursive_refs: Set[str] = set()
        self._not_document_recursive_refs: Set[str] = {"Filter"}
        self.path_cache: Dict[str, List[Path]] = {}
        self._doc_recursive_paths: Set[str] = set()
        self.name_recursive_ref_mapping: Dict[str, str] = {}

    def _replace_refs(
        self,
        schema: Union[Dict[str, Any], List[Dict[str, Any]]],
        parent: Optional[str] = None,
        seen_refs: Optional[set] = None,
    ) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """Replace refs in schema with their definitions

        Args:
            schema: schema to parse
            parent: previous level key
            seen_refs: set of seen refs to spot recursive paths

        Returns:
            schema with replaced refs
        """
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
                return self._replace_refs(schema=self._defs[def_key], seen_refs=copy(seen_refs))

            schemes = {}
            for k, v in schema.items():
                if isinstance(v, dict) and "properties" in v:
                    schemes[k] = self._replace_refs(schema=v, parent=k, seen_refs=copy(seen_refs))
                else:
                    schemes[k] = self._replace_refs(
                        schema=v, parent=parent, seen_refs=copy(seen_refs)
                    )
            return schemes
        elif isinstance(schema, list):
            return [
                self._replace_refs(schema=item, parent=parent, seen_refs=copy(seen_refs))  # type: ignore
                for item in schema
            ]
        else:
            return schema

    def _find_document_paths(
        self,
        schema: Union[Dict[str, Any], List[Dict[str, Any]]],
        current_path: str = "",
        after_properties: bool = False,
        seen_refs: Optional[Set] = None,
    ) -> List[str]:
        """Read a schema and find paths to objects requiring inference

        Populates model fields names to ref names mapping

        Args:
            schema: schema to parse
            current_path: current path in the schema
            after_properties: flag indicating if the current path is after "properties" key
            seen_refs: set of seen refs to spot recursive paths

        Returns:
            List of string dot separated paths to objects requiring inference
        """
        document_paths: List[str] = []
        seen_recursive_refs = seen_refs if seen_refs is not None else set()

        parts = current_path.split(".")
        if len(parts) != len(set(parts)):  # check for recursive paths
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

            if after_properties:  # field name seen in pydantic models comes after "properties" key
                if current_path:
                    new_path = f"{current_path}.{key}"
                else:
                    new_path = key
            else:
                new_path = current_path

            if isinstance(value, dict):
                document_paths.extend(
                    self._find_document_paths(
                        value, new_path, key == "properties", seen_refs=seen_recursive_refs
                    )
                )
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        document_paths.extend(
                            self._find_document_paths(
                                item,
                                new_path,
                                key == "properties",
                                seen_refs=seen_recursive_refs,
                            )
                        )

        return sorted(set(document_paths))

    def parse_model(self, model: Type[BaseModel]) -> List[str]:
        """Parse model schema to retrieve paths to objects requiring inference.

        Checks model json schema, extracts definitions and finds paths to objects requiring inference.
        No parsing happens if model has already been processed.

        Args:
            model: model to parse

        Returns:
            List of string dot separated paths to objects requiring inference
        """
        model_name = model.__name__
        if model_name in self._cache:
            return self._cache[model_name]

        schema = model.model_json_schema()
        self._defs.update(schema.get("$defs", {}))

        defs = self._replace_refs(schema)
        self._cache[model_name] = self._find_document_paths(defs)

        for ref in self._recursive_refs:
            if ref in self._not_document_recursive_refs or ref in self._doc_recursive_paths:
                continue

            if self._find_document_paths(self._defs[ref]):
                self._doc_recursive_paths.add(ref)
            else:
                self._not_document_recursive_refs.add(ref)

        # convert str paths to Path objects which group path parts and reduce the time of the traversal
        self.path_cache = {model: convert_paths(paths) for model, paths in self._cache.items()}
        return self._cache[model.__name__]
